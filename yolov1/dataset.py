"""
Creates a Pytorch dataset to load the Pascal VOC dataset
"""

import numpy as np
import torch
import os
from PIL import Image


class BeanDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        img_dir,
        label_dir,
        S=7,
        B=2,
        C=2,
        transform=None,
    ):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C
        self.image_files = [
            f
            for f in os.listdir(self.img_dir)
            if f.endswith((".png", ".jpg", ".jpeg"))
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_file = self.image_files[index]
        label_path = os.path.join(
            self.label_dir, image_file.split(".", 1)[0] + ".txt"
        )
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]

                boxes.append([class_label, x, y, width, height])

        img_path = os.path.join(self.img_dir, image_file)
        image = Image.open(img_path).convert("RGB")
        boxes = torch.tensor(boxes)
        original_image_np = np.array(image)
        original_image_tensor = torch.from_numpy(
            original_image_np.transpose((2, 0, 1))
        )

        if self.transform:
            # image = self.transform(image)
            image, boxes = self.transform(image, boxes)

        # Convert To Cells
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            # i,j represents the cell row and cell column
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i

            """
            Calculating the width and height of cell of bounding box,
            relative to the cell is done by the following, with
            width as the example:

            width_pixels = (width*self.image_width)
            cell_pixels = (self.image_width)

            Then to find the width relative to the cell is simply:
            width_pixels/cell_pixels, simplification leads to the
            formulas below.
            """
            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )

            # If no object already found for specific cell i,j
            # Note: This means we restrict to ONE object
            # per cell!
            if label_matrix[i, j, 2] == 0:
                # Set that there exists an object
                label_matrix[i, j, 2] = 1

                # Box coordinates
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                label_matrix[i, j, 3:7] = box_coordinates

                # Set one hot encoding for class_label
                label_matrix[i, j, class_label] = 1

        return (original_image_tensor, image, label_matrix, boxes)
