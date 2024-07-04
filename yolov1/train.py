"""
Main file for training Yolo model on Pascal VOC dataset

"""

import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
import torchvision.models as models
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import BeanDataset
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
    process_images_with_bboxes,
    filter_bboxes,
)
from loss import YoloLoss

seed = 42
torch.manual_seed(seed)

vgg16 = models.vgg16(pretrained=True)

# Freeze convolutional layers
for param in vgg16.parameters():
    param.requires_grad = False

# Modify the classifier
num_features = vgg16.classifier[6].in_features
vgg16.classifier[6] = nn.Sequential(
    nn.Linear(num_features, 4096),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 2),
)  # Adjust based on your number of classes


# Define loss function and optimizer


# Hyperparameters etc.
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = (
    16  # 64 in original paper but I don't have that much vram, grad accum?
)
WEIGHT_DECAY = 0
EPOCHS = 10
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "model.pth.tar"
LOAD_MODEL_FILE2 = "vgg16.pth.tar"
TRAIN_IMG_DIR = "/home/paineni/MasterThesis/yolov8/images/train"
TRAIN_LABEL_DIR = "/home/paineni/MasterThesis/yolov8/labels/train"
TEST_IMG_DIR = "/home/paineni/MasterThesis/yolov8/images/test"
TEST_LABEL_DIR = "/home/paineni/MasterThesis/yolov8/labels/test"


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


transform = Compose(
    [
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
    ]
)


def train_fn(
    train_loader, model, optimizer, loss_fn, criterion, DEVICE, vgg16
):
    vgg16.train()
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (z, x, y, w) in enumerate(loop):
        x, y, z, w = x.to(DEVICE), y.to(DEVICE), z.to(DEVICE), w.to(DEVICE)

        optimizer.zero_grad()
        out = model(x)
        yolo_loss = loss_fn(out, y)
        cum_class_loss = 0

        # Post-processing for bounding boxes
        for idx in range(len(x)):
            bboxes = cellboxes_to_boxes(out)
            bboxes2 = non_max_suppression(
                bboxes[idx],
                iou_threshold=0.5,
                threshold=0.4,
                box_format="midpoint",
            )
            if bboxes2 is None or len(bboxes2) == 0:
                continue
            else:
                filtered_bboxes = filter_bboxes(w[idx], bboxes2)
            if filtered_bboxes is None or len(filtered_bboxes) == 0:
                continue
            cropped_images, labels = process_images_with_bboxes(
                z[idx],
                filtered_bboxes,
                output_size=(224, 224),
                device=DEVICE,
            )
            if cropped_images is None:
                classifier_loss = 0
            else:
                # Normalize images
                normalize = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
                normalized_images = torch.stack(
                    [normalize(img) for img in cropped_images]
                ).to(DEVICE)

                # Forward pass through VGG-16
                classifier_outputs = vgg16(normalized_images)
                classifier_loss = criterion(
                    classifier_outputs, labels.to(torch.long)
                )

            cum_class_loss += classifier_loss
        print(f"cum_class_loss: {cum_class_loss}")
        # Total loss
        loss = yolo_loss + cum_class_loss
        mean_loss.append(loss.item())

        # Backward pass and optimization step
        loss.backward()

        optimizer.step()

        # Update progress bar
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")


def main():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=2).to(DEVICE)
    optimizer = optim.Adam(
        [
            {"params": model.parameters()},  # parameters of YOLO model
            {"params": vgg16.classifier.parameters()},
        ],  # parameters of VGG-16 classifier
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    loss_fn = YoloLoss()
    criterion = nn.CrossEntropyLoss()
    vgg16.to(DEVICE)

    if LOAD_MODEL:
        # Load YOLO model and optimizer
        load_checkpoint(LOAD_MODEL_FILE, model, optimizer)

        # Load VGG-16 model (no need to load optimizer again)
        load_checkpoint(LOAD_MODEL_FILE2, vgg16)

    train_dataset = BeanDataset(
        transform=transform,
        img_dir=TRAIN_IMG_DIR,
        label_dir=TRAIN_LABEL_DIR,
    )

    test_dataset = BeanDataset(
        transform=transform,
        img_dir=TEST_IMG_DIR,
        label_dir=TEST_LABEL_DIR,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    for epoch in range(EPOCHS):
        # for x, y in test_loader:
        #     x = x.to(DEVICE)
        #     for idx in range(1):

        #         bboxes = cellboxes_to_boxes(model(x))

        #         bboxes = non_max_suppression(
        #             bboxes[idx],
        #             iou_threshold=0.5,
        #             threshold=0.4,
        #             box_format="midpoint",
        #         )
        #         print(bboxes)
        #         plot_image(
        #             x[idx].permute(1, 2, 0).to("cpu"),
        #             bboxes,
        #             "/home/paineni/MasterThesis/yolov8/images/output.jpg",
        #         )

        #     sys.exit()

        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print(f"Train mAP: {mean_avg_prec}")

        if mean_avg_prec > 0.1:
            checkpoint_yolo = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            checkpoint_vgg16 = {
                "state_dict": vgg16.state_dict(),
            }
            save_checkpoint(checkpoint_yolo, filename=LOAD_MODEL_FILE)
            save_checkpoint(checkpoint_vgg16, filename=LOAD_MODEL_FILE2)
            import time

            time.sleep(10)

        train_fn(
            train_loader, model, optimizer, loss_fn, criterion, DEVICE, vgg16
        )


if __name__ == "__main__":
    main()
