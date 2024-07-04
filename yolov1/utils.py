import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
from PIL import Image


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]  # (N, 1)
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def non_max_suppression(
    bboxes, iou_threshold, threshold, box_format="corners"
):
    """
    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU)
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert isinstance(bboxes, list)

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def mean_average_precision(
    pred_boxes,
    true_boxes,
    iou_threshold=0.5,
    box_format="midpoint",
    num_classes=2,
):
    """
    Calculates mean average precision

    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes

    Returns:
        float: mAP value across all classes given a specific IoU threshold
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)


def plot_image(image, boxes, output_path):
    """Plots predicted bounding boxes on the image and saves the output to a file."""
    # Ensure the image is a NumPy array
    im = np.array(image)
    height, width, _ = im.shape

    # Create a figure without displaying it
    fig, ax = plt.subplots(1, figsize=(width / 100, height / 100), dpi=100)

    # Display the image
    ax.imshow(im)

    # Iterate over each box
    for box in boxes:
        # Extract the box coordinates (assuming x_center, y_center, width, height format)
        x_center, y_center, box_width, box_height = box[2:]
        # Calculate the upper-left coordinates
        upper_left_x = x_center - box_width / 2
        upper_left_y = y_center - box_height / 2
        # Create a Rectangle patch
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box_width * width,
            box_height * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)

    # Remove axes for a clean image
    plt.axis("off")

    # Save the plot to a file
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def get_bboxes(
    loader,
    model,
    iou_threshold,
    threshold,
    pred_format="cells",
    box_format="midpoint",
    device="cuda",
):
    all_pred_boxes = []
    all_true_boxes = []

    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0

    for batch_idx, (z, x, labels, w) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )

            # if batch_idx == 0 and idx == 0:
            #    plot_image(x[idx].permute(1,2,0).to("cpu"), nms_boxes)
            #    print(nms_boxes)

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes


def convert_cellboxes(predictions, S=7):
    """
    Converts bounding boxes output from YOLO with an image split size of S
    into entire image ratios rather than relative to cell ratios.

    Args:
    - predictions (torch.Tensor): The raw predictions from the YOLO model.
    - S (int): The grid size the image is divided into (default 7).

    Returns:
    - torch.Tensor: Converted bounding box predictions.
    """
    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]

    # Ensure predictions are reshaped correctly to [batch_size, S, S, 12]
    predictions = predictions.reshape(batch_size, S, S, 12)

    # Split predictions into individual bounding boxes and their scores
    bboxes1 = predictions[..., 3:7]  # Bounding box 1 (x, y, w, h)
    bboxes2 = predictions[..., 8:12]  # Bounding box 2 (x, y, w, h)

    # Scores of the bounding boxes
    scores = torch.cat(
        (predictions[..., 2].unsqueeze(0), predictions[..., 7].unsqueeze(0)),
        dim=0,
    )

    # Determine the best bounding box based on the highest score
    best_box = scores.argmax(0).unsqueeze(-1)

    # Select the best bounding box
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2

    # Calculate cell indices to convert coordinates to the entire image
    cell_indices = torch.arange(S).repeat(batch_size, S, 1).unsqueeze(-1)

    # Convert x and y coordinates to be relative to the entire image
    x = 1 / S * (best_boxes[..., 0:1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))

    # Normalize width and height
    w = 1 / S * best_boxes[..., 2:3]
    h = 1 / S * best_boxes[..., 3:4]

    # Concatenate the converted coordinates
    converted_bboxes = torch.cat((x, y, w, h), dim=-1)

    # Get the predicted class (argmax of the first two elements) and confidence score
    predicted_class = predictions[..., :2].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(
        predictions[..., 2], predictions[..., 7]
    ).unsqueeze(-1)

    # Concatenate class, confidence, and bounding boxes into the final format
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds


def cellboxes_to_boxes(out, S=7):
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S * S):
            bboxes.append(
                [x.item() for x in converted_pred[ex_idx, bbox_idx, :]]
            )
        all_bboxes.append(bboxes)

    return all_bboxes


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    print(f"Saving checkpoint to {filename}")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer=None):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])


def upscale_boxes(boxes, scale_factor):
    """
    Upscales bounding boxes based on a scaling factor.

    Args:
        boxes: List of bounding boxes in xywh format (x, y, w, h)
        scale_factor: Float representing the upscaling factor

    Returns:
        upscaled_boxes: List of upscaled bounding boxes in xywh format
    """
    upscaled_boxes = []
    for box in boxes:
        c, p, x, y, w, h = box
        upscaled_boxes.append(
            [
                c,
                p,
                x * scale_factor,
                y * scale_factor,
                w * scale_factor,
                h * scale_factor,
            ]
        )
    return upscaled_boxes


def process_images_with_bboxes(
    image_tensor, predictions, output_size=(224, 224), device="cuda"
):
    """
    Processes images with bounding boxes: crops the regions and resizes them.

    Args:
    - image_tensor (torch.Tensor): The input image tensor of shape (3, H, W).
    - predictions (torch.Tensor): A tensor of predictions with shape (N, 6),
                                  where each row is (class, probability, x_center, y_center, width, height).
    - output_size (tuple): The desired output size of the cropped images (default is (224, 224)).
    - device (str): The device to perform operations on ('cuda' or 'cpu').

    Returns:
    - tuple: A tuple containing:
        - list: A list of torch.Tensor objects (cropped and resized images).
        - torch.Tensor: A tensor of class labels corresponding to each cropped image.
    """
    # Ensure the input tensor is of type float
    image_tensor = image_tensor.float() / 255.0

    _, height, width = image_tensor.shape
    cropped_images = []
    cropped_labels = []

    for prediction in predictions:
        class_label, prob, x_center, y_center, box_width, box_height = (
            prediction
        )

        if x_center > 0 and y_center > 0 and box_width > 0 and box_height > 0:
            # Calculate bounding box coordinates
            left = int((x_center - box_width / 2) * width)
            top = int((y_center - box_height / 2) * height)
            right = int((x_center + box_width / 2) * width)
            bottom = int((y_center + box_height / 2) * height)

            # Ensure coordinates are within image bounds
            left = max(0, left)
            top = max(0, top)
            right = min(width, right)
            bottom = min(height, bottom)

            # Crop the image
            cropped_img = image_tensor[:, top:bottom, left:right]

            if cropped_img.numel() == 0:
                # Skip if the cropped image is empty
                continue

            # Resize the cropped image
            resized_img = torch.nn.functional.interpolate(
                cropped_img.unsqueeze(0),
                size=output_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            cropped_images.append(resized_img)
            cropped_labels.append(class_label.item())

    if cropped_labels:
        # Convert cropped_labels to a tensor
        cropped_labels_tensor = torch.tensor(cropped_labels, device=device)
        return cropped_images, cropped_labels_tensor
    else:
        return [], torch.tensor([], device=device)


def filter_bboxes(boxes, bboxes2_list, threshold=0.8):
    # Convert bboxes2 list to a tensor
    bboxes2 = torch.tensor(
        bboxes2_list, device=boxes.device
    )  # Assuming boxes is already a tensor

    # List to store filtered bounding boxes
    bboxes = []

    for box in boxes:
        # Expand box to match the shape of bboxes2 for broadcasting
        expanded_box = box.unsqueeze(0).expand(bboxes2.size(0), -1)

        # Calculate IoU between the current box and all bboxes2
        iou = calculate_iou(expanded_box, bboxes2)

        # Filter bboxes2 with IoU >= threshold
        mask = iou >= threshold

        # Select the filtered bboxes2
        filtered_bboxes2 = bboxes2[mask]

        if len(filtered_bboxes2) > 0:
            # Replace the class label in filtered_bboxes2 with the class label from the current box
            filtered_bboxes2[:, 0] = box[0]
            bboxes.append(filtered_bboxes2)

    if len(bboxes) > 0:
        # Concatenate all filtered bboxes into a single tensor
        bboxes = torch.cat(bboxes, dim=0)
    else:
        # Create an empty tensor on the same device as bboxes2
        bboxes = torch.tensor([]).to(bboxes2.device)

    return bboxes


def calculate_iou(box1, box2):
    # box1: [class_label, x, y, w, h] (single box)
    # box2: [class_label, confidence, x, y, w, h] (multiple boxes)

    # Extract coordinates for box1
    x1_min, y1_min = (
        box1[..., 1] - box1[..., 3] / 2,
        box1[..., 2] - box1[..., 4] / 2,
    )
    x1_max, y1_max = (
        box1[..., 1] + box1[..., 3] / 2,
        box1[..., 2] + box1[..., 4] / 2,
    )

    # Extract coordinates for box2
    x2_min, y2_min = (
        box2[..., 2] - box2[..., 4] / 2,
        box2[..., 3] - box2[..., 5] / 2,
    )
    x2_max, y2_max = (
        box2[..., 2] + box2[..., 4] / 2,
        box2[..., 3] + box2[..., 5] / 2,
    )

    # Calculate intersection coordinates
    inter_min_x = torch.max(x1_min, x2_min)
    inter_min_y = torch.max(y1_min, y2_min)
    inter_max_x = torch.min(x1_max, x2_max)
    inter_max_y = torch.min(y1_max, y2_max)

    # Calculate intersection area
    inter_area = torch.clamp(inter_max_x - inter_min_x, min=0) * torch.clamp(
        inter_max_y - inter_min_y, min=0
    )

    # Calculate union area
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area

    # Calculate IoU
    iou = inter_area / union_area

    return iou
