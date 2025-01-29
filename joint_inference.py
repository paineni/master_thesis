from ultralytics import YOLO
import torch
from ultralytics.engine.trainer import get_vgg16_model
from torchvision import transforms
import fiftyone as fo
from PIL import Image
import torch.nn.functional as F
from typing import Union
import os
import tempfile
import mlflow
from matplotlib.figure import Figure
import pandas as pd
import numpy as np

v8_model = YOLO("saved_joint_weights/weights_640/train33/weights/best.pt")
vgg_model, _ = get_vgg16_model()
vgg_model_path = "saved_joint_weights/weights_640/vgg_16_16.pth"
vgg_model.load_state_dict(torch.load(vgg_model_path), strict=False)
vgg_model.to("cuda")
vgg_model.eval()
preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)


def get_class_prediction(cropped_image, model):
    image = preprocess(cropped_image)
    image = image.unsqueeze(0).to(
        "cuda"
    )  # Add batch dimension and move to device
    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(
            output, dim=1
        )  # Apply softmax to get probabilities
        confidence, predicted = torch.max(probabilities, 1)
    return predicted.item(), confidence.item()


def xyxy2normalized_topleftxywh(
    xmin: Union[float, np.ndarray],
    ymin: Union[float, np.ndarray],
    xmax: Union[float, np.ndarray],
    ymax: Union[float, np.ndarray],
    bg_w: Union[int, np.ndarray],
    bg_h: Union[int, np.ndarray],
):
    """
    :return:
    """
    w = xmax - xmin
    h = ymax - ymin
    x_topleft_norm = xmin / bg_w
    y_topleft_norm = ymin / bg_h
    w_norm = w / bg_w
    h_norm = h / bg_h

    return x_topleft_norm, y_topleft_norm, w_norm, h_norm


ds = fo.Dataset.from_dir(
    dataset_dir="prepared",
    dataset_type=fo.types.FiftyOneDataset,
)
for sample in ds:
    detections = []
    v8_result = v8_model.predict(sample.filepath, imgsz=640)
    img = v8_result[0].orig_img
    preds = v8_result[0].boxes.data
    for i, box in enumerate(preds):
        x1, y1, x2, y2 = box[:4].int().tolist()
        cropped_np = img[y1:y2, x1:x2, :]
        cropped_image = Image.fromarray(
            cropped_np
        )  # Convert to PIL Image for preprocessing
        class_id, cnf = get_class_prediction(cropped_image, vgg_model)
        x_topleft_norm, y_topleft_norm, w_norm, h_norm = (
            xyxy2normalized_topleftxywh(x1, y1, x2, y2, 5184, 3888)
        )
        detections.append(
            fo.Detection(
                label=ds.default_classes[int(class_id)],
                bounding_box=[x_topleft_norm, y_topleft_norm, w_norm, h_norm],
                confidence=cnf,
            )
        )
        sample["predictions"] = fo.Detections(detections=detections)
        sample.save()


artifacts_path = "atrifacts"
splits = ["train", "valid", "test"]
mlflow.set_experiment("joint_detection")
mlflow.start_run()
for split_tag in splits:
    view = ds.match_tags([split_tag])

    # Evaluate the objects in the `predictions`
    # field with respect to the
    # objects in the `ground_truth` field
    eval_key = f"eval_predictions_{split_tag}"
    results = view.evaluate_detections(
        "predictions",
        gt_field="ground_truth",
        eval_key=eval_key,
        compute_mAP=True,
        classes=ds.default_classes,
        missing="background",
        classwise=False,
    )
    # whether to consider objects with different label
    # values as always non-overlapping (True) or to compute IoUs
    # for all objects regardless of label (False)

    # the COCO mAP evaluator averages the mAP
    # over 10 IoU thresholds from 0.5 to 0.95
    # with a step size of 0.05 (AP@[0.5:0.05:0.95])
    # To be found in the source of fiftyone.
    # "https://github.com/voxel51/fiftyone/blob/"
    # "acf3a8f886505d852903e320d057057813261993/fiftyone/"
    # "utils/eval/coco.py#L91"
    mAP = results.mAP()
    mlflow.log_metric(key=f"mAP_{split_tag}", value=mAP)
    print(f"mAP@[0.5:0.05:0.95] {split_tag} : " + str(mAP))
    classwise_ap_df = pd.DataFrame(columns=["Label", "AP@[0.5:0.05:0.95]"])
    label_values = ds.count_values("ground_truth.detections.label")
    for label in ds.default_classes:
        class_AP = results.mAP([label])
        print(f"AP@[0.5:0.05:0.95] of {split_tag} ({label}): " + str(class_AP))
        classwise_ap_df = classwise_ap_df._append(
            {
                "Label": label,
                "AP@[0.5:0.05:0.95]": class_AP,
                "support": label_values[label],
            },
            ignore_index=True,
        )

    with tempfile.TemporaryDirectory() as tmpworkdir:
        temp_csv_path = os.path.join(
            tmpworkdir, f"{split_tag}_classwise_ap_values.csv"
        )
        classwise_ap_df.to_csv(temp_csv_path, index=False)
        mlflow.log_artifact(temp_csv_path)

    results.print_report()
    report = results.report()
    weighted_avg_precision = report["weighted avg"]["precision"]
    weighted_avg_recall = report["weighted avg"]["recall"]
    mlflow.log_metric(
        key=f"weigthed avg precision - {split_tag}",
        value=weighted_avg_precision,
    )
    mlflow.log_metric(
        key=f"weighted avg recall - {split_tag}",
        value=weighted_avg_recall,
    )

    # Print some statistics about the total TP/FP/FN counts
    mean_tp = view.sum(f"{eval_key}_tp")
    mean_fp = view.sum(f"{eval_key}_fp")
    mean_fn = view.sum(f"{eval_key}_fn")
    print(f"TP ({split_tag}): {mean_tp}")
    print(f"FP ({split_tag}): {mean_fp}")
    print(f"FN ({split_tag}): {mean_fn}")

    mlflow.log_metric(key=f"mean_TP_{split_tag}", value=mean_tp)
    mlflow.log_metric(key=f"mean_FP_{split_tag}", value=mean_fp)
    mlflow.log_metric(key=f"mean_FN_{split_tag}", value=mean_fn)

    class_counts = view.count_values("predictions.detections.label")

    pr_curve_path = os.path.join(artifacts_path, f"PR_curve_{split_tag}.png")
    pr_curve_plot: Figure = results.plot_pr_curves(
        classes=list(class_counts.keys()),
        backend="matplotlib",
        style="dark_background",
    )
    pr_curve_plot.savefig(pr_curve_path, dpi=250)
    mlflow.log_artifact(pr_curve_path)

    conf_mat_path = os.path.join(
        artifacts_path, f"confusion_matrix_{split_tag}.png"
    )
    conf_mat_plot: Figure = results.plot_confusion_matrix(backend="matplotlib")
    conf_mat_plot.savefig(conf_mat_path, dpi=250)
    mlflow.log_artifact(conf_mat_path)

mlflow.end_run()
ds.save()

ds.export("original_ds1", fo.types.FiftyOneDataset)
