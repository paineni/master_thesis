from ultralytics import YOLO

# from EMA import (
#     EMA_region_begin,
#     EMA_region_define,
#     EMA_region_end,
#     EMA_init,
#     EMA_finalize,
# )

# EMA_init()
# region = EMA_region_define("abc")
# EMA_region_begin(region)
# Load a model
model = YOLO(
    "temp/yolo_pretreined_models/yolov8n.pt"
)  # pretrained YOLOv8n model

results = model.train(
    data="roboflow_yolov8/dataset.yaml",
    epochs=400,
    imgsz=640,
)
# EMA_region_end(region)
# EMA_finalize()
