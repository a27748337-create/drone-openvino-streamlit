from ultralytics import YOLO

model = YOLO("drone_yolo11n_best.pt")

model.export(
    format="openvino",
    imgsz=640,
    half=False,
    dynamic=False
)