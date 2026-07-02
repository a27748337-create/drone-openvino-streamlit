from ultralytics import YOLO

model = YOLO("drone_yolo11n_best_openvino_model")

results = model.predict(
    source="test_images",
    imgsz=640,
    conf=0.25,
    device="cpu",
    save=True
)

print("추론 완료")