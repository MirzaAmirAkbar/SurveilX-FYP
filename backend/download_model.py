from ultralytics import YOLO

model = YOLO(
    "https://huggingface.co/Zcket/gun_dtct/resolve/main/best.pt"
)

print(model.names)
