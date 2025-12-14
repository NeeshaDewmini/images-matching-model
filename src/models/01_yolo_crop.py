from ultralytics import YOLO
import os
from PIL import Image

# Load YOLOv8 model (pretrained on COCO)
model = YOLO("yolov8n.pt")  # nano model, fast for PoC

input_dir = "data/raw_images"
output_dir = "data/cropped_images"

os.makedirs(output_dir, exist_ok=True)

for class_folder in os.listdir(input_dir):
    class_path = os.path.join(input_dir, class_folder)
    if not os.path.isdir(class_path):
        continue

    out_class_path = os.path.join(output_dir, class_folder)
    os.makedirs(out_class_path, exist_ok=True)

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        results = model.predict(img_path, save=False)
        # Crop the first detected object (you can loop if multiple objects)
        if len(results) > 0 and len(results[0].boxes) > 0:
            crop = results[0].boxes.xyxy[0].cpu().numpy().astype(int)
            img = Image.open(img_path)
            x1, y1, x2, y2 = crop
            cropped_img = img.crop((x1, y1, x2, y2))
            cropped_img.save(os.path.join(out_class_path, img_name))
