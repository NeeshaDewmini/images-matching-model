import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os

# -------------------------
# CONFIG
# -------------------------
INPUT_DIR = r"data\raw_images"   # contains class subfolders
OUTPUT_FILE = "train_features.npy"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# -------------------------
# MODEL
# -------------------------
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = torch.nn.Identity()  # feature extractor
model.eval()
model.to(DEVICE)

# -------------------------
# TRANSFORM
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------
# COLLECT IMAGES (RECURSIVE)
# -------------------------
image_paths = []

for root, dirs, files in os.walk(INPUT_DIR):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            image_paths.append(os.path.join(root, file))

print("🖼️ Images found for feature extraction:", len(image_paths))

if len(image_paths) == 0:
    raise RuntimeError("❌ No images found. Check INPUT_DIR structure.")

# -------------------------
# EXTRACT FEATURES
# -------------------------
features_dict = {}

for img_path in image_paths:
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"⚠️ Skipping {img_path}: {e}")
        continue

    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        feat = model(x).cpu().numpy().flatten()

    features_dict[img_path] = feat

# -------------------------
# SAVE FEATURES
# -------------------------
np.save(OUTPUT_FILE, features_dict)
print(f"✅ Feature extraction complete! Saved {len(features_dict)} features.")
