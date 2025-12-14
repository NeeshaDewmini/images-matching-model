import os
import random
from shutil import copy2

raw_dir = "data/raw_images"
train_dir = "data/train"
val_dir = "data/val"
test_dir = "data/test"

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

images = [f for f in os.listdir(raw_dir) if f.lower().endswith(('.jpg', '.png'))]
random.shuffle(images)

train_split = int(0.7 * len(images))
val_split = int(0.85 * len(images))

for i, img in enumerate(images):
    src = os.path.join(raw_dir, img)
    if i < train_split:
        dst = os.path.join(train_dir, img)
    elif i < val_split:
        dst = os.path.join(val_dir, img)
    else:
        dst = os.path.join(test_dir, img)
    copy2(src, dst)

print("✅ Split done!")
