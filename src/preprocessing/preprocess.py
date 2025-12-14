import cv2
import os
import numpy as np

def preprocess_image(input_path, output_path, size=(224, 224)):
    """
    Preprocess a single image:
    - Resize
    - Convert to RGB
    - Apply denoising / smoothing
    """
    # Read image
    img = cv2.imread(input_path)
    if img is None:
        print(f"[Warning] Image not found or cannot be read: {input_path}")
        return

    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Apply denoising / smoothing
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    # Resize
    img = cv2.resize(img, size)

    # Save processed image
    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def preprocess_folder(input_dir, output_dir, size=(224, 224)):
    """
    Preprocess all images in a folder
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png'))]

    if not images:
        print(f"No images found in {input_dir}")
        return

    for img_name in images:
        input_path = os.path.join(input_dir, img_name)
        output_path = os.path.join(output_dir, img_name)
        preprocess_image(input_path, output_path, size)
        print(f"Processed: {img_name}")

    print(f"All images processed! Saved to {output_dir}")


if __name__ == "__main__":
    input_dir = r"D:\image-matching-model\data\raw_images"   # update to your raw_images path
    output_dir = r"D:\image-matching-model\data\train_processed"  # output folder
    preprocess_folder(input_dir, output_dir)
