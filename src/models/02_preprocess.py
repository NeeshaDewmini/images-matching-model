import cv2
import os

def preprocess_image(input_path, output_path):
    image = cv2.imread(input_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Simple threshold to separate object
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        cv2.imwrite(output_path, image)
        return
    
    # Get largest contour assuming it's the object
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Crop object
    cropped = image[y:y+h, x:x+w]
    
    # Resize to model input
    resized = cv2.resize(cropped, (224, 224))
    
    cv2.imwrite(output_path, resized)

# Apply to all images
input_dir = "data/train"
output_dir = "data/cropped_images"
os.makedirs(output_dir, exist_ok=True)

for img_file in os.listdir(input_dir):
    input_path = os.path.join(input_dir, img_file)
    output_path = os.path.join(output_dir, img_file)
    preprocess_image(input_path, output_path)

print("✅ Preprocessing done!")
