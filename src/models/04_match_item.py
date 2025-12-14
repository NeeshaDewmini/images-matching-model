import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.preprocess import preprocess_image
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model = torch.nn.Sequential(*list(model.children())[:-1])  # remove fc
model.eval().to(DEVICE)

# Transform
transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# Load features
features_dict = np.load("train_features.npy", allow_pickle=True).item()
print("Number of indexed training images:", len(features_dict))

# Preprocess lost image
lost_image=r"D:\image-matching-model\data\lost\test1.jpg"
preprocess_image(lost_image, "temp.jpg")
img = Image.open("temp.jpg").convert("RGB")
x = transform(img).unsqueeze(0).to(DEVICE)

# Extract feature
with torch.no_grad():
    lost_feat = model(x).cpu().numpy().flatten()

# Compute similarities
scores = {k: cosine_similarity([lost_feat], [v])[0][0] for k,v in features_dict.items()}
print("\n🔍 Sample similarity scores:")
for k in list(scores.keys())[:5]:
    print(k, scores[k])

THRESHOLD = 0.65  # PoC threshold

filtered = [(k, s) for k, s in scores.items() if s >= THRESHOLD]
filtered = sorted(filtered, key=lambda x: x[1], reverse=True)[:5]

print("\n🎯 Matching result:")
if not filtered:
    print("❌ No confident match found")
else:
    print("✅ Top matches:")
    for path, score in filtered:
        print(f"{path} | similarity: {score:.2f}")
