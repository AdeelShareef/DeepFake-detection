import sys
import json
import os

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ------------------ MODEL DEFINITION ------------------
class DeepShieldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.xception(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

    def forward(self, x):
        return self.model(x)

# ------------------ LOAD MODEL ------------------
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "best_model.pth")

    model = DeepShieldModel()
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

# ------------------ IMAGE TRANSFORM ------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ------------------ PREDICTION ------------------
def predict(image_path):
    model = load_model()

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    label = "REAL" if pred.item() == 0 else "FAKE"

    return label, round(conf.item() * 100, 2)

# ------------------ MAIN ------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No image path provided"}))
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(json.dumps({"error": "Image not found"}))
        sys.exit(1)

    result, confidence = predict(image_path)

    print(json.dumps({
        "result": result,
        "confidence": confidence
    }))
