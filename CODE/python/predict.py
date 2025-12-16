import sys
import json
import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import numpy as np

from face_utils import extract_face
from video_utils import extract_frames

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pth")

# ---------------- DEVICE ----------------
device = torch.device("cpu")

# ---------------- MODEL ----------------
model = timm.create_model("legacy_xception", pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),   # converts to [0,1]
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- IMAGE ----------------
def predict_image(path):
    img = Image.open(path).convert("RGB")
    face_img = extract_face(img)

    if face_img is None:
        raise RuntimeError("No face detected")

    face = transform(face_img).unsqueeze(0).to(device)

    with torch.no_grad():
        prob = torch.sigmoid(model(face)).item()

    return prob, 1

# ---------------- VIDEO ----------------
def predict_video(path):
    frames = extract_frames(path)
    probs = []

    for frame in frames:
        face_img = extract_face(frame)
        if face_img is None:
            continue

        face = transform(face_img).unsqueeze(0).to(device)

        with torch.no_grad():
            prob = torch.sigmoid(model(face)).item()
            probs.append(prob)

    if len(probs) == 0:
        raise RuntimeError("No faces detected in video")

    probs = np.array(probs)

    # 🔥 FIXED aggregation (mean was killing signal)
    top_k = max(1, int(0.3 * len(probs)))
    strongest = np.sort(probs)[-top_k:]

    avg_conf = float(np.mean(strongest))
    fake_ratio = float((probs >= 0.65).mean())

    return fake_ratio, avg_conf, len(probs)

# ---------------- DECISION ----------------
def decide_video(fake_ratio, avg_conf):
    if avg_conf >= 0.7 and fake_ratio >= 0.25:
        return "FAKE", avg_conf
    elif avg_conf <= 0.35:
        return "REAL", 1 - avg_conf
    else:
        return "UNCERTAIN", avg_conf

# ---------------- MAIN ----------------
if __name__ == "__main__":
    path = sys.argv[1]
    ext = os.path.splitext(path)[1].lower()

    try:
        if ext in [".mp4", ".avi", ".mov", ".webm"]:
            fake_ratio, avg_conf, frames_used = predict_video(path)
            label, conf = decide_video(fake_ratio, avg_conf)

            output = {
                "result": label,
                "confidence": round(conf, 4),
                "fake_ratio": round(fake_ratio, 3),
                "frames_used": frames_used,
                "media_type": "video"
            }

        else:
            prob, frames_used = predict_image(path)

            if prob >= 0.65:
                label = "FAKE"
                conf = prob
            elif prob <= 0.35:
                label = "REAL"
                conf = 1 - prob
            else:
                label = "UNCERTAIN"
                conf = prob

            output = {
                "result": label,
                "confidence": round(conf, 4),
                "frames_used": frames_used,
                "media_type": "image"
            }

        print(json.dumps(output))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
