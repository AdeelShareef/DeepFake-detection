import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os


MODEL_PATH = os.path.join('model', 'deepshield_xceptio.pth')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# NOTE: Adjust the model architecture to match what you trained.
# Below is a generic example of using a pretrained backbone + classifier head.
class SimpleClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        import torchvision.models as models
        self.backbone = models.resnet18(pretrained=False)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(in_feats, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x
    
_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


_model = None

def load_model():
    global _model
    if _model is not None:
        return _model
    model = SimpleClassifier(num_classes=2)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    try:
        model.load_state_dict(state)
    except Exception:
            # if saved as dict with key 'model_state' or similar adapt here
            if isinstance(state, dict) and 'model_state' in state:
                model.load_state_dict(state['model_state'])
            else:
                # attempt flexible loading
                model.load_state_dict(state, strict=False)
    model.to(DEVICE)
    model.eval()
    _model = model
    return _model


def predict_image(image_path):
    """Returns (predicted_label, confidence) where predicted_label is 0 or 1 and confidence is probability for predicted class."""
    model = load_model()
    img = Image.open(image_path).convert('RGB')
    x = _transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    # class 0: real, class 1: manipulated (match your training labels)
    pred = int(np.argmax(probs))
    confidence = float(probs[pred])
    return pred, confidence

# utility to extract frames from a video and get predictions for each frame
def predict_video(video_path, out_dir='tmp_frames', step=30):
    import cv2
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    results = []
    while True:
        grabbed, frame = cap.read()
        if not grabbed:
            break
        if frame_idx % step == 0:
            fname = os.path.join(out_dir, f'frame_{frame_idx}.jpg')
            cv2.imwrite(fname, frame)
            pred, conf = predict_image(fname)
            results.append({'frame': fname, 'pred': pred, 'confidence': conf})
        frame_idx += 1
    cap.release()
    return results