import torch
from facenet_pytorch import MTCNN
from PIL import Image

device = torch.device("cpu")

mtcnn = MTCNN(
    image_size=224,
    margin=20,
    min_face_size=40,
    thresholds=[0.6, 0.7, 0.7],
    post_process=False,   # IMPORTANT
    device=device
)

def extract_face(pil_img):
    """
    Returns cropped PIL face image or None
    """
    if not isinstance(pil_img, Image.Image):
        return None

    face = mtcnn(pil_img)

    if face is None:
        return None

    # Convert tensor → PIL (model expects normalized tensor later)
    face = face.permute(1, 2, 0).byte().cpu().numpy()
    return Image.fromarray(face)
