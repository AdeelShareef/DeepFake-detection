import cv2
from PIL import Image

def extract_frames(video_path, frame_skip=1, max_frames=100):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % frame_skip == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))

            if len(frames) >= max_frames:
                break

        frame_id += 1

    cap.release()
    return frames
