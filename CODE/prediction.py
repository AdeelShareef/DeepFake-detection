import sys
import os
import cv2 # pyright: ignore[reportMissingImports]
import numpy as np # pyright: ignore[reportMissingImports]
import tensorflow as tf # pyright: ignore[reportMissingImports]
from mtcnn import MTCNN # pyright: ignore[reportMissingImports]

# Suppress TensorFlow logs (makes PHP output cleaner)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def analyze_media(file_path):
    try:
        # 1. Load the Model
        # Ensure the .h5 file is in the same folder
        model = tf.keras.models.load_model('deepfake_detector_model.h5')
        detector = MTCNN()
        
        # 2. Open File (Video or Image)
        cap = cv2.VideoCapture(file_path)
        fake_votes = 0
        total_frames = 0
        max_frames_to_scan = 15 # Scan up to 15 frames to be quick
        
        while cap.isOpened() and total_frames < max_frames_to_scan:
            ret, frame = cap.read()
            if not ret: break
            
            # 3. Detect Face
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(rgb_frame)
            
            if faces:
                # Take the first face found
                x, y, w, h = faces[0]['box']
                
                # Add margin (important because model trained with margin)
                margin = int(w * 0.2)
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(frame.shape[1], x + w + margin)
                y2 = min(frame.shape[0], y + h + margin)
                
                face_crop = frame[y1:y2, x1:x2]
                
                if face_crop.size != 0:
                    # 4. Preprocess for AI (Resize to 128x128 and Normalize)
                    face_crop = cv2.resize(face_crop, (128, 128))
                    face_crop = face_crop / 255.0
                    face_crop = np.expand_dims(face_crop, axis=0) # Add batch dimension
                    
                    # 5. Predict
                    prediction = model.predict(face_crop, verbose=0)[0][0]
                    
                    # Model outputs 0 for Fake, 1 for Real (or vice versa depending on training)
                    # Assuming standard: < 0.5 is Class 0, > 0.5 is Class 1
                    # In our training script 'fake' folder usually gets index 0 or 1.
                    # We will assume < 0.5 is FAKE (since output layer is sigmoid 0-1)
                    # Adjust this threshold if results are flipped!
                    if prediction < 0.5:
                        fake_votes += 1
                    
                    total_frames += 1
        
        cap.release()
        
        # 6. Final Verdict
        if total_frames == 0:
            print("ERROR: No faces detected in media.")
        else:
            fake_ratio = fake_votes / total_frames
            # If more than 60% of analyzed frames look fake, mark as fake
            if fake_ratio > 0.6:
                print("FAKE")
            else:
                print("REAL")
                
    except Exception as e:
        print(f"ERROR: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_media(sys.argv[1])
    else:
        print("ERROR: No file provided")