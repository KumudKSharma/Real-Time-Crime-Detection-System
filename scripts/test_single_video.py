import torch
import cv2
from train_model import UCFCrimeDataset, SimpleCrimeDetector
from pathlib import Path
import numpy as np

# ---- SETTINGS ----
VIDEO_PATH = "./data/UCF-Crime/Abuse/Abuse001_x264.mp4"
MODEL_PATH = "crime_model.pth"

crime_labels = [
    'Normal', 'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary',
    'Explosion', 'Fighting', 'RoadAccident', 'Robbery',
    'Shooting', 'Shoplifting', 'Stealing', 'Vandalism'
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_frames(video_path, seq_len=16):
    frames = []
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count == 0:
        raise RuntimeError(f"Video has 0 frames or cannot be read: {video_path}")

    if frame_count > seq_len:
        indices = np.linspace(0, frame_count-1, seq_len, dtype=int)
    else:
        indices = list(range(frame_count))

    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    # 🔥 FIX: avoid IndexError when frames == []
    if len(frames) == 0:
        raise RuntimeError(f"No frames could be read from: {video_path}")

    # pad frames
    while len(frames) < seq_len:
        frames.append(frames[-1].copy())

    frames = np.stack(frames).transpose(3, 0, 1, 2)
    return torch.tensor(frames).float().unsqueeze(0) / 255.0



def main():
    print("Loading model...")
    model = SimpleCrimeDetector(num_classes=len(crime_labels))
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    print("Loading video:", VIDEO_PATH)
    frames = load_frames(VIDEO_PATH).to(DEVICE)

    print("Running inference...")
    with torch.no_grad():
        output = model(frames)
        pred = torch.argmax(output, dim=1).item()

    print("\n=== RESULT ===")
    print("Predicted class:", crime_labels[pred])


if __name__ == "__main__":
    main()
