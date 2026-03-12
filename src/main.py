import cv2
from src.utils.video import load_video, get_frames
from src.detection.player_detector import PlayerDetector
from src.utils.visualization import draw_boxes
import os

video_path = "data/raw_videos/match.mp4"
output_path = "outputs/detection_videos/match_detected.mp4"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Cargar video
cap = load_video(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Inicializar detector
detector = PlayerDetector()

# VideoWriter para guardar el video con detecciones
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

for frame_idx, frame in get_frames(cap):
    detections = detector.detect_players(frame)
    frame_drawn = draw_boxes(frame, detections)
    out.write(frame_drawn)

cap.release()
out.release()
print("Video guardado en", output_path)