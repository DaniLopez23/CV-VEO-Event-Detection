import cv2
from utils.video import load_video, get_frames
from detection.player_detection import PlayerDetector
from detection.ball_detector import BallDetector
from utils.visualization_elements import draw_boxes
import os

video_path = "data/raw_videos/demo_2_video_1.mp4"
output_path = "data/outputs/detection_videos/match_detected.mp4"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Cargar video
cap = load_video(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Inicializar detectores
player_detector = PlayerDetector()
ball_detector = BallDetector()

# VideoWriter para guardar video con detecciones
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

for frame_idx, frame in get_frames(cap):
    player_dets = player_detector.detect_players(frame)
    ball_dets = ball_detector.detect_ball(frame)

    # Dibujar jugadores en verde y balón en rojo
    frame_drawn = draw_boxes(frame, player_dets, color=(0,255,0), label_prefix="P:")
    frame_drawn = draw_boxes(frame_drawn, ball_dets, color=(0,0,255), label_prefix="B:")

    out.write(frame_drawn)

cap.release()
out.release()
print("Video guardado en", output_path)