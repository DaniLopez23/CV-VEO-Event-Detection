import cv2

def load_video(video_path):
    """Carga un video y devuelve un VideoCapture"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir {video_path}")
    return cap

def get_frames(cap):
    """Generador de frames del video"""
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame_idx, frame
        frame_idx += 1