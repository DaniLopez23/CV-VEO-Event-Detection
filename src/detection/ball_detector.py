from ultralytics import YOLO

class BallDetector:
    def __init__(self, model_path="yolov8n.pt"):
        """
        Inicializa YOLO para detección de balón.
        model_path: modelo YOLO entrenado para balón
        """
        self.model = YOLO(model_path)

    def detect_ball(self, frame, conf_thresh=0.3):
        """
        Detecta balón en un frame
        Devuelve lista de bbox [x1, y1, x2, y2, conf]
        """
        results = self.model(frame)
        detections = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if cls == 32 and conf >= conf_thresh:  # clase sports ball en COCO
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append([x1, y1, x2, y2, conf])
        return detections