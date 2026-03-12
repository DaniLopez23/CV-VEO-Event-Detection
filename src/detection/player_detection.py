from ultralytics import YOLO

class PlayerDetector:
    def __init__(self, model_path="yolov8n.pt"):
        """
        Inicializa YOLOv8
        model_path: ruta al modelo YOLOv8 (puede ser 'yolov8n.pt')
        """
        self.model = YOLO(model_path)

    def detect_players(self, frame, conf_thresh=0.3):
        """
        Detecta jugadores en un frame
        Devuelve lista de bbox [x1, y1, x2, y2, conf]
        """
        results = self.model(frame)
        detections = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if cls == 0 and conf >= conf_thresh:  # clase persona
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append([x1, y1, x2, y2, conf])
        return detections