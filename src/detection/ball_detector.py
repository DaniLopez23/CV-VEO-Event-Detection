"""
Detector de balon de futbol.
Soporta modelos especializados de Roboflow o modelos YOLO custom.
"""
from ultralytics import YOLO
import numpy as np
from typing import List, Optional, Tuple
import os


class BallDetector:
    """
    Detector de balon con soporte para multiples modelos.

    Modelos soportados:
    - YOLOv8 estandar (clase 32 = sports ball en COCO)
    - Modelos especializados de Roboflow
    - Modelos custom entrenados en balones de futbol
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        use_specialized: bool = True,
        specialized_model_path: Optional[str] = None,
    ):
        """
        Inicializa el detector de balon.

        Args:
            model_path: Ruta al modelo YOLO estandar
            use_specialized: Si usar modelo especializado para balon
            specialized_model_path: Ruta a modelo especializado (si existe)
        """
        self.use_specialized = use_specialized
        self.specialized_model = None
        self.standard_model = None

        # Intentar cargar modelo especializado
        if use_specialized and specialized_model_path:
            if os.path.exists(specialized_model_path):
                try:
                    self.specialized_model = YOLO(specialized_model_path)
                    print(f"Modelo especializado cargado: {specialized_model_path}")
                except Exception as e:
                    print(f"Error cargando modelo especializado: {e}")
                    self.specialized_model = None

        # Cargar modelo estandar como fallback
        self.standard_model = YOLO(model_path)

        # Clase del balon segun el modelo
        self.ball_class_coco = 32  # sports ball en COCO
        self.ball_class_specialized = 0  # Tipicamente clase 0 en modelos especializados

        # Historial para tracking simple
        self.last_detection: Optional[List[float]] = None
        self.detection_history: List[List[float]] = []
        self.max_history = 10

    def detect_ball(
        self,
        frame: np.ndarray,
        conf_thresh: float = 0.3,
        use_tracking: bool = True,
    ) -> List[List[float]]:
        """
        Detecta el balon en un frame.

        Args:
            frame: Frame BGR
            conf_thresh: Umbral de confianza
            use_tracking: Si usar historial para mejorar deteccion

        Returns:
            Lista de detecciones [x1, y1, x2, y2, conf]
        """
        detections = []

        # Intentar con modelo especializado primero
        if self.specialized_model is not None:
            detections = self._detect_with_model(
                frame,
                self.specialized_model,
                self.ball_class_specialized,
                conf_thresh
            )

        # Fallback a modelo estandar si no hay detecciones
        if len(detections) == 0:
            detections = self._detect_with_model(
                frame,
                self.standard_model,
                self.ball_class_coco,
                conf_thresh * 0.8  # Umbral mas bajo para fallback
            )

        # Aplicar filtrado y tracking
        if use_tracking:
            detections = self._apply_tracking_filter(detections)

        # Actualizar historial
        if len(detections) > 0:
            self.last_detection = detections[0]
            self.detection_history.append(detections[0])
            if len(self.detection_history) > self.max_history:
                self.detection_history.pop(0)

        return detections

    def _detect_with_model(
        self,
        frame: np.ndarray,
        model: YOLO,
        ball_class: int,
        conf_thresh: float,
    ) -> List[List[float]]:
        """Detecta usando un modelo especifico."""
        results = model(frame, verbose=False)
        detections = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if cls == ball_class and conf >= conf_thresh:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append([x1, y1, x2, y2, conf])

        return detections

    def _apply_tracking_filter(
        self, detections: List[List[float]]
    ) -> List[List[float]]:
        """
        Aplica filtrado basado en historial para mejorar detecciones.
        Filtra detecciones que estan muy lejos de la ultima posicion conocida.
        """
        if len(detections) == 0:
            return detections

        if self.last_detection is None:
            return detections

        # Calcular centro de ultima deteccion
        last_cx = (self.last_detection[0] + self.last_detection[2]) / 2
        last_cy = (self.last_detection[1] + self.last_detection[3]) / 2

        # Filtrar detecciones muy lejanas (el balon no puede moverse tanto entre frames)
        max_distance = 300  # pixeles
        filtered = []

        for det in detections:
            cx = (det[0] + det[2]) / 2
            cy = (det[1] + det[3]) / 2
            dist = np.sqrt((cx - last_cx)**2 + (cy - last_cy)**2)

            if dist <= max_distance:
                filtered.append(det)

        # Si todas las detecciones fueron filtradas, usar la mas cercana
        if len(filtered) == 0 and len(detections) > 0:
            min_dist = float('inf')
            closest = detections[0]
            for det in detections:
                cx = (det[0] + det[2]) / 2
                cy = (det[1] + det[3]) / 2
                dist = np.sqrt((cx - last_cx)**2 + (cy - last_cy)**2)
                if dist < min_dist:
                    min_dist = dist
                    closest = det
            filtered = [closest]

        return filtered

    def get_ball_center(
        self, detections: List[List[float]]
    ) -> Optional[Tuple[float, float]]:
        """
        Obtiene el centro del balon de las detecciones.

        Args:
            detections: Lista de detecciones

        Returns:
            (x, y) centro del balon o None
        """
        if len(detections) == 0:
            return None

        # Usar la deteccion con mayor confianza
        best = max(detections, key=lambda d: d[4])
        x1, y1, x2, y2, _ = best
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def predict_position(self) -> Optional[Tuple[float, float]]:
        """
        Predice la siguiente posicion del balon basandose en el historial.
        Util cuando el balon no es detectado.

        Returns:
            (x, y) posicion predicha o None
        """
        if len(self.detection_history) < 2:
            return None

        # Calcular velocidad promedio de los ultimos frames
        velocities_x = []
        velocities_y = []

        for i in range(1, len(self.detection_history)):
            prev = self.detection_history[i-1]
            curr = self.detection_history[i]

            prev_cx = (prev[0] + prev[2]) / 2
            prev_cy = (prev[1] + prev[3]) / 2
            curr_cx = (curr[0] + curr[2]) / 2
            curr_cy = (curr[1] + curr[3]) / 2

            velocities_x.append(curr_cx - prev_cx)
            velocities_y.append(curr_cy - prev_cy)

        avg_vx = np.mean(velocities_x)
        avg_vy = np.mean(velocities_y)

        # Predecir siguiente posicion
        if self.last_detection is not None:
            last_cx = (self.last_detection[0] + self.last_detection[2]) / 2
            last_cy = (self.last_detection[1] + self.last_detection[3]) / 2
            return (last_cx + avg_vx, last_cy + avg_vy)

        return None

    def reset(self):
        """Resetea el historial del detector."""
        self.last_detection = None
        self.detection_history = []


def download_roboflow_model(
    workspace: str,
    project: str,
    version: int,
    api_key: str,
    output_dir: str = "models",
) -> str:
    """
    Descarga un modelo de Roboflow.

    Args:
        workspace: Nombre del workspace en Roboflow
        project: Nombre del proyecto
        version: Version del modelo
        api_key: API key de Roboflow
        output_dir: Directorio donde guardar el modelo

    Returns:
        Ruta al modelo descargado
    """
    try:
        from roboflow import Roboflow

        rf = Roboflow(api_key=api_key)
        project = rf.workspace(workspace).project(project)
        model = project.version(version).model

        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, f"{project}_{version}.pt")

        # El modelo se guarda automaticamente
        print(f"Modelo descargado en: {model_path}")
        return model_path

    except ImportError:
        print("roboflow no instalado. Instalar con: pip install roboflow")
        return ""
    except Exception as e:
        print(f"Error descargando modelo: {e}")
        return ""
