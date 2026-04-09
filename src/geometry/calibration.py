"""
Calibracion de camara para establecer homografia.
Soporta calibracion manual (seleccion de puntos) y automatica.
"""
import cv2
import numpy as np
import json
import os
from typing import List, Tuple, Dict, Optional


class CameraCalibrator:
    """
    Calibrador de camara para establecer la homografia imagen-campo.

    Modos:
    - manual: El usuario selecciona 4+ puntos en el frame y sus correspondientes en el campo
    - semi_auto: Detecta lineas automaticamente pero el usuario confirma las correspondencias
    - load: Carga calibracion guardada de un archivo
    """

    # Puntos de referencia comunes del campo con sus coordenadas
    REFERENCE_POINTS = {
        "corner_top_left": (-52.5, 34.0, "Esquina superior izquierda"),
        "corner_top_right": (52.5, 34.0, "Esquina superior derecha"),
        "corner_bottom_left": (-52.5, -34.0, "Esquina inferior izquierda"),
        "corner_bottom_right": (52.5, -34.0, "Esquina inferior derecha"),
        "center_top": (0.0, 34.0, "Linea central - arriba"),
        "center_bottom": (0.0, -34.0, "Linea central - abajo"),
        "center_spot": (0.0, 0.0, "Centro del campo"),
        "penalty_area_left_top": (-36.0, 20.16, "Area penal izq - arriba"),
        "penalty_area_left_bottom": (-36.0, -20.16, "Area penal izq - abajo"),
        "penalty_area_right_top": (36.0, 20.16, "Area penal der - arriba"),
        "penalty_area_right_bottom": (36.0, -20.16, "Area penal der - abajo"),
        "goal_area_left_top": (-47.0, 9.16, "Area chica izq - arriba"),
        "goal_area_left_bottom": (-47.0, -9.16, "Area chica izq - abajo"),
        "goal_area_right_top": (47.0, 9.16, "Area chica der - arriba"),
        "goal_area_right_bottom": (47.0, -9.16, "Area chica der - abajo"),
    }

    def __init__(self):
        self.image_points: List[Tuple[float, float]] = []
        self.field_points: List[Tuple[float, float]] = []
        self.homography: Optional[np.ndarray] = None
        self.calibration_file: Optional[str] = None

        # Para modo interactivo
        self._current_frame: Optional[np.ndarray] = None
        self._click_points: List[Tuple[int, int]] = []
        self._selected_field_points: List[str] = []

    def calibrate_manual(
        self,
        frame: np.ndarray,
        predefined_points: Optional[Dict[str, Tuple[int, int]]] = None,
    ) -> Optional[np.ndarray]:
        """
        Calibracion manual interactiva.

        Args:
            frame: Frame de video para calibrar
            predefined_points: Dict con {nombre_punto: (x, y)} si ya se conocen

        Returns:
            Matriz de homografia o None si falla
        """
        if predefined_points:
            # Usar puntos predefinidos
            for name, img_point in predefined_points.items():
                if name in self.REFERENCE_POINTS:
                    field_x, field_y, _ = self.REFERENCE_POINTS[name]
                    self.image_points.append(img_point)
                    self.field_points.append((field_x, field_y))
        else:
            # Modo interactivo con clicks
            self._interactive_calibration(frame)

        if len(self.image_points) >= 4:
            self.homography = self._compute_homography()
            return self.homography

        return None

    def _interactive_calibration(self, frame: np.ndarray):
        """Calibracion interactiva con ventana OpenCV."""
        self._current_frame = frame.copy()
        self._click_points = []

        window_name = "Calibracion - Click en 4+ puntos del campo"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self._mouse_callback)

        print("\n" + "="*60)
        print("MODO DE CALIBRACION MANUAL")
        print("="*60)
        print("Instrucciones:")
        print("1. Haz click en puntos de referencia del campo (minimo 4)")
        print("2. Despues de cada click, selecciona que punto es")
        print("3. Presiona 'q' cuando termines")
        print("4. Presiona 'r' para resetear")
        print("="*60)
        print("\nPuntos disponibles:")
        for i, (name, (x, y, desc)) in enumerate(self.REFERENCE_POINTS.items()):
            print(f"  {i+1}. {desc} ({name})")

        while True:
            display = self._current_frame.copy()

            # Dibujar puntos ya seleccionados
            for i, point in enumerate(self._click_points):
                cv2.circle(display, point, 8, (0, 255, 0), -1)
                cv2.circle(display, point, 8, (0, 0, 0), 2)
                if i < len(self._selected_field_points):
                    cv2.putText(
                        display,
                        self._selected_field_points[i][:10],
                        (point[0] + 10, point[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )

            # Info en pantalla
            cv2.putText(
                display,
                f"Puntos: {len(self._click_points)} (min 4) | Q=terminar | R=reset",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('r'):
                self._click_points = []
                self._selected_field_points = []
                self.image_points = []
                self.field_points = []
                print("Resetado.")

        cv2.destroyAllWindows()

    def _mouse_callback(self, event, x, y, flags, param):
        """Callback para clicks del mouse."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self._click_points.append((x, y))
            print(f"\nClick en ({x}, {y})")
            print("Selecciona el punto del campo (enter numero):")

            for i, (name, (fx, fy, desc)) in enumerate(self.REFERENCE_POINTS.items()):
                print(f"  {i+1}. {desc}")

            try:
                choice = int(input("Numero: ")) - 1
                point_names = list(self.REFERENCE_POINTS.keys())
                if 0 <= choice < len(point_names):
                    name = point_names[choice]
                    field_x, field_y, desc = self.REFERENCE_POINTS[name]
                    self.image_points.append((x, y))
                    self.field_points.append((field_x, field_y))
                    self._selected_field_points.append(name)
                    print(f"Asignado: {desc}")
                else:
                    print("Numero invalido, punto ignorado")
                    self._click_points.pop()
            except ValueError:
                print("Input invalido, punto ignorado")
                self._click_points.pop()

    def _compute_homography(self) -> Optional[np.ndarray]:
        """Calcula la homografia a partir de los puntos."""
        if len(self.image_points) < 4:
            return None

        img_pts = np.array(self.image_points, dtype=np.float32)
        field_pts = np.array(self.field_points, dtype=np.float32)

        H, mask = cv2.findHomography(img_pts, field_pts, cv2.RANSAC, 5.0)

        return H

    def save_calibration(self, filepath: str):
        """Guarda la calibracion a un archivo JSON."""
        data = {
            "image_points": self.image_points,
            "field_points": self.field_points,
            "homography": self.homography.tolist() if self.homography is not None else None,
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Calibracion guardada en: {filepath}")

    def load_calibration(self, filepath: str) -> bool:
        """Carga calibracion desde archivo JSON."""
        if not os.path.exists(filepath):
            return False

        with open(filepath, 'r') as f:
            data = json.load(f)

        self.image_points = [tuple(p) for p in data.get("image_points", [])]
        self.field_points = [tuple(p) for p in data.get("field_points", [])]

        if data.get("homography"):
            self.homography = np.array(data["homography"])
        else:
            self.homography = self._compute_homography()

        self.calibration_file = filepath
        return True

    def get_homography(self) -> Optional[np.ndarray]:
        """Retorna la homografia calculada."""
        return self.homography

    def is_calibrated(self) -> bool:
        """Verifica si hay calibracion valida."""
        return self.homography is not None

    def transform_point(
        self, point: Tuple[float, float]
    ) -> Optional[Tuple[float, float]]:
        """Transforma un punto de imagen a coordenadas de campo."""
        if self.homography is None:
            return None

        pt = np.array([[[point[0], point[1]]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.homography)

        return (float(transformed[0][0][0]), float(transformed[0][0][1]))


def calibrate_from_preset(
    preset: str = "tv_broadcast_full"
) -> Dict[str, Tuple[int, int]]:
    """
    Retorna puntos de calibracion predefinidos para configuraciones comunes.

    Presets disponibles:
    - tv_broadcast_full: Vista TV tipica de campo completo (1920x1080)
    - tv_broadcast_half_left: Vista TV mitad izquierda
    - tv_broadcast_half_right: Vista TV mitad derecha
    """
    presets = {
        "tv_broadcast_full": {
            # Asume video 1920x1080, vista tipica TV del campo completo
            # Estos valores son aproximados y deben ajustarse
            "corner_top_left": (100, 200),
            "corner_top_right": (1820, 200),
            "corner_bottom_left": (100, 880),
            "corner_bottom_right": (1820, 880),
        },
        "tv_broadcast_half_left": {
            "corner_top_left": (50, 150),
            "penalty_area_left_top": (400, 250),
            "penalty_area_left_bottom": (400, 750),
            "corner_bottom_left": (50, 900),
        },
    }

    return presets.get(preset, {})
