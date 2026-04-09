"""
Modelo 2D del campo de futbol con dimensiones FIFA.
"""
import numpy as np
from typing import Dict, Tuple, List
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config.field_config import FIELD_CONFIG, FIELD_KEYPOINTS


class FieldModel:
    """Modelo geometrico 2D del campo de futbol."""

    def __init__(self, config: Dict = None, keypoints: Dict = None):
        self.config = config or FIELD_CONFIG
        self.keypoints = keypoints or FIELD_KEYPOINTS

        # Dimensiones principales
        self.length = self.config["field_length"]
        self.width = self.config["field_width"]
        self.half_length = self.length / 2
        self.half_width = self.width / 2

    def get_keypoint(self, name: str) -> Tuple[float, float]:
        """Obtiene las coordenadas de un keypoint por nombre."""
        if name not in self.keypoints:
            raise KeyError(f"Keypoint '{name}' no encontrado")
        return self.keypoints[name]

    def get_all_keypoints(self) -> Dict[str, Tuple[float, float]]:
        """Retorna todos los keypoints."""
        return self.keypoints.copy()

    def get_keypoints_array(self) -> np.ndarray:
        """Retorna keypoints como array numpy (Nx2)."""
        return np.array(list(self.keypoints.values()))

    def get_keypoint_names(self) -> List[str]:
        """Retorna lista de nombres de keypoints."""
        return list(self.keypoints.keys())

    def get_penalty_area_points(self, side: str = "left") -> List[Tuple[float, float]]:
        """Retorna los 4 puntos del area penal."""
        if side == "left":
            return [
                self.keypoints["penalty_area_left_top_corner"],
                self.keypoints["penalty_area_left_top"],
                self.keypoints["penalty_area_left_bottom"],
                self.keypoints["penalty_area_left_bottom_corner"],
            ]
        else:
            return [
                self.keypoints["penalty_area_right_top_corner"],
                self.keypoints["penalty_area_right_top"],
                self.keypoints["penalty_area_right_bottom"],
                self.keypoints["penalty_area_right_bottom_corner"],
            ]

    def get_goal_area_points(self, side: str = "left") -> List[Tuple[float, float]]:
        """Retorna los 4 puntos del area chica."""
        if side == "left":
            return [
                self.keypoints["goal_area_left_top_corner"],
                self.keypoints["goal_area_left_top"],
                self.keypoints["goal_area_left_bottom"],
                self.keypoints["goal_area_left_bottom_corner"],
            ]
        else:
            return [
                self.keypoints["goal_area_right_top_corner"],
                self.keypoints["goal_area_right_top"],
                self.keypoints["goal_area_right_bottom"],
                self.keypoints["goal_area_right_bottom_corner"],
            ]

    def get_center_circle_points(self, num_points: int = 36) -> List[Tuple[float, float]]:
        """Genera puntos del circulo central."""
        radius = self.config["center_circle_radius"]
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        return [(radius * np.cos(a), radius * np.sin(a)) for a in angles]

    def get_field_corners(self) -> List[Tuple[float, float]]:
        """Retorna las 4 esquinas del campo."""
        return [
            self.keypoints["corner_top_left"],
            self.keypoints["corner_top_right"],
            self.keypoints["corner_bottom_right"],
            self.keypoints["corner_bottom_left"],
        ]

    def is_point_in_field(self, point: Tuple[float, float]) -> bool:
        """Verifica si un punto esta dentro del campo."""
        x, y = point
        return abs(x) <= self.half_length and abs(y) <= self.half_width

    def is_point_in_penalty_area(
        self, point: Tuple[float, float], side: str = "left"
    ) -> bool:
        """Verifica si un punto esta dentro del area penal."""
        x, y = point
        pa_length = self.config["penalty_area_length"]
        pa_half_width = self.config["penalty_area_width"] / 2

        if side == "left":
            return x <= -self.half_length + pa_length and abs(y) <= pa_half_width
        else:
            return x >= self.half_length - pa_length and abs(y) <= pa_half_width

    def get_line_segments(self) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Retorna todos los segmentos de linea del campo como pares de puntos.
        Util para matching con lineas detectadas.
        """
        segments = []

        # Contorno del campo
        corners = self.get_field_corners()
        for i in range(4):
            segments.append((corners[i], corners[(i + 1) % 4]))

        # Linea central
        segments.append(
            (self.keypoints["center_top"], self.keypoints["center_bottom"])
        )

        # Area penal izquierda
        segments.append(
            (
                self.keypoints["penalty_area_left_top_corner"],
                self.keypoints["penalty_area_left_top"],
            )
        )
        segments.append(
            (
                self.keypoints["penalty_area_left_top"],
                self.keypoints["penalty_area_left_bottom"],
            )
        )
        segments.append(
            (
                self.keypoints["penalty_area_left_bottom"],
                self.keypoints["penalty_area_left_bottom_corner"],
            )
        )

        # Area penal derecha
        segments.append(
            (
                self.keypoints["penalty_area_right_top_corner"],
                self.keypoints["penalty_area_right_top"],
            )
        )
        segments.append(
            (
                self.keypoints["penalty_area_right_top"],
                self.keypoints["penalty_area_right_bottom"],
            )
        )
        segments.append(
            (
                self.keypoints["penalty_area_right_bottom"],
                self.keypoints["penalty_area_right_bottom_corner"],
            )
        )

        # Area chica izquierda
        segments.append(
            (
                self.keypoints["goal_area_left_top_corner"],
                self.keypoints["goal_area_left_top"],
            )
        )
        segments.append(
            (
                self.keypoints["goal_area_left_top"],
                self.keypoints["goal_area_left_bottom"],
            )
        )
        segments.append(
            (
                self.keypoints["goal_area_left_bottom"],
                self.keypoints["goal_area_left_bottom_corner"],
            )
        )

        # Area chica derecha
        segments.append(
            (
                self.keypoints["goal_area_right_top_corner"],
                self.keypoints["goal_area_right_top"],
            )
        )
        segments.append(
            (
                self.keypoints["goal_area_right_top"],
                self.keypoints["goal_area_right_bottom"],
            )
        )
        segments.append(
            (
                self.keypoints["goal_area_right_bottom"],
                self.keypoints["goal_area_right_bottom_corner"],
            )
        )

        return segments
