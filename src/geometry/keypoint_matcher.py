"""
Correspondencia de keypoints detectados con puntos del modelo del campo.
"""
import numpy as np
from typing import List, Tuple, Dict, Optional
from itertools import combinations


class KeypointMatcher:
    """
    Asocia puntos detectados en la imagen con puntos conocidos del campo.
    Usa heuristicas geometricas y analisis de patrones.
    """

    def __init__(self, field_model):
        self.field_model = field_model
        self.detected_keypoints: Dict[str, Tuple[int, int]] = {}

    def match_intersections(
        self,
        intersections: List[Tuple[int, int]],
        lines_info: Dict,
        frame_shape: Tuple[int, int],
        min_matches: int = 4,
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[float, float]]]:
        """
        Empareja intersecciones detectadas con keypoints del campo.

        Args:
            intersections: Lista de puntos de interseccion detectados
            lines_info: Informacion sobre las lineas detectadas
            frame_shape: Forma del frame (height, width)
            min_matches: Minimo de matches requeridos

        Returns:
            Tupla de (puntos_imagen, puntos_campo)
        """
        if len(intersections) < min_matches:
            return [], []

        h, w = frame_shape[:2]

        # Clasificar intersecciones por region del frame
        regions = self._classify_by_region(intersections, w, h)

        # Intentar identificar patrones conocidos
        matches = self._identify_patterns(intersections, regions, lines_info)

        if len(matches) < min_matches:
            # Fallback: usar heuristica simple basada en posicion
            matches = self._heuristic_match(intersections, w, h)

        image_points = [m[0] for m in matches]
        field_points = [m[1] for m in matches]

        return image_points, field_points

    def _classify_by_region(
        self, points: List[Tuple[int, int]], width: int, height: int
    ) -> Dict[str, List[Tuple[int, int]]]:
        """Clasifica puntos por region del frame."""
        regions = {"left": [], "center": [], "right": [], "top": [], "bottom": []}

        for point in points:
            x, y = point
            rel_x = x / width
            rel_y = y / height

            # Horizontal
            if rel_x < 0.33:
                regions["left"].append(point)
            elif rel_x > 0.66:
                regions["right"].append(point)
            else:
                regions["center"].append(point)

            # Vertical
            if rel_y < 0.4:
                regions["top"].append(point)
            elif rel_y > 0.6:
                regions["bottom"].append(point)

        return regions

    def _identify_patterns(
        self,
        intersections: List[Tuple[int, int]],
        regions: Dict,
        lines_info: Dict,
    ) -> List[Tuple[Tuple[int, int], Tuple[float, float]]]:
        """
        Intenta identificar patrones geometricos conocidos.
        Por ejemplo: area penal, linea central, etc.
        """
        matches = []

        # Buscar patrones de area penal (rectangulo caracteristico)
        penalty_matches = self._find_penalty_area_pattern(intersections, regions)
        matches.extend(penalty_matches)

        # Buscar linea central
        center_matches = self._find_center_line_pattern(intersections, regions)
        matches.extend(center_matches)

        return matches

    def _find_penalty_area_pattern(
        self,
        intersections: List[Tuple[int, int]],
        regions: Dict,
    ) -> List[Tuple[Tuple[int, int], Tuple[float, float]]]:
        """
        Busca el patron de esquinas del area penal.
        """
        matches = []

        # Buscar en region izquierda (area penal izquierda)
        left_points = regions.get("left", [])
        if len(left_points) >= 2:
            # Ordenar por Y para encontrar esquinas superior e inferior
            sorted_by_y = sorted(left_points, key=lambda p: p[1])

            # Los puntos mas arriba y abajo podrian ser las esquinas del area
            if len(sorted_by_y) >= 2:
                top_point = sorted_by_y[0]
                bottom_point = sorted_by_y[-1]

                # Verificar que estan a distancia razonable
                y_dist = abs(bottom_point[1] - top_point[1])
                if y_dist > 50:  # Minimo 50 pixeles de separacion
                    matches.append(
                        (top_point, self.field_model.keypoints["penalty_area_left_top"])
                    )
                    matches.append(
                        (
                            bottom_point,
                            self.field_model.keypoints["penalty_area_left_bottom"],
                        )
                    )

        # Similar para region derecha
        right_points = regions.get("right", [])
        if len(right_points) >= 2:
            sorted_by_y = sorted(right_points, key=lambda p: p[1])
            if len(sorted_by_y) >= 2:
                top_point = sorted_by_y[0]
                bottom_point = sorted_by_y[-1]
                y_dist = abs(bottom_point[1] - top_point[1])
                if y_dist > 50:
                    matches.append(
                        (
                            top_point,
                            self.field_model.keypoints["penalty_area_right_top"],
                        )
                    )
                    matches.append(
                        (
                            bottom_point,
                            self.field_model.keypoints["penalty_area_right_bottom"],
                        )
                    )

        return matches

    def _find_center_line_pattern(
        self,
        intersections: List[Tuple[int, int]],
        regions: Dict,
    ) -> List[Tuple[Tuple[int, int], Tuple[float, float]]]:
        """
        Busca puntos de la linea central.
        """
        matches = []
        center_points = regions.get("center", [])

        if len(center_points) >= 2:
            # Ordenar por Y
            sorted_by_y = sorted(center_points, key=lambda p: p[1])
            top_point = sorted_by_y[0]
            bottom_point = sorted_by_y[-1]

            y_dist = abs(bottom_point[1] - top_point[1])
            if y_dist > 100:  # La linea central debe ser larga
                matches.append(
                    (top_point, self.field_model.keypoints["center_top"])
                )
                matches.append(
                    (bottom_point, self.field_model.keypoints["center_bottom"])
                )

        return matches

    def _heuristic_match(
        self, intersections: List[Tuple[int, int]], width: int, height: int
    ) -> List[Tuple[Tuple[int, int], Tuple[float, float]]]:
        """
        Matching heuristico simple basado en posicion relativa.
        Asume camara lateral tipica de transmision TV.
        """
        matches = []

        # Ordenar puntos
        sorted_by_x = sorted(intersections, key=lambda p: p[0])
        sorted_by_y = sorted(intersections, key=lambda p: p[1])

        # Esquinas aproximadas
        if len(sorted_by_x) >= 4:
            # Puntos mas a la izquierda
            leftmost = sorted_by_x[:2]
            leftmost_sorted = sorted(leftmost, key=lambda p: p[1])
            if len(leftmost_sorted) >= 2:
                matches.append(
                    (
                        leftmost_sorted[0],
                        self.field_model.keypoints["corner_top_left"],
                    )
                )
                matches.append(
                    (
                        leftmost_sorted[1],
                        self.field_model.keypoints["corner_bottom_left"],
                    )
                )

            # Puntos mas a la derecha
            rightmost = sorted_by_x[-2:]
            rightmost_sorted = sorted(rightmost, key=lambda p: p[1])
            if len(rightmost_sorted) >= 2:
                matches.append(
                    (
                        rightmost_sorted[0],
                        self.field_model.keypoints["corner_top_right"],
                    )
                )
                matches.append(
                    (
                        rightmost_sorted[1],
                        self.field_model.keypoints["corner_bottom_right"],
                    )
                )

        return matches

    def refine_matches_with_ransac(
        self,
        image_points: List[Tuple[int, int]],
        field_points: List[Tuple[float, float]],
        threshold: float = 10.0,
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[float, float]]]:
        """
        Refina matches eliminando outliers usando RANSAC.
        """
        if len(image_points) < 4:
            return image_points, field_points

        # Este metodo se usaria internamente por HomographyEstimator
        # Aqui solo retornamos los puntos sin modificar
        return image_points, field_points

    def estimate_visible_area(
        self, intersections: List[Tuple[int, int]], frame_shape: Tuple[int, int]
    ) -> str:
        """
        Estima que parte del campo es visible basandose en los puntos detectados.

        Returns:
            "full_field", "left_half", "right_half", "center", "unknown"
        """
        if len(intersections) < 2:
            return "unknown"

        h, w = frame_shape[:2]
        x_coords = [p[0] for p in intersections]
        x_range = max(x_coords) - min(x_coords)
        x_center = np.mean(x_coords)

        # Si los puntos cubren mas del 60% del ancho, probablemente es campo completo
        if x_range > 0.6 * w:
            return "full_field"

        # Determinar por posicion del centroide
        if x_center < w * 0.4:
            return "left_half"
        elif x_center > w * 0.6:
            return "right_half"
        else:
            return "center"
