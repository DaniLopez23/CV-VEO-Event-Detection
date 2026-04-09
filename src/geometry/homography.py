"""
Calculo y aplicacion de homografias para transformar
coordenadas de imagen a coordenadas del campo.
"""
import cv2
import numpy as np
from typing import Tuple, Optional, List


class HomographyEstimator:
    """Calcula y aplica homografias imagen <-> campo."""

    def __init__(self):
        self.current_homography: Optional[np.ndarray] = None
        self.inverse_homography: Optional[np.ndarray] = None
        self.confidence: float = 0.0
        self.inlier_mask: Optional[np.ndarray] = None

    def compute_homography(
        self,
        image_points: np.ndarray,
        field_points: np.ndarray,
        method: int = cv2.RANSAC,
        reproj_threshold: float = 5.0,
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        Calcula homografia de imagen a campo.

        Args:
            image_points: Puntos en la imagen (Nx2)
            field_points: Puntos correspondientes en el campo (Nx2)
            method: Metodo de estimacion (RANSAC recomendado)
            reproj_threshold: Umbral de reproyeccion para RANSAC

        Returns:
            H: Matriz de homografia 3x3 (o None si falla)
            mask: Mascara de inliers
        """
        if len(image_points) < 4 or len(field_points) < 4:
            return None, np.array([])

        image_pts = np.float32(image_points).reshape(-1, 1, 2)
        field_pts = np.float32(field_points).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(image_pts, field_pts, method, reproj_threshold)

        if H is not None:
            self.current_homography = H
            self.inverse_homography = np.linalg.inv(H)
            self.inlier_mask = mask.ravel() if mask is not None else None
            self.confidence = (
                np.sum(mask) / len(mask) if mask is not None and len(mask) > 0 else 0.0
            )

        return H, mask if mask is not None else np.array([])

    def transform_point(
        self, point: Tuple[float, float], H: Optional[np.ndarray] = None
    ) -> Optional[Tuple[float, float]]:
        """
        Transforma un punto de coordenadas imagen a coordenadas campo.

        Args:
            point: Punto en la imagen (x, y)
            H: Matriz de homografia (usa current_homography si None)

        Returns:
            Punto transformado (x, y) o None si no hay homografia
        """
        if H is None:
            H = self.current_homography

        if H is None:
            return None

        pt = np.array([[[point[0], point[1]]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, H)

        return (float(transformed[0][0][0]), float(transformed[0][0][1]))

    def transform_points(
        self, points: np.ndarray, H: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Transforma multiples puntos.

        Args:
            points: Array de puntos (Nx2)
            H: Matriz de homografia (usa current_homography si None)

        Returns:
            Array de puntos transformados (Nx2)
        """
        if H is None:
            H = self.current_homography

        if H is None or len(points) == 0:
            return np.array([])

        pts = np.float32(points).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, H)

        return transformed.reshape(-1, 2)

    def transform_point_inverse(
        self, point: Tuple[float, float]
    ) -> Optional[Tuple[float, float]]:
        """
        Transforma un punto de coordenadas campo a coordenadas imagen.
        """
        if self.inverse_homography is None:
            return None

        return self.transform_point(point, self.inverse_homography)

    def get_reprojection_error(
        self, image_points: np.ndarray, field_points: np.ndarray
    ) -> float:
        """
        Calcula el error de reproyeccion promedio.

        Args:
            image_points: Puntos originales en imagen (Nx2)
            field_points: Puntos objetivo en campo (Nx2)

        Returns:
            Error promedio en pixeles
        """
        if self.current_homography is None:
            return float("inf")

        transformed = self.transform_points(image_points)
        if len(transformed) == 0:
            return float("inf")

        errors = np.sqrt(np.sum((transformed - field_points) ** 2, axis=1))
        return float(np.mean(errors))

    def is_valid(self, min_confidence: float = 0.5) -> bool:
        """Verifica si la homografia actual es valida."""
        return (
            self.current_homography is not None and self.confidence >= min_confidence
        )

    def project_field_to_image(
        self, field_points: List[Tuple[float, float]]
    ) -> List[Tuple[int, int]]:
        """
        Proyecta puntos del campo a la imagen (para visualizacion).
        """
        if self.inverse_homography is None:
            return []

        result = []
        for pt in field_points:
            img_pt = self.transform_point(pt, self.inverse_homography)
            if img_pt is not None:
                result.append((int(img_pt[0]), int(img_pt[1])))

        return result

    def warp_field_view(
        self, frame: np.ndarray, output_size: Tuple[int, int] = (1050, 680)
    ) -> Optional[np.ndarray]:
        """
        Genera vista de pajaro del campo desde el frame.

        Args:
            frame: Frame de entrada
            output_size: Tamano de salida (ancho, alto)

        Returns:
            Imagen warpeada o None
        """
        if self.current_homography is None:
            return None

        return cv2.warpPerspective(frame, self.current_homography, output_size)
