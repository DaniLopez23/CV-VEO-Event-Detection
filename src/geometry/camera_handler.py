"""
Manejo de camara movil y cambios de toma.
Incluye suavizado temporal de homografias y deteccion de cambios de escena.
"""
import cv2
import numpy as np
from collections import deque
from typing import Optional, Tuple


class ShotChangeDetector:
    """Detecta cambios de toma usando comparacion de histogramas."""

    def __init__(self, threshold: float = 0.5):
        """
        Args:
            threshold: Umbral de correlacion bajo el cual se considera cambio de toma.
                       Valores tipicos: 0.3-0.6
        """
        self.threshold = threshold
        self.prev_hist: Optional[np.ndarray] = None

    def is_shot_change(self, frame: np.ndarray) -> bool:
        """
        Detecta si hubo un cambio de toma respecto al frame anterior.

        Args:
            frame: Frame actual (BGR)

        Returns:
            True si se detecto cambio de toma
        """
        # Calcular histograma del frame actual
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        if self.prev_hist is None:
            self.prev_hist = hist
            return False

        # Comparar con frame anterior
        correlation = cv2.compareHist(self.prev_hist, hist, cv2.HISTCMP_CORREL)

        self.prev_hist = hist

        return correlation < self.threshold

    def reset(self):
        """Resetea el detector."""
        self.prev_hist = None


class CameraHandler:
    """
    Maneja la estabilidad de la homografia ante movimientos de camara.
    Implementa suavizado temporal y recuperacion ante cambios de toma.
    """

    def __init__(
        self,
        history_size: int = 30,
        min_confidence: float = 0.6,
        angle_change_threshold: float = 0.5,
    ):
        """
        Args:
            history_size: Numero de homografias a mantener en historial
            min_confidence: Confianza minima para aceptar una homografia
            angle_change_threshold: Umbral para detectar cambio drastico de angulo
        """
        self.history_size = history_size
        self.min_confidence = min_confidence
        self.angle_change_threshold = angle_change_threshold

        self.homography_history: deque = deque(maxlen=history_size)
        self.last_valid_homography: Optional[np.ndarray] = None
        self.shot_detector = ShotChangeDetector()

        # Metricas
        self.frames_without_homography = 0
        self.total_frames = 0

    def update(
        self, frame: np.ndarray, new_homography: Optional[np.ndarray], confidence: float
    ) -> Optional[np.ndarray]:
        """
        Actualiza la homografia con suavizado temporal y manejo de cambios de toma.

        Args:
            frame: Frame actual
            new_homography: Nueva homografia calculada (puede ser None)
            confidence: Confianza de la nueva homografia

        Returns:
            Homografia a usar (suavizada o la ultima valida)
        """
        self.total_frames += 1

        # Detectar cambio de toma
        is_shot_change = self.shot_detector.is_shot_change(frame)

        if is_shot_change:
            # Reset en cambio de toma
            self.homography_history.clear()
            self.last_valid_homography = None
            self.frames_without_homography = 0
            return None

        # Si no hay nueva homografia valida
        if new_homography is None or confidence < self.min_confidence:
            self.frames_without_homography += 1

            # Si llevamos muchos frames sin homografia, podria ser un close-up
            if self.frames_without_homography > 30:
                # Considerar resetear
                pass

            return self.last_valid_homography

        # Verificar cambio drastico de angulo
        if self.last_valid_homography is not None:
            if self._is_drastic_angle_change(new_homography):
                # Posible cambio de camara, resetear
                self.homography_history.clear()
                self.last_valid_homography = new_homography
                self.frames_without_homography = 0
                return new_homography

        # Agregar al historial
        self.homography_history.append(new_homography)
        self.last_valid_homography = new_homography
        self.frames_without_homography = 0

        # Suavizar si hay suficiente historial
        if len(self.homography_history) >= 3:
            return self._smooth_homography()

        return new_homography

    def _is_drastic_angle_change(self, new_H: np.ndarray) -> bool:
        """
        Detecta si hay un cambio drastico en la homografia (cambio de angulo de camara).
        """
        if self.last_valid_homography is None:
            return False

        # Comparar las matrices de homografia
        # Metodo simple: comparar la diferencia normalizada
        diff = np.abs(new_H - self.last_valid_homography)
        normalized_diff = np.sum(diff) / (np.sum(np.abs(self.last_valid_homography)) + 1e-6)

        return normalized_diff > self.angle_change_threshold

    def _smooth_homography(self) -> np.ndarray:
        """
        Promedia las ultimas homografias para suavizar.
        Usa pesos exponenciales (mas recientes tienen mas peso).
        """
        n = len(self.homography_history)
        if n == 0:
            return self.last_valid_homography

        # Pesos exponenciales
        weights = np.exp(np.linspace(0, 1, n))
        weights /= weights.sum()

        H_smooth = np.zeros((3, 3), dtype=np.float64)
        for i, H in enumerate(self.homography_history):
            H_smooth += weights[i] * H

        # Normalizar para mantener H[2,2] = 1
        if abs(H_smooth[2, 2]) > 1e-6:
            H_smooth /= H_smooth[2, 2]

        return H_smooth

    def get_stats(self) -> dict:
        """Retorna estadisticas del handler."""
        return {
            "total_frames": self.total_frames,
            "frames_without_homography": self.frames_without_homography,
            "history_size": len(self.homography_history),
            "has_valid_homography": self.last_valid_homography is not None,
        }

    def reset(self):
        """Resetea completamente el handler."""
        self.homography_history.clear()
        self.last_valid_homography = None
        self.frames_without_homography = 0
        self.total_frames = 0
        self.shot_detector.reset()
