"""
Deteccion de lineas y caracteristicas del campo de futbol.
Usa tecnicas clasicas de vision por computador: HSV, Canny, Hough Lines.

MEJORADO: Enfoque en detectar SOLO lineas blancas del campo,
filtrando agresivamente el ruido (jugadores, publicidad, etc.)
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict


class FieldDetector:
    """Detecta lineas y caracteristicas del campo de futbol."""

    def __init__(
        self,
        green_lower: Tuple[int, int, int] = (30, 30, 30),
        green_upper: Tuple[int, int, int] = (90, 255, 255),
        white_lower: Tuple[int, int, int] = (0, 0, 200),
        white_upper: Tuple[int, int, int] = (180, 40, 255),
    ):
        """
        Args:
            green_lower: Limite inferior HSV para cesped
            green_upper: Limite superior HSV para cesped
            white_lower: Limite inferior HSV para lineas blancas
            white_upper: Limite superior HSV para lineas blancas
        """
        self.green_lower = np.array(green_lower)
        self.green_upper = np.array(green_upper)
        self.white_lower = np.array(white_lower)
        self.white_upper = np.array(white_upper)

        # Parametros de Hough Lines - MAS ESTRICTOS
        self.hough_rho = 1
        self.hough_theta = np.pi / 180
        self.hough_threshold = 100  # Mas alto para menos falsos positivos
        self.hough_min_line_length = 100  # Solo lineas largas
        self.hough_max_line_gap = 30  # Permitir gaps en lineas ocluidas

        # Parametros de Canny
        self.canny_low = 50
        self.canny_high = 150

        # Filtros adicionales
        self.min_line_length_ratio = 0.05  # Linea minima como % del ancho de imagen

    def segment_field(self, frame: np.ndarray) -> np.ndarray:
        """
        Segmenta el area del cesped usando color HSV.

        Args:
            frame: Frame BGR

        Returns:
            Mascara binaria del cesped
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.green_lower, self.green_upper)

        # Operaciones morfologicas para limpiar
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Rellenar huecos
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large)

        return mask

    def detect_white_lines(
        self, frame: np.ndarray, field_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Detecta lineas blancas usando segmentacion de color.
        MEJORADO: Filtrado mas agresivo para eliminar camisetas/publicidad.

        Args:
            frame: Frame BGR
            field_mask: Mascara opcional del cesped

        Returns:
            Mascara binaria de lineas blancas
        """
        # Convertir a HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, self.white_lower, self.white_upper)

        if field_mask is not None:
            # Solo lineas dentro del campo
            white_mask = cv2.bitwise_and(white_mask, white_mask, mask=field_mask)

        # FILTRO 1: Eliminar blobs grandes (jugadores, publicidad)
        # Las lineas del campo son delgadas, no blobs grandes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)

        # FILTRO 2: Erosion para eliminar ruido y adelgazar
        white_mask = cv2.erode(white_mask, kernel, iterations=1)

        # FILTRO 3: Dilatacion controlada para reconectar lineas
        white_mask = cv2.dilate(white_mask, kernel, iterations=1)

        # FILTRO 4: Eliminar componentes conectados pequeños
        white_mask = self._remove_small_components(white_mask, min_area=200)

        return white_mask

    def _remove_small_components(
        self, mask: np.ndarray, min_area: int = 100
    ) -> np.ndarray:
        """Elimina componentes conectados pequeños de la mascara."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        result = np.zeros_like(mask)

        for i in range(1, num_labels):  # Saltar fondo (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]

            # Filtrar por area minima
            if area < min_area:
                continue

            # Filtrar blobs muy cuadrados (probablemente no son lineas)
            aspect_ratio = max(width, height) / (min(width, height) + 1)
            if aspect_ratio < 2 and area > 500:
                # Blob cuadrado grande, probablemente no es linea
                continue

            result[labels == i] = 255

        return result

    def detect_lines(
        self, frame: np.ndarray, field_mask: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """
        Detecta lineas BLANCAS del campo usando segmentacion de color + Hough.
        Este enfoque es mas robusto que Canny porque solo busca lineas blancas.

        Args:
            frame: Frame BGR
            field_mask: Mascara opcional del cesped

        Returns:
            Array de lineas [N, 1, 4] donde cada linea es [x1, y1, x2, y2]
        """
        h, w = frame.shape[:2]

        # PASO 1: Detectar pixeles blancos (lineas del campo)
        white_mask = self.detect_white_lines(frame, field_mask)

        # PASO 2: Limpiar la mascara agresivamente
        # Eliminar objetos pequeños (jugadores, publicidad)
        kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel_small)

        # Adelgazar lineas para Hough
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel_small)

        # PASO 3: Esqueletizar para obtener lineas de 1 pixel
        # Esto ayuda a que Hough detecte mejor
        skeleton = cv2.ximgproc.thinning(white_mask) if hasattr(cv2, 'ximgproc') else white_mask

        # PASO 4: Detectar lineas con Hough
        min_length = max(int(w * self.min_line_length_ratio), self.hough_min_line_length)

        lines = cv2.HoughLinesP(
            skeleton if hasattr(cv2, 'ximgproc') else white_mask,
            rho=self.hough_rho,
            theta=self.hough_theta,
            threshold=self.hough_threshold,
            minLineLength=min_length,
            maxLineGap=self.hough_max_line_gap,
        )

        return lines

    def classify_lines(
        self, lines: Optional[np.ndarray], angle_threshold: float = 20.0
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Clasifica lineas en horizontales, verticales y otras.
        MEJORADO: Umbral de angulo mas amplio para perspectiva.

        Args:
            lines: Array de lineas de HoughLinesP
            angle_threshold: Umbral de angulo en grados

        Returns:
            Tupla de (horizontales, verticales, otras)
        """
        if lines is None:
            return [], [], []

        horizontal = []
        vertical = []
        other = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calcular longitud de la linea
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # Filtrar lineas muy cortas
            if length < 50:
                continue

            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

            # Normalizar angulo a [-90, 90]
            if angle > 90:
                angle -= 180
            elif angle < -90:
                angle += 180

            if abs(angle) < angle_threshold:
                # Horizontal (incluye lineas casi horizontales por perspectiva)
                horizontal.append(line[0])
            elif abs(abs(angle) - 90) < angle_threshold:
                # Vertical
                vertical.append(line[0])
            else:
                other.append(line[0])

        return horizontal, vertical, other

    def merge_similar_lines(
        self, lines: List[np.ndarray], distance_threshold: float = 20.0
    ) -> List[np.ndarray]:
        """
        Fusiona lineas similares que probablemente son la misma linea del campo.

        Args:
            lines: Lista de lineas
            distance_threshold: Distancia maxima para fusionar

        Returns:
            Lista de lineas fusionadas
        """
        if len(lines) <= 1:
            return lines

        merged = []
        used = [False] * len(lines)

        for i, line1 in enumerate(lines):
            if used[i]:
                continue

            # Encontrar lineas similares
            group = [line1]
            used[i] = True

            for j, line2 in enumerate(lines):
                if used[j]:
                    continue

                if self._lines_are_similar(line1, line2, distance_threshold):
                    group.append(line2)
                    used[j] = True

            # Fusionar grupo
            merged.append(self._merge_line_group(group))

        return merged

    def _lines_are_similar(
        self, line1: np.ndarray, line2: np.ndarray, threshold: float
    ) -> bool:
        """Verifica si dos lineas son similares."""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        # Calcular distancia entre centros
        cx1, cy1 = (x1 + x2) / 2, (y1 + y2) / 2
        cx2, cy2 = (x3 + x4) / 2, (y3 + y4) / 2
        dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

        if dist > threshold * 3:
            return False

        # Verificar angulos similares
        angle1 = np.arctan2(y2 - y1, x2 - x1)
        angle2 = np.arctan2(y4 - y3, x4 - x3)
        angle_diff = abs(angle1 - angle2)
        if angle_diff > np.pi:
            angle_diff = 2 * np.pi - angle_diff

        return angle_diff < 0.2  # ~11 grados

    def _merge_line_group(self, lines: List[np.ndarray]) -> np.ndarray:
        """Fusiona un grupo de lineas en una sola."""
        if len(lines) == 1:
            return lines[0]

        # Tomar los puntos extremos
        all_points = []
        for line in lines:
            x1, y1, x2, y2 = line
            all_points.extend([(x1, y1), (x2, y2)])

        # Encontrar los dos puntos mas lejanos
        max_dist = 0
        best_pair = (all_points[0], all_points[1])

        for i, p1 in enumerate(all_points):
            for j, p2 in enumerate(all_points):
                if i >= j:
                    continue
                dist = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
                if dist > max_dist:
                    max_dist = dist
                    best_pair = (p1, p2)

        return np.array(
            [best_pair[0][0], best_pair[0][1], best_pair[1][0], best_pair[1][1]]
        )

    def find_intersections(
        self,
        horizontal_lines: List[np.ndarray],
        vertical_lines: List[np.ndarray],
        frame_shape: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """
        Encuentra intersecciones entre lineas horizontales y verticales.

        Args:
            horizontal_lines: Lista de lineas horizontales
            vertical_lines: Lista de lineas verticales
            frame_shape: (height, width) del frame

        Returns:
            Lista de puntos de interseccion (x, y)
        """
        intersections = []
        h, w = frame_shape[:2]

        for h_line in horizontal_lines:
            for v_line in vertical_lines:
                point = self._line_intersection(h_line, v_line)
                if point is not None:
                    x, y = point
                    # Verificar que esta dentro del frame
                    if 0 <= x < w and 0 <= y < h:
                        intersections.append(point)

        # Eliminar duplicados cercanos
        intersections = self._remove_duplicate_points(intersections, threshold=15)

        return intersections

    def _line_intersection(
        self, line1: np.ndarray, line2: np.ndarray
    ) -> Optional[Tuple[int, int]]:
        """
        Calcula la interseccion de dos lineas.

        Args:
            line1: [x1, y1, x2, y2]
            line2: [x3, y3, x4, y4]

        Returns:
            Punto de interseccion o None si son paralelas
        """
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)

        return (int(px), int(py))

    def _remove_duplicate_points(
        self, points: List[Tuple[int, int]], threshold: float = 10
    ) -> List[Tuple[int, int]]:
        """Elimina puntos duplicados cercanos."""
        if len(points) <= 1:
            return points

        unique = []
        for point in points:
            is_dup = False
            for existing in unique:
                dist = np.sqrt(
                    (point[0] - existing[0]) ** 2 + (point[1] - existing[1]) ** 2
                )
                if dist < threshold:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(point)

        return unique

    def detect_center_circle(
        self, frame: np.ndarray, field_mask: Optional[np.ndarray] = None
    ) -> Optional[Tuple[int, int, int]]:
        """
        Intenta detectar el circulo central usando Hough Circles.

        Args:
            frame: Frame BGR
            field_mask: Mascara del cesped

        Returns:
            (x, y, radius) del circulo o None
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if field_mask is not None:
            gray = cv2.bitwise_and(gray, gray, mask=field_mask)

        gray = cv2.GaussianBlur(gray, (9, 9), 2)

        # Detectar circulos
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=100,
            param1=50,
            param2=30,
            minRadius=50,
            maxRadius=300,
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            # Retornar el circulo mas grande (probablemente el central)
            largest = max(circles[0], key=lambda c: c[2])
            return (int(largest[0]), int(largest[1]), int(largest[2]))

        return None

    def process_frame(
        self, frame: np.ndarray
    ) -> Dict:
        """
        Procesa un frame completo y retorna toda la informacion detectada.

        Args:
            frame: Frame BGR

        Returns:
            Diccionario con toda la informacion detectada
        """
        # Segmentar cesped
        field_mask = self.segment_field(frame)

        # Detectar lineas
        lines = self.detect_lines(frame, field_mask)

        # Clasificar lineas
        h_lines, v_lines, other_lines = self.classify_lines(lines)

        # Fusionar lineas similares
        h_lines = self.merge_similar_lines(h_lines)
        v_lines = self.merge_similar_lines(v_lines)

        # Encontrar intersecciones
        intersections = self.find_intersections(h_lines, v_lines, frame.shape)

        # Intentar detectar circulo central
        center_circle = self.detect_center_circle(frame, field_mask)

        return {
            "field_mask": field_mask,
            "horizontal_lines": h_lines,
            "vertical_lines": v_lines,
            "other_lines": other_lines,
            "intersections": intersections,
            "center_circle": center_circle,
            "num_lines": len(h_lines) + len(v_lines),
            "num_intersections": len(intersections),
        }

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: Dict,
        draw_mask: bool = False,
        draw_lines: bool = True,
        draw_intersections: bool = True,
    ) -> np.ndarray:
        """
        Dibuja las detecciones sobre el frame.

        Args:
            frame: Frame original
            detections: Diccionario de detecciones de process_frame
            draw_mask: Si dibujar la mascara del cesped
            draw_lines: Si dibujar las lineas
            draw_intersections: Si dibujar las intersecciones

        Returns:
            Frame con visualizaciones
        """
        result = frame.copy()

        if draw_mask and "field_mask" in detections:
            # Overlay verde semi-transparente
            overlay = result.copy()
            overlay[detections["field_mask"] > 0] = [0, 255, 0]
            cv2.addWeighted(overlay, 0.3, result, 0.7, 0, result)

        if draw_lines:
            # Lineas horizontales en azul
            for line in detections.get("horizontal_lines", []):
                x1, y1, x2, y2 = line
                cv2.line(result, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Lineas verticales en rojo
            for line in detections.get("vertical_lines", []):
                x1, y1, x2, y2 = line
                cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)

        if draw_intersections:
            for point in detections.get("intersections", []):
                cv2.circle(result, point, 8, (0, 255, 255), -1)
                cv2.circle(result, point, 8, (0, 0, 0), 2)

        # Circulo central en magenta
        if detections.get("center_circle") is not None:
            x, y, r = detections["center_circle"]
            cv2.circle(result, (x, y), r, (255, 0, 255), 2)
            cv2.circle(result, (x, y), 5, (255, 0, 255), -1)

        return result
