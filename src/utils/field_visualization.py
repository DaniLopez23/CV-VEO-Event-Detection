"""
Visualizacion del campo de futbol 2D.
Genera una vista tactica del campo con posiciones de jugadores.
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config.field_config import FIELD_CONFIG


class FieldVisualizer:
    """Genera visualizaciones 2D del campo de futbol."""

    def __init__(self, config: Dict = None, scale: float = 10.0, padding: int = 20):
        """
        Args:
            config: Configuracion del campo (usa FIELD_CONFIG si None)
            scale: Pixeles por metro
            padding: Padding alrededor del campo en pixeles
        """
        self.config = config or FIELD_CONFIG
        self.scale = scale
        self.padding = padding

        # Dimensiones de la imagen del campo
        self.field_length = self.config["field_length"]
        self.field_width = self.config["field_width"]

        self.img_width = int(self.field_length * scale) + 2 * padding
        self.img_height = int(self.field_width * scale) + 2 * padding

        # Offset para centrar (origen en centro del campo)
        self.offset_x = self.img_width // 2
        self.offset_y = self.img_height // 2

        # Colores
        self.grass_color = (34, 139, 34)  # Verde cesped
        self.line_color = (255, 255, 255)  # Blanco
        self.player_colors = {
            "team_a": (255, 0, 0),  # Rojo
            "team_b": (0, 0, 255),  # Azul
            "ball": (255, 255, 0),  # Amarillo
        }

    def field_to_image(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convierte coordenadas del campo (metros) a coordenadas de imagen (pixeles).

        Args:
            x: Coordenada X en el campo (metros)
            y: Coordenada Y en el campo (metros)

        Returns:
            (px, py) coordenadas en la imagen
        """
        px = int(x * self.scale + self.offset_x)
        py = int(-y * self.scale + self.offset_y)  # Y invertido
        return (px, py)

    def image_to_field(self, px: int, py: int) -> Tuple[float, float]:
        """
        Convierte coordenadas de imagen a coordenadas del campo.

        Args:
            px: Coordenada X en pixeles
            py: Coordenada Y en pixeles

        Returns:
            (x, y) coordenadas en el campo (metros)
        """
        x = (px - self.offset_x) / self.scale
        y = -(py - self.offset_y) / self.scale
        return (x, y)

    def draw_field(self) -> np.ndarray:
        """
        Dibuja el campo de futbol 2D vacio.

        Returns:
            Imagen del campo
        """
        img = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
        img[:] = self.grass_color

        # Dimensiones en pixeles
        half_l = int(self.field_length * self.scale / 2)
        half_w = int(self.field_width * self.scale / 2)

        # Contorno del campo
        cv2.rectangle(
            img,
            (self.offset_x - half_l, self.offset_y - half_w),
            (self.offset_x + half_l, self.offset_y + half_w),
            self.line_color,
            2,
        )

        # Linea central
        cv2.line(
            img,
            (self.offset_x, self.offset_y - half_w),
            (self.offset_x, self.offset_y + half_w),
            self.line_color,
            2,
        )

        # Circulo central
        center_radius = int(self.config["center_circle_radius"] * self.scale)
        cv2.circle(img, (self.offset_x, self.offset_y), center_radius, self.line_color, 2)

        # Punto central
        cv2.circle(img, (self.offset_x, self.offset_y), 5, self.line_color, -1)

        # Areas penales
        self._draw_penalty_area(img, "left")
        self._draw_penalty_area(img, "right")

        # Areas de meta
        self._draw_goal_area(img, "left")
        self._draw_goal_area(img, "right")

        # Puntos de penal
        self._draw_penalty_spots(img)

        # Porterias
        self._draw_goals(img)

        # Arcos del area penal (semicirculos)
        self._draw_penalty_arcs(img)

        return img

    def _draw_penalty_area(self, img: np.ndarray, side: str):
        """Dibuja el area penal."""
        pa_length = int(self.config["penalty_area_length"] * self.scale)
        pa_half_width = int(self.config["penalty_area_width"] * self.scale / 2)
        half_l = int(self.field_length * self.scale / 2)

        if side == "left":
            x1 = self.offset_x - half_l
            x2 = self.offset_x - half_l + pa_length
        else:
            x1 = self.offset_x + half_l - pa_length
            x2 = self.offset_x + half_l

        y1 = self.offset_y - pa_half_width
        y2 = self.offset_y + pa_half_width

        cv2.rectangle(img, (x1, y1), (x2, y2), self.line_color, 2)

    def _draw_goal_area(self, img: np.ndarray, side: str):
        """Dibuja el area de meta (area chica)."""
        ga_length = int(self.config["goal_area_length"] * self.scale)
        ga_half_width = int(self.config["goal_area_width"] * self.scale / 2)
        half_l = int(self.field_length * self.scale / 2)

        if side == "left":
            x1 = self.offset_x - half_l
            x2 = self.offset_x - half_l + ga_length
        else:
            x1 = self.offset_x + half_l - ga_length
            x2 = self.offset_x + half_l

        y1 = self.offset_y - ga_half_width
        y2 = self.offset_y + ga_half_width

        cv2.rectangle(img, (x1, y1), (x2, y2), self.line_color, 2)

    def _draw_penalty_spots(self, img: np.ndarray):
        """Dibuja los puntos de penal."""
        penalty_dist = int(self.config["penalty_spot_distance"] * self.scale)
        half_l = int(self.field_length * self.scale / 2)

        # Izquierdo
        cv2.circle(
            img,
            (self.offset_x - half_l + penalty_dist, self.offset_y),
            4,
            self.line_color,
            -1,
        )

        # Derecho
        cv2.circle(
            img,
            (self.offset_x + half_l - penalty_dist, self.offset_y),
            4,
            self.line_color,
            -1,
        )

    def _draw_goals(self, img: np.ndarray):
        """Dibuja las porterias."""
        goal_half_width = int(self.config["goal_width"] * self.scale / 2)
        half_l = int(self.field_length * self.scale / 2)
        goal_depth = 20  # Pixeles de profundidad visual

        # Porteria izquierda
        cv2.rectangle(
            img,
            (self.offset_x - half_l - goal_depth, self.offset_y - goal_half_width),
            (self.offset_x - half_l, self.offset_y + goal_half_width),
            self.line_color,
            2,
        )

        # Porteria derecha
        cv2.rectangle(
            img,
            (self.offset_x + half_l, self.offset_y - goal_half_width),
            (self.offset_x + half_l + goal_depth, self.offset_y + goal_half_width),
            self.line_color,
            2,
        )

    def _draw_penalty_arcs(self, img: np.ndarray):
        """Dibuja los arcos del area penal."""
        arc_radius = int(self.config["penalty_arc_radius"] * self.scale)
        penalty_dist = int(self.config["penalty_spot_distance"] * self.scale)
        half_l = int(self.field_length * self.scale / 2)

        # Arco izquierdo
        center_left = (self.offset_x - half_l + penalty_dist, self.offset_y)
        cv2.ellipse(img, center_left, (arc_radius, arc_radius), 0, -53, 53, self.line_color, 2)

        # Arco derecho
        center_right = (self.offset_x + half_l - penalty_dist, self.offset_y)
        cv2.ellipse(img, center_right, (arc_radius, arc_radius), 0, 127, 233, self.line_color, 2)

    def plot_positions(
        self,
        field_img: np.ndarray,
        positions: List[Tuple[float, float]],
        color: Tuple[int, int, int] = (255, 0, 0),
        radius: int = 8,
        team_label: str = "",
    ) -> np.ndarray:
        """
        Dibuja posiciones de jugadores en el campo 2D.

        Args:
            field_img: Imagen del campo
            positions: Lista de posiciones (x, y) en metros
            color: Color BGR de los marcadores
            radius: Radio de los marcadores
            team_label: Etiqueta opcional para mostrar

        Returns:
            Imagen con posiciones dibujadas
        """
        img = field_img.copy()

        for i, pos in enumerate(positions):
            x, y = pos
            # Verificar que el punto esta dentro de limites razonables
            if abs(x) > 60 or abs(y) > 40:
                continue

            px, py = self.field_to_image(x, y)

            # Verificar que esta dentro de la imagen
            if 0 <= px < self.img_width and 0 <= py < self.img_height:
                cv2.circle(img, (px, py), radius, color, -1)
                cv2.circle(img, (px, py), radius, (0, 0, 0), 1)

                # Numero de jugador
                if team_label:
                    cv2.putText(
                        img,
                        str(i + 1),
                        (px - 4, py + 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1,
                    )

        return img

    def plot_ball(
        self,
        field_img: np.ndarray,
        position: Tuple[float, float],
        radius: int = 6,
    ) -> np.ndarray:
        """
        Dibuja la posicion del balon.

        Args:
            field_img: Imagen del campo
            position: Posicion (x, y) en metros
            radius: Radio del marcador

        Returns:
            Imagen con el balon dibujado
        """
        img = field_img.copy()
        x, y = position

        if abs(x) > 60 or abs(y) > 40:
            return img

        px, py = self.field_to_image(x, y)

        if 0 <= px < self.img_width and 0 <= py < self.img_height:
            cv2.circle(img, (px, py), radius, self.player_colors["ball"], -1)
            cv2.circle(img, (px, py), radius, (0, 0, 0), 2)

        return img

    def plot_trajectory(
        self,
        field_img: np.ndarray,
        positions: List[Tuple[float, float]],
        color: Tuple[int, int, int] = (255, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Dibuja una trayectoria en el campo.

        Args:
            field_img: Imagen del campo
            positions: Lista de posiciones en orden temporal
            color: Color de la trayectoria
            thickness: Grosor de la linea

        Returns:
            Imagen con trayectoria dibujada
        """
        img = field_img.copy()

        if len(positions) < 2:
            return img

        points = []
        for pos in positions:
            x, y = pos
            if abs(x) <= 60 and abs(y) <= 40:
                px, py = self.field_to_image(x, y)
                points.append((px, py))

        for i in range(len(points) - 1):
            cv2.line(img, points[i], points[i + 1], color, thickness)

        return img

    def create_frame(
        self,
        player_positions: List[Tuple[float, float]] = None,
        ball_position: Tuple[float, float] = None,
        team_a_positions: List[Tuple[float, float]] = None,
        team_b_positions: List[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """
        Crea un frame completo con campo y todas las posiciones.

        Args:
            player_positions: Posiciones de jugadores (si no hay equipos separados)
            ball_position: Posicion del balon
            team_a_positions: Posiciones equipo A
            team_b_positions: Posiciones equipo B

        Returns:
            Imagen del campo con todas las posiciones
        """
        img = self.draw_field()

        # Jugadores
        if team_a_positions:
            img = self.plot_positions(
                img, team_a_positions, self.player_colors["team_a"], team_label="A"
            )

        if team_b_positions:
            img = self.plot_positions(
                img, team_b_positions, self.player_colors["team_b"], team_label="B"
            )

        if player_positions and not (team_a_positions or team_b_positions):
            img = self.plot_positions(img, player_positions, (0, 0, 255))

        # Balon
        if ball_position:
            img = self.plot_ball(img, ball_position)

        return img

    def get_dimensions(self) -> Tuple[int, int]:
        """Retorna las dimensiones de la imagen del campo."""
        return (self.img_width, self.img_height)
