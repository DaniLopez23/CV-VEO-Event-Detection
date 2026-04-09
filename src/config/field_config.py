"""
Configuracion del campo de futbol segun dimensiones FIFA.
Sistema de coordenadas: origen en centro del campo.
X: largo del campo (-52.5 a +52.5 metros)
Y: ancho del campo (-34 a +34 metros)
"""

FIELD_CONFIG = {
    # Dimensiones totales del campo (FIFA estandar)
    "field_length": 105.0,  # metros
    "field_width": 68.0,    # metros

    # Area penal
    "penalty_area_length": 16.5,
    "penalty_area_width": 40.32,

    # Area de meta (area chica)
    "goal_area_length": 5.5,
    "goal_area_width": 18.32,

    # Punto penal
    "penalty_spot_distance": 11.0,

    # Circulo central
    "center_circle_radius": 9.15,

    # Arco del area penal
    "penalty_arc_radius": 9.15,

    # Porteria
    "goal_width": 7.32,

    # Esquinas
    "corner_arc_radius": 1.0,
}

# Puntos de referencia clave del campo (en metros)
# Origen en el centro del campo
FIELD_KEYPOINTS = {
    # Esquinas del campo
    "corner_top_left": (-52.5, 34.0),
    "corner_top_right": (52.5, 34.0),
    "corner_bottom_left": (-52.5, -34.0),
    "corner_bottom_right": (52.5, -34.0),

    # Linea central
    "center_top": (0.0, 34.0),
    "center_bottom": (0.0, -34.0),
    "center_spot": (0.0, 0.0),

    # Area penal izquierda
    "penalty_area_left_top_corner": (-52.5, 20.16),
    "penalty_area_left_bottom_corner": (-52.5, -20.16),
    "penalty_area_left_top": (-36.0, 20.16),
    "penalty_area_left_bottom": (-36.0, -20.16),
    "penalty_spot_left": (-41.5, 0.0),

    # Area penal derecha
    "penalty_area_right_top_corner": (52.5, 20.16),
    "penalty_area_right_bottom_corner": (52.5, -20.16),
    "penalty_area_right_top": (36.0, 20.16),
    "penalty_area_right_bottom": (36.0, -20.16),
    "penalty_spot_right": (41.5, 0.0),

    # Area chica izquierda
    "goal_area_left_top_corner": (-52.5, 9.16),
    "goal_area_left_bottom_corner": (-52.5, -9.16),
    "goal_area_left_top": (-47.0, 9.16),
    "goal_area_left_bottom": (-47.0, -9.16),

    # Area chica derecha
    "goal_area_right_top_corner": (52.5, 9.16),
    "goal_area_right_bottom_corner": (52.5, -9.16),
    "goal_area_right_top": (47.0, 9.16),
    "goal_area_right_bottom": (47.0, -9.16),

    # Postes de porteria
    "goal_post_left_top": (-52.5, 3.66),
    "goal_post_left_bottom": (-52.5, -3.66),
    "goal_post_right_top": (52.5, 3.66),
    "goal_post_right_bottom": (52.5, -3.66),
}
