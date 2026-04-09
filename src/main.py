"""
Pipeline principal de deteccion de eventos en partidos de futbol.
Detecta jugadores, balon, lineas del campo y calcula homografias.

Uso:
    # Primero calibrar (una sola vez):
    python calibrate.py --video data/raw_videos/mi_video.mp4

    # Luego procesar:
    python main.py
"""
import cv2
import numpy as np
import os
import sys
import argparse
from typing import List, Optional, Tuple

# Agregar src al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.video import load_video, get_frames
from detection.player_detection import PlayerDetector
from detection.ball_detector import BallDetector
from detection.field_detection import FieldDetector
from geometry.calibration import CameraCalibrator
from utils.visualization_elements import draw_boxes
from utils.field_visualization import FieldVisualizer


# Limites oficiales FIFA en metros (origen en el centro del campo)
FIELD_X_LIMIT = 52.5
FIELD_Y_LIMIT = 34.0


def is_valid_field_point(point: Optional[Tuple[float, float]]) -> bool:
    """Valida que una posicion transformada sea finita y este dentro del campo."""
    if point is None:
        return False
    x, y = point
    if not (np.isfinite(x) and np.isfinite(y)):
        return False
    return abs(x) <= FIELD_X_LIMIT and abs(y) <= FIELD_Y_LIMIT


def filter_player_detections(
    detections: List[List[float]], frame_h: int, min_conf: float
) -> List[List[float]]:
    """
    Filtra detecciones de jugador con reglas geometricas simples.
    Reduce falsos positivos (gradas, carteles, publico).
    """
    filtered: List[List[float]] = []
    for det in detections:
        x1, y1, x2, y2, conf = det
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        aspect_ratio = w / h

        # Regla simple: persona no extremadamente ancha y con alto minimo
        if h < 25:
            continue
        if aspect_ratio < 0.18 or aspect_ratio > 1.2:
            continue
        # Ignorar detecciones demasiado arriba de la imagen
        if y2 < frame_h * 0.20:
            continue
        # Umbral de confianza mas estricto para estabilizar
        if conf < min_conf:
            continue

        filtered.append(det)

    # Limitar cantidad para evitar poblar el mapa 2D con ruido
    filtered.sort(key=lambda d: d[4], reverse=True)
    return filtered[:30]


def smooth_ball_position(
    candidate: Optional[Tuple[float, float]],
    previous: Optional[Tuple[float, float]],
    max_jump_m: float = 8.0,
) -> Optional[Tuple[float, float]]:
    """
    Rechaza saltos bruscos del balon en coordenadas de campo.
    Si salta demasiado, conserva la ultima posicion valida.
    """
    if candidate is None:
        return previous
    if previous is None:
        return candidate

    dx = candidate[0] - previous[0]
    dy = candidate[1] - previous[1]
    if np.hypot(dx, dy) > max_jump_m:
        return previous

    return candidate


def main():
    parser = argparse.ArgumentParser(description="Procesamiento de video de futbol")
    parser.add_argument(
        "--video",
        default="data/raw_videos/demo_2_video_1.mp4",
        help="Ruta al video",
    )
    parser.add_argument(
        "--calibration",
        default="data/calibration.json",
        help="Archivo de calibracion",
    )
    parser.add_argument(
        "--output-dir",
        default="data/outputs/detection_videos",
        help="Directorio de salida",
    )
    parser.add_argument(
        "--show-lines",
        action="store_true",
        help="Mostrar lineas detectadas (debug)",
    )
    parser.add_argument(
        "--player-conf",
        type=float,
        default=0.45,
        help="Confianza minima para mapear jugadores a 2D",
    )
    parser.add_argument(
        "--ball-conf",
        type=float,
        default=0.55,
        help="Confianza minima para detectar/mapeo de balon",
    )
    args = parser.parse_args()

    # Configuracion
    video_path = args.video
    calibration_path = args.calibration
    output_dir = args.output_dir

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, f"{video_name}_detected.mp4")
    output_2d_path = os.path.join(output_dir, f"{video_name}_2d_view.mp4")
    output_combined_path = os.path.join(output_dir, f"{video_name}_combined.mp4")

    # Crear directorios de salida
    os.makedirs(output_dir, exist_ok=True)

    # Cargar video
    cap = load_video(video_path)
    if cap is None:
        print(f"Error: No se pudo abrir {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height} @ {fps:.1f} fps, {total_frames} frames")

    # Cargar calibracion
    calibrator = CameraCalibrator()
    has_calibration = calibrator.load_calibration(calibration_path)

    if has_calibration:
        print(f"Calibracion cargada: {calibration_path}")
        print(f"  Puntos de calibracion: {len(calibrator.image_points)}")
    else:
        print("\n" + "="*60)
        print("ADVERTENCIA: No hay calibracion del campo")
        print("="*60)
        print("La vista 2D no estara disponible.")
        print("Para calibrar, ejecuta:")
        print(f"  python calibrate.py --video {video_path}")
        print("="*60 + "\n")

    # Inicializar detectores
    player_detector = PlayerDetector()
    ball_detector = BallDetector(
        use_specialized=True,
        specialized_model_path="models/football_ball.pt"
    )
    field_detector = FieldDetector()

    # Inicializar visualizador
    field_visualizer = FieldVisualizer(scale=8.0)
    field_w, field_h = field_visualizer.get_dimensions()

    # VideoWriters
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Video combinado (solo si hay calibracion)
    out_2d = None
    out_combined = None
    combined_width = width + field_w
    combined_height = max(height, field_h)

    if has_calibration:
        out_2d = cv2.VideoWriter(output_2d_path, fourcc, fps, (field_w, field_h))
        out_combined = cv2.VideoWriter(
            output_combined_path, fourcc, fps, (combined_width, combined_height)
        )

    # Estadisticas
    total_players_detected = 0
    total_balls_detected = 0
    last_ball_position_field: Optional[Tuple[float, float]] = None

    print("Procesando video...")

    for frame_idx, frame in get_frames(cap):
        # Progreso
        if frame_idx % 100 == 0:
            print(f"  Frame {frame_idx}/{total_frames}")

        # 1. Deteccion de jugadores
        player_dets = player_detector.detect_players(frame)
        player_dets = filter_player_detections(
            player_dets, frame_h=height, min_conf=args.player_conf
        )
        total_players_detected += len(player_dets)

        # 2. Deteccion de balon
        ball_dets = ball_detector.detect_ball(frame, conf_thresh=args.ball_conf)
        total_balls_detected += len(ball_dets)

        # 3. Deteccion del campo (para visualizacion de debug)
        field_info = None
        if args.show_lines:
            field_info = field_detector.process_frame(frame)

        # 4. Transformar posiciones si hay calibracion
        player_positions_field = []
        ball_position_field = None

        if has_calibration:
            # Jugadores (usar punto inferior central del bbox = pies)
            for det in player_dets:
                x1, y1, x2, y2, conf = det
                foot_point = ((x1 + x2) / 2, y2)
                field_pos = calibrator.transform_point(foot_point)
                if is_valid_field_point(field_pos):
                    player_positions_field.append(field_pos)

            # Balon
            ball_center = ball_detector.get_ball_center(ball_dets)
            if ball_center is not None:
                candidate_ball = calibrator.transform_point(ball_center)
                if is_valid_field_point(candidate_ball):
                    ball_position_field = smooth_ball_position(
                        candidate_ball, last_ball_position_field
                    )
                else:
                    ball_position_field = last_ball_position_field

                last_ball_position_field = ball_position_field

        # 5. Visualizacion - Frame original con detecciones
        frame_drawn = frame.copy()

        # Dibujar lineas detectadas (debug)
        if args.show_lines and field_info:
            frame_drawn = field_detector.draw_detections(
                frame_drawn, field_info, draw_mask=False, draw_lines=True
            )

        # Dibujar jugadores
        frame_drawn = draw_boxes(
            frame_drawn, player_dets, color=(0, 255, 0), label_prefix="P:"
        )

        # Dibujar balon
        frame_drawn = draw_boxes(
            frame_drawn, ball_dets, color=(0, 0, 255), label_prefix="B:"
        )

        # Info en pantalla
        info_text = f"Players: {len(player_dets)} | Ball: {len(ball_dets)}"
        if has_calibration:
            info_text += f" | Mapped: {len(player_positions_field)}"

        cv2.putText(
            frame_drawn,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Indicador de calibracion
        status_color = (0, 255, 0) if has_calibration else (0, 0, 255)
        status_text = "Calibrated" if has_calibration else "No calibration"
        cv2.putText(
            frame_drawn,
            status_text,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            status_color,
            2,
        )

        # Guardar frame con detecciones
        out.write(frame_drawn)

        # 6. Vista 2D (solo si hay calibracion)
        if has_calibration and out_2d is not None:
            field_view = field_visualizer.create_frame(
                player_positions=player_positions_field,
                ball_position=ball_position_field,
            )
            out_2d.write(field_view)

            # Video combinado
            if out_combined is not None:
                combined = np.zeros(
                    (combined_height, combined_width, 3), dtype=np.uint8
                )
                combined[:height, :width] = frame_drawn
                y_offset = (combined_height - field_h) // 2
                combined[y_offset : y_offset + field_h, width:] = field_view
                out_combined.write(combined)

    # Liberar recursos
    cap.release()
    out.release()
    if out_2d is not None:
        out_2d.release()
    if out_combined is not None:
        out_combined.release()

    # Estadisticas finales
    print("\n" + "="*50)
    print("PROCESAMIENTO COMPLETADO")
    print("="*50)
    print(f"Frames procesados: {total_frames}")
    print(f"Jugadores detectados (total): {total_players_detected}")
    print(f"Balones detectados (total): {total_balls_detected}")
    print(f"\nVideos guardados:")
    print(f"  - Detecciones: {output_path}")

    if has_calibration:
        print(f"  - Vista 2D: {output_2d_path}")
        print(f"  - Combinado: {output_combined_path}")


if __name__ == "__main__":
    main()
