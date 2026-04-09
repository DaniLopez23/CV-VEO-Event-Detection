"""
Script de calibracion manual del campo.
Ejecutar este script para establecer la homografia antes de procesar el video.

Uso:
    python calibrate.py --video data/raw_videos/demo_2_video_1.mp4
"""
import cv2
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from geometry.calibration import CameraCalibrator


def main():
    parser = argparse.ArgumentParser(description="Calibracion de campo de futbol")
    parser.add_argument("--video", required=True, help="Ruta al video")
    parser.add_argument(
        "--output",
        default="data/calibration.json",
        help="Archivo de salida para calibracion",
    )
    parser.add_argument(
        "--frame", type=int, default=0, help="Numero de frame para calibrar"
    )
    args = parser.parse_args()

    # Cargar video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir {args.video}")
        return

    # Ir al frame especificado
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: No se pudo leer el frame {args.frame}")
        return

    print(f"Video cargado: {frame.shape[1]}x{frame.shape[0]}")

    # Calibrar
    calibrator = CameraCalibrator()

    print("\n" + "="*60)
    print("CALIBRACION MANUAL DEL CAMPO")
    print("="*60)
    print("\nVas a seleccionar puntos de referencia en el video")
    print("y asignarlos a puntos conocidos del campo de futbol.")
    print("\nNecesitas MINIMO 4 puntos para calcular la homografia.")
    print("Mas puntos = mejor precision.")
    print("\nPuntos recomendados:")
    print("  - Esquinas del campo (si son visibles)")
    print("  - Esquinas del area penal")
    print("  - Interseccion linea central con lineas laterales")
    print("  - Centro del campo")
    print("="*60)

    input("\nPresiona ENTER para comenzar...")

    # Calibracion interactiva
    homography = calibrator.calibrate_manual(frame)

    if homography is not None:
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

        # Guardar
        calibrator.save_calibration(args.output)

        print("\n" + "="*60)
        print("CALIBRACION COMPLETADA")
        print("="*60)
        print(f"Homografia guardada en: {args.output}")
        print(f"Puntos usados: {len(calibrator.image_points)}")
        print("\nAhora puedes ejecutar main.py y usara esta calibracion.")

        # Mostrar resultado
        print("\nMostrando resultado (presiona cualquier tecla para cerrar)...")

        # Transformar algunos puntos de prueba
        display = frame.copy()
        for img_pt, field_pt in zip(
            calibrator.image_points, calibrator.field_points
        ):
            cv2.circle(display, (int(img_pt[0]), int(img_pt[1])), 10, (0, 255, 0), -1)
            cv2.putText(
                display,
                f"({field_pt[0]:.0f}, {field_pt[1]:.0f})",
                (int(img_pt[0]) + 15, int(img_pt[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        cv2.imshow("Calibracion completada", display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("\nError: No se pudo calcular la homografia.")
        print("Asegurate de seleccionar al menos 4 puntos.")


if __name__ == "__main__":
    main()
