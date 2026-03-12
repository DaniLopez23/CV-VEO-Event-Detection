import cv2

def draw_boxes(frame, detections, color=(0,255,0), label_prefix=""):
    for det in detections:
        x1, y1, x2, y2, conf = map(int, det)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label_prefix}{conf:.2f}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame