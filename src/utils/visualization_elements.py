import cv2

def draw_boxes(frame, detections):
    for det in detections:
        x1, y1, x2, y2, conf = map(int, det)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"{conf:.2f}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return frame