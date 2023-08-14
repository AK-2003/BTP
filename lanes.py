import cv2
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def draw_lines(img, lines, color=(0, 0, 255), thickness=3):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

cap = cv2.VideoCapture('cars.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    height, width = frame.shape[:2]
    region_of_interest_vertices = [
        (width * 0.1, height),
        (width * 0.45, height * 0.6),
        (width * 0.55, height * 0.6),
        (width * 0.9, height)
    ]

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canny_frame = cv2.Canny(gray_frame, 50, 150)
    cropped_frame = region_of_interest(canny_frame, np.array([region_of_interest_vertices], np.int32))

    lines = cv2.HoughLinesP(cropped_frame, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=50)

    line_frame = np.zeros_like(frame)
    if lines is not None:
        draw_lines(line_frame, lines)

    lane_detected_frame = cv2.addWeighted(frame, 0.8, line_frame, 1, 1)
    cv2.imshow("Lane Detection", lane_detected_frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
