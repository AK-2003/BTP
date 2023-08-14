import cv2
import numpy as np

# Define rectangle variables
rect_start = None
rect_end = None
rect_color = (0, 255, 0)
rect_thickness = 2

# Define grid variables
grid_size = (10, 10)
grid_color = (255, 255, 255)
cell_width = None
cell_height = None

# Define vehicle counting variables
vehicle_count = 0
last_frame = None
vehicle_color = (0, 0, 255)
vehicle_thickness = 2

# Define vehicle detection
fgbg = cv2.createBackgroundSubtractorMOG2()

# Define video capture
cap = cv2.VideoCapture('cars.mp4')

# Mouse callback function to draw rectangle
def draw_rectangle(event, x, y, flags, param):
    global rect_start, rect_end
    if event == cv2.EVENT_LBUTTONDOWN:
        rect_start = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        rect_end = (x, y)

# Create window and set mouse callback function
cv2.namedWindow('Vehicle Counting')
cv2.setMouseCallback('Vehicle Counting', draw_rectangle)

# Main loop
while cap.isOpened():
    # Read frame
    ret, frame = cap.read()
    if not ret:
        break
    # Draw rectangle
    if rect_start is not None and rect_end is not None:
        cv2.rectangle(frame, rect_start, rect_end, rect_color, rect_thickness)
        # Create grid
        mask = np.zeros_like(frame[:,:,0])
        x1, y1 = rect_start
        x2, y2 = rect_end
        mask[y1:y2, x1:x2] = 1
        # Create grid
        if cell_width is None or cell_height is None:
            height, width, _ = frame.shape
            cell_width = (x2 - x1) // grid_size[0]
            cell_height = (y2 - y1) // grid_size[1]
        for x in range(x1, x2, cell_width):
            cv2.line(frame, (x, y1), (x, y2), grid_color, 1)
        for y in range(y1, y2, cell_height):
            cv2.line(frame, (x1, y), (x2, y), grid_color, 1)
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Grayscale Image', gray)
        # Convert to binary image
        thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
        cv2.imshow('binnary  Image', thresh)
        # Detect vehicles
        fgmask = fgbg.apply(thresh)
        cv2.imshow('sepration', fgmask)
        fgmask = cv2.bitwise_and(fgmask, fgmask, mask=mask)

        cv2.imshow('background sepration', fgmask)
        
        contours,hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
         (x, y, w, h) = cv2.boundingRect(contour)
         if x >= rect_start[0] and x + w <= rect_end[0] and y >= rect_start[1] and y + h <= rect_end[1]:
             cv2.rectangle(frame, (x, y), (x + w, y + h), vehicle_color, vehicle_thickness)
             vehicle_count += 1

        # Update vehicle count
        if last_frame is not None:
            diff = cv2.absdiff(thresh, last_frame)
            vehicle_count += cv2.countNonZero(diff)
        last_frame = thresh
    # Display vehicle count
    #cv2.putText(frame, f'Vehicles: {vehicle_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    # Handle events
    cv2.imshow('Vehicle Counting', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('r'):
        rect_start = None
        rect_end = None
        cell_width = None
        cell_height = None
        vehicle_count = 0
        last_frame = None
    elif key == ord('s'):
        cv2.imwrite('output.jpg', frame)
    elif key == ord(' '):
        if rect_start is None:
            rect_start = (0, 0)
        elif rect_end is None:
            rect_end = (frame.shape[1], frame.shape[0])
        else:
            rect_start = None
            rect_end = None
            cell_width = None
            cell_height = None
            vehicle_count = 0
            last_frame = None

# Clean up
cap.release()
cv2.destroyAllWindows()