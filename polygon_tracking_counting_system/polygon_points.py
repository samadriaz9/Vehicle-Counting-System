import cv2
import numpy as np

# Create a window
cv2.namedWindow('frame')

# Variables for polygon drawing
drawing = False
polygon_points = []
temp_frame = None  # Initialize temp_frame here

# Function to handle mouse events for drawing the polygon
def draw_polygon(event, x, y, flags, param):
    global drawing, polygon_points, temp_frame

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        polygon_points.append((x, y))
        temp_frame = frame.copy()  # Make a copy of the frame when the drawing starts
        # Draw circles for all points
        for pt in polygon_points:
            cv2.circle(temp_frame, pt, 5, (0, 0, 255), -1)
        cv2.imshow('frame', temp_frame)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Temporarily copy the frame to draw the polygon and points
            temp_frame = frame.copy()
            for pt in polygon_points[:-1]:  # Draw all points except the last one
                cv2.circle(temp_frame, pt, 5, (0, 0, 255), -1)  # Draw red circles for selected points
            cv2.circle(temp_frame, (x, y), 5, (0, 0, 255), -1)  # Draw the current point
            cv2.line(temp_frame, polygon_points[-1], (x, y), (0, 0, 255), 2)  # Draw line while dragging mouse
            cv2.imshow('frame', temp_frame)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if len(polygon_points) > 2 and np.linalg.norm(np.array(polygon_points[0]) - np.array((x, y))) < 20:
            # Close the polygon
            polygon_points.append(polygon_points[0])
            cv2.polylines(frame, [np.array(polygon_points)], True, (0, 255, 0), thickness=2)

            # Create a mask for the polygon
            mask = np.zeros_like(frame)
            cv2.fillPoly(mask, [np.array(polygon_points)], (255, 255, 255))

            # Draw circles for all points
            for pt in polygon_points:
                cv2.circle(frame, pt, 5, (0, 0, 255), -1)

            # Apply bitwise operations to keep only the areas inside the polygon
            final_frame = cv2.bitwise_and(frame, mask)
            cv2.imshow('frame', final_frame)
            cv2.waitKey(0)
    print(polygon_points)

# Set mouse callback for drawing polygon
cv2.setMouseCallback('frame', draw_polygon)

# Set up video capture
cap = cv2.VideoCapture("testing/D10_20240318093900.mp4")

# Read the first frame
success, frame = cap.read()

# Display the first frame
cv2.imshow('frame', frame)

# Loop until the polygon is drawn
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if not drawing:
        continue

# Release the video capture
cap.release()

# Release the window
cv2.destroyAllWindows()
