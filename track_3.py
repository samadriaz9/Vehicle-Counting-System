import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Set up video capture
cap = cv2.VideoCapture("testing/D10_20240318093900.mp4")

# Store the track history
track_history = defaultdict(lambda: [])

# Create a dictionary to keep track of objects that have crossed the line
crossed_objects = {}

# Function to handle mouse events for drawing the rectangle and line
def draw_rect_line(event, x, y, flags, param):
    global rect_start, rect_end, drawing_rect, rect_defined, line_start, line_end, drawing_line
    if event == cv2.EVENT_LBUTTONDOWN:
        if not rect_defined:
            rect_start = (x, y)
            drawing_rect = True
        else:
            line_start = (x, y)
            drawing_line = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing_rect:
            rect_end = (x, y)
        elif drawing_line:
            line_end = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        if drawing_rect:
            rect_end = (x, y)
            drawing_rect = False
            rect_defined = True
        elif drawing_line:
            line_end = (x, y)
            drawing_line = False

# Initialize variables for rectangle and line drawing
rect_start = (0, 0)
rect_end = (0, 0)
drawing_rect = False
rect_defined = False
line_start = (0, 0)
line_end = (0, 0)
drawing_line = False

# Create a window and set the mouse callback function
cv2.namedWindow('frame')
cv2.setMouseCallback('frame', draw_rect_line)

while cap.isOpened():
    success, frame = cap.read()

    if success:
        # Copy the frame for displaying purposes
        display_frame = frame.copy()

        if rect_defined:
            # Draw the rectangle on the display frame
            cv2.rectangle(display_frame, rect_start, rect_end, (0, 255, 0), 2)

            # Draw the line inside the rectangle
            if line_start != line_end:
                cv2.line(display_frame, line_start, line_end, (255, 0, 0), 2)

            # Crop the frame to the specified rectangle area
            x, y, w, h = cv2.boundingRect(np.array([rect_start, rect_end]))
            cropped_frame = frame[y:y+h, x:x+w]

            # Run YOLOv8 tracking on the cropped frame, persisting tracks between frames
            results = model.track(cropped_frame, classes=[2, 3, 5, 7], persist=True, verbose=False, tracker="bytetrack.yaml")

            try:
                if results is not None and len(results) > 0 and results[0].boxes is not None:
                    # Get the boxes and track IDs
                    boxes = results[0].boxes.xywh.cpu()
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    display_frame[y:y+h, x:x+w] = results[0].plot()
                    cv2.line(display_frame, line_start, line_end, (255, 0, 0), 2)
                    # Plot the tracks and count objects crossing the line
                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box
                        x += rect_start[0]  # Adjust for the position within the full frame
                        y += rect_start[1]
                        track = track_history[track_id]
                        track.append((float(x), float(y)))  # x, y center point
                        if len(track) > 30:  # retain 30 tracks for 30 frames
                            track.pop(0)

                        # Check if the object crosses the line
                        if line_start[0] < x < line_end[0] and abs(y - line_start[1]) < 5:  # Assuming objects cross horizontally
                            if track_id not in crossed_objects:
                                crossed_objects[track_id] = True

                            # Annotate the object as it crosses the line
                            cv2.rectangle(display_frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 255, 0), 2)

                    # Write the count of objects on each frame
                    count_text = f"Cars Leaved: {len(crossed_objects)}"
                    cv2.putText(display_frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
            except Exception as e:
                # print(f"Error: {e}")
                pass
        count_text = f"Cars Leaved: {len(crossed_objects)}"
        cv2.putText(display_frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Display the frame with rectangle, line, and detections
        cv2.imshow('frame', display_frame)
        cv2.waitKey(1)

    else:
        break

# Release the video capture
cap.release()
cv2.destroyAllWindows()
