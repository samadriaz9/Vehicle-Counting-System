import cv2
from collections import defaultdict
import supervision as sv
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')
# Set up video capture
cap = cv2.VideoCapture("testing/D10_20240318093900.mp4")

# Store the track history
track_history = defaultdict(lambda: [])

# Create a dictionary to keep track of objects that have crossed the line
crossed_objects = {}

# Function to handle mouse events for drawing the line
def draw_line(event, x, y, flags, param):
    global START, END, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        START = sv.Point(x, y)
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            END = sv.Point(x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        END = sv.Point(x, y)
        drawing = False
    print(START)
    print(END)
# Initialize the line coordinates
START = sv.Point(0, 0)
END = sv.Point(0, 0)
drawing = False

# Create a window and set the mouse callback function
cv2.namedWindow('frame')
cv2.setMouseCallback('frame', draw_line)

while cap.isOpened():
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, classes=[2, 3, 5, 7], persist=True, verbose=False, tracker="bytetrack.yaml")

        try:
            if results is not None and len(results) > 0 and results[0].boxes is not None:
                # Get the boxes and track IDs
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()

                # Visualize the results on the frame
                annotated_frame = results[0].plot()

                # Plot the tracks and count objects crossing the line
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point
                    if len(track) > 30:  # retain 30 tracks for 30 frames
                        track.pop(0)

                    # Check if the object crosses the line
                    if START.x < x < END.x and abs(y - START.y) < 5:  # Assuming objects cross horizontally
                        if track_id not in crossed_objects:
                            crossed_objects[track_id] = True

                        # Annotate the object as it crosses the line
                        cv2.rectangle(annotated_frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 255, 0), 2)

                # Draw the dynamically drawn line on the frame
                cv2.line(annotated_frame, (START.x, START.y), (END.x, END.y), (0, 255, 0), 2)

                # Write the count of objects on each frame
                count_text = f"Objects crossed: {len(crossed_objects)}"
                cv2.putText(annotated_frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow('frame', annotated_frame)
                cv2.waitKey(1)
            else:
                # No objects detected in this frame
                cv2.imshow('frame', frame)
                cv2.waitKey(1)
        except Exception as e:
            # print(f"Error: {e}")
            pass
    else:
        break

# Release the video capture
cap.release()
cv2.destroyAllWindows()
