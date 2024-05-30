import cv2
from collections import defaultdict
import numpy as np
from ultralytics import YOLO
import supervision as sv

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Set up video capture
cap = cv2.VideoCapture("testing/D10_20240318093900.mp4")

polygon_points = [(1016, 424), (980, 424), (934, 432), (887, 443), (860, 458), (740, 484), (575, 484), (530, 430),
                   (498, 421), (404, 427), (320, 436), (223, 432), (134, 426), (112, 436), (120, 497), (84, 530),
                   (3, 541), (6, 661), (65, 717), (276, 717), (472, 713), (742, 719), (921, 715), (1042, 715),
                   (1097, 676), (1121, 579), (1130, 495), (1110, 428), (1122, 393), (1124, 332), (1125, 256),
                   (1118, 171), (1111, 115), (1079, 94), (1042, 91), (997, 138), (978, 208), (968, 265), (966, 351),
                   (1025, 361), (1018, 425), (1016, 424)]

# Convert polygon points to numpy array
polygon_points = np.array(polygon_points)

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
    # print(START)
    # print(END)
# Initialize the line coordinates
START = sv.Point(467, 452)
END = sv.Point(1171, 446)
drawing = False

# Create a window and set the mouse callback function
cv2.namedWindow('frame')
cv2.setMouseCallback('frame', draw_line)

# Initialize object count
object_count = 0

while cap.isOpened():
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        result = model.track(source='rtsp://192.168.19.33:8080/h264_ulaw.sdp', classes=[2, 3, 5, 7], persist=True, verbose=False, tracker="bytetrack.yaml")
        print(result)
        if result is not None and len(result) > 0 and result[0].boxes is not None:
            # Get the detected boxes
            boxes = result[0].boxes.xywh.cpu()
            track_ids = []
            for box in result[0].boxes:
                if box.id is not None:
                    track_ids.append(box.id.item())
            # Iterate over the detected boxes
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

                x_int = int(x)
                y_int = int(y)

                # Check if the box is within the polygon region
                if cv2.pointPolygonTest(polygon_points, (x_int, y_int), False) >= 0:
                    # Annotate the object within the polygon
                    cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 255, 0), 2)

       
        # Draw the dynamically drawn line on the frame
       # Draw the polygon on the frame
        cv2.polylines(frame, [polygon_points], isClosed=True, color=(255, 0, 0), thickness=2)

        cv2.line(frame, (START.x, START.y), (END.x, END.y), (0, 255, 0), 2)
        cv2.polylines(frame, [polygon_points], isClosed=True, color=(255, 0, 0), thickness=2)

        # Write the count of objects on each frame
        count_text = f"Objects crossed: {len(crossed_objects)}"
        cv2.putText(frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('frame', frame)
        cv2.waitKey(1)
    else:
        break

cap.release()
cv2.destroyAllWindows()
