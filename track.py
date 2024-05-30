import cv2
from collections import defaultdict
import numpy as np
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

# Define polygon points
polygon_points = [(1016, 424), (980, 424), (934, 432), (887, 443), (860, 458), (740, 484), (575, 484), (530, 430),
                   (498, 421), (404, 427), (320, 436), (223, 432), (134, 426), (112, 436), (120, 497), (84, 530), 
                   (3, 541), (6, 661), (65, 717), (276, 717), (472, 713), (742, 719), (921, 715), (1042, 715), 
                   (1097, 676), (1121, 579), (1130, 495), (1110, 428), (1122, 393), (1124, 332), (1125, 256), 
                   (1118, 171), (1111, 115), (1079, 94), (1042, 91), (997, 138), (978, 208), (968, 265), (966, 351),
                   (1025, 361), (1018, 425), (1016, 424)]

# Create a black image (mask) of the same size as the frame
mask = np.zeros((720, 1280), dtype=np.uint8)

# Draw the polygon on the mask
cv2.fillPoly(mask, [np.array(polygon_points)], 255)

while cap.isOpened():
    success, frame = cap.read()

    if success:
        # Apply the mask on the frame using bitwise AND operation
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

        # # Run YOLOv8 tracking on the masked frame, persisting tracks between frames
        # results = model.track(masked_frame, classes=[2, 3, 5, 7], persist=True, verbose=False, tracker="bytetrack.yaml")

        # try:
        #     if results is not None and len(results) > 0 and results[0].boxes is not None:
        #         # Get the boxes and track IDs
        #         boxes = results[0].boxes.xywh.cpu()
        #         track_ids = results[0].boxes.id.int().cpu().tolist()

        #         # Visualize the results on the frame
        #         annotated_frame = results[0].plot()

        #         # Plot the tracks and count objects crossing the line
        #         for box, track_id in zip(boxes, track_ids):
        #             x, y, w, h = box
        #             track = track_history[track_id]
        #             track.append((float(x), float(y)))  # x, y center point
        #             if len(track) > 30:  # retain 30 tracks for 30 frames
        #                 track.pop(0)

        #             # Check if the object crosses the line
        #             if START.x < x < END.x and abs(y - START.y) < 5:  # Assuming objects cross horizontally
        #                 if track_id not in crossed_objects:
        #                     crossed_objects[track_id] = True

        #                 # Annotate the object as it crosses the line
        #                 cv2.rectangle(annotated_frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 255, 0), 2)

        #         # Draw the dynamically drawn line on the frame
        #         cv2.line(annotated_frame, (START.x, START.y), (END.x, END.y), (0, 255, 0), 2)

        #         # Write the count of objects on each frame
        #         count_text = f"Objects crossed: {len(crossed_objects)}"
        #         cv2.putText(annotated_frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        #         cv2.imshow('frame', annotated_frame)
        #         cv2.waitKey(1)
        #     else:
                # No objects detected in this frame
        cv2.imshow('frame', masked_frame)
        cv2.waitKey(1)
#         except Exception as e:
#             # print(f"Error: {e}")
#             pass
#     else:
#         break

# # Release the video capture
cap.release()
cv2.destroyAllWindows()
