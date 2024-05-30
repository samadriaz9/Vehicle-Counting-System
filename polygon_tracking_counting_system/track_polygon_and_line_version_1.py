from dataloaders import LoadStreams
import cv2
from collections import defaultdict
import numpy as np
from ultralytics import YOLO
import supervision as sv
model = YOLO('yolov8n.pt')
polygon_points = [(1016, 424), (980, 424), (934, 432), (887, 443), (860, 458), (740, 484), (575, 484), (530, 430),
                   (498, 421), (404, 427), (320, 436), (223, 432), (134, 426), (112, 436), (120, 497), (84, 530),
                   (3, 541), (6, 661), (65, 717), (276, 717), (472, 713), (742, 719), (921, 715), (1042, 715),
                   (1097, 676), (1121, 579), (1130, 495), (1110, 428), (1122, 393), (1124, 332), (1125, 256),
                   (1118, 171), (1111, 115), (1079, 94), (1042, 91), (997, 138), (978, 208), (968, 265), (966, 351),
                   (1025, 361), (1018, 425), (1016, 424)]

# Convert polygon points to numpy array
polygon_points = np.array(polygon_points)
START = sv.Point(467, 452)
END = sv.Point(1171, 446)
# Store the track history
track_history = defaultdict(lambda: [])
source = 'rtsp://192.168.138.164:8080/h264_ulaw.sdp'
dataset = LoadStreams(source)
# Create a dictionary to keep track of objects that have crossed the line
crossed_objects = {}
for im0s in dataset:
    for frame in im0s:

 # Run YOLOv8 tracking on the frame, persisting tracks between frames
        result = model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")
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
# # Configure the tracking parameters and run the tracker
# model = YOLO('yolov8n.pt')
# # results = model.track(source="https://youtu.be/LNwODJXcvt4", conf=0.3, iou=0.5, show=True)
# for result in model.track(source="https://youtu.be/LNwODJXcvt4", conf=0.3,verbose=False ,iou=0.5, show=True):
#     print(result)
#     print('result printed')




# from ultralytics import YOLO












# from ultralytics import YOLO
# import cv2
# from collections import defaultdict
# import numpy as np
# from ultralytics import YOLO
# import supervision as sv
# START = sv.Point(467, 452)
# END = sv.Point(1171, 446)
# # Load the YOLOv8 model
# # model = YOLO('yolov8n.pt')

# # Set up video capture


# polygon_points = [(1016, 424), (980, 424), (934, 432), (887, 443), (860, 458), (740, 484), (575, 484), (530, 430),
#                    (498, 421), (404, 427), (320, 436), (223, 432), (134, 426), (112, 436), (120, 497), (84, 530),
#                    (3, 541), (6, 661), (65, 717), (276, 717), (472, 713), (742, 719), (921, 715), (1042, 715),
#                    (1097, 676), (1121, 579), (1130, 495), (1110, 428), (1122, 393), (1124, 332), (1125, 256),
#                    (1118, 171), (1111, 115), (1079, 94), (1042, 91), (997, 138), (978, 208), (968, 265), (966, 351),
#                    (1025, 361), (1018, 425), (1016, 424)]

# # Convert polygon points to numpy array
# polygon_points = np.array(polygon_points)

# # Store the track history
# track_history = defaultdict(lambda: [])

# # Create a dictionary to keep track of objects that have crossed the line
# crossed_objects = {}
# # Configure the tracking parameters and run the tracker

# # if result is not None and len(result) > 0 and result[0].boxes is not None:
# #     print('done')
# #     boxes = result[0].boxes.xywh.cpu()
# #     track_ids = []
# #     for box in result[0].boxes:
# #         if box.id is not None:
# #             track_ids.append(box.id.item())
# #     # Iterate over the detected boxes
# #     for box, track_id in zip(boxes, track_ids):
# #         x, y, w, h = box
# #         track = track_history[track_id]
# #         track.append((float(x), float(y)))  # x, y center point
# #         if len(track) > 30:  # retain 30 tracks for 30 frames
# #             track.pop(0)
# # print('done1')
# result = None

# import threading
 
 
# def print_cube():
#     model = YOLO('yolov8n.pt')

#     result = model.track(source='rtsp://192.168.19.33:8080/h264_ulaw.sdp', conf=0.3, iou=0.5,verbose=False,stream=True)
 
 
# def print_square():
#     cv2.waitKey(1000000)
#     print('haha')
#     # while True:
#     #     global result
#     #     if result is not None and len(result) > 0 and result[0].boxes is not None:
#     #         print('done')
#     #         boxes = result[0].boxes.xywh.cpu()
#     #         track_ids = []
#     #         for box in result[0].boxes:
#     #             if box.id is not None:
#     #                 track_ids.append(box.id.item())
#     #         # Iterate over the detected boxes
#     #         for box, track_id in zip(boxes, track_ids):
#     #             x, y, w, h = box
#     #             track = track_history[track_id]
#     #             track.append((float(x), float(y)))  # x, y center point
#     #             if len(track) > 30:  # retain 30 tracks for 30 frames
#     #                 track.pop(0)
#     #     else:
#     #         print('done1')
#     cv2.waitKey(50000)
 
 
# if __name__ =="__main__":
#     t1 = threading.Thread(target=print_cube)
#     # t2 = threading.Thread(target=print_square)
 
#     t1.start()
#     # t2.start()
#     t1.join()
#     # t2.join()
 
#     print("Done!")