from datetime import datetime
from pathlib import Path

from identify import identify_ball_color
from shakh_color import ball_color
import cv2
import sys
from ultralytics import YOLO
import time


detection = YOLO('yolov8-n-31.05.24.pt')


def color_identification(video_path):
    cap = cv2.VideoCapture(f"{video_path}")
    if not cap.isOpened():
        print("Error reading video file")
        sys.exit()

    prev_frame_time = 0
    new_frame_time = 0
    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        new_frame_time = time.time() 
        fps = 1/(new_frame_time - prev_frame_time) 
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps = str(fps)
        frame_skip = 1
        if success:
            results = detection(frame, classes=[0], conf=0.6)
            #cv2.imwrite('frame.jpeg', frame)
            for i in range(len(results)):
                # if frame_count % frame_skip != 0:
                #     continue  # Skip detection

                result = results[i]
                boxes = result.boxes
                names = result.names
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                box_number = 1
                for box, cls, conf in zip(boxes.data, boxes.cls, boxes.conf):
                    x1, y1, x2, y2 = box[:4].int().tolist()
                    start = time.perf_counter()
                    if names[int(cls)] == 'ball':
                        crop = img[y1:y2, x1:x2].copy()
                        color, _, _, _ = identify_ball_color(image_crop=crop)
                        # color = ball_color.get_main_color(crop)
                        # crop2save = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        # cv2.imwrite(f'crops/{color}/crop_{frame_count}_{box_number}.jpeg', crop2save)
                        box_number += 1
                    # elif names[int(cls)] == 'glove':
                    #     color = glove_color.get_main_color(crop)
                    else:
                        color = 'no color for this shit'
                    end = time.perf_counter()
                    # if color == 'white':
                    #     centers = get_dots(crop)
                    #     print(f'===================COLOR: {centers}====================')
                    #     for x, y in centers:
                    #         cv2.circle(frame, (x2 - x, y2 - y), radius=2, color=(255, 0, 0), thickness= -1)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"{color}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)        
            cv2.imshow("YOLOv8 Detection", frame)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    video = Path("videos", "1.MOV")
    color_identification(str(video))
