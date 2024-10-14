from pathlib import Path
import requests

import PIL
import torch

from kolya_color import device, data_transforms_val, padding, best_model_wts
import cv2
import sys
from ultralytics import YOLO
import time

backend_url = "http://localhost:5000/track"

detection = YOLO('yolov8-n-31.05.24.pt')

map_dict = {
    0: 'yellow',
    1: 'blue_str',
    2: 'red_str',
    3: 'pink_str',
    4: 'orange_str',
    5: 'green_str',
    6: 'brown_str',
    7: 'blue',
    8: 'red',
    9: 'pink',
    10: 'orange',
    11: 'green',
    12: 'brown',
    13: 'black',
    14: 'yellow_str',
    15: 'cue_ball',
 }

color_store = {}


def predict_color(img, x1, y1, x2, y2):
    crop = img[y1:y2, x1:x2].copy()
    # crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop = padding(image=crop)['image']
    crop = PIL.Image.fromarray(crop)
    crop = data_transforms_val(crop).unsqueeze(0).to(device)

    with torch.no_grad():
        y_pred = best_model_wts(crop)
        predicted = torch.argmax(y_pred, axis=1)
        color = map_dict[predicted.item()]
    return color


def color_identification(video_path):
    cap = cv2.VideoCapture(f"{video_path}")
    if not cap.isOpened():
        print("Error reading video file")
        sys.exit()

    prev_frame_time = 0
    new_frame_time = 0
    frame_count = 0
    prevFrame = None
    max_frame_count = 10000
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        if frame_count >= max_frame_count:
            break
        new_frame_time = time.time() 
        fps = 1/(new_frame_time - prev_frame_time) 
        prev_frame_time = new_frame_time
        fps = str(int(fps))
        frame_skip = 8

        results = detection(frame, classes=[0], conf=0.6)

        if len(results) == 0:
            continue

        result = results[0]
        boxes = result.boxes
        names = result.names
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        box_number = 1

        calculate = frame_count % frame_skip == 0

        detected_balls = []
        for box, cls, conf in zip(boxes.data, boxes.cls, boxes.conf):
            if names[int(cls)] != 'ball':
                continue
            x1, y1, x2, y2 = box[:4].int().tolist()
            ball_data = {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            }
            detected_balls.append(ball_data)

        payload = {
            "frame": int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
            "balls": detected_balls
        }
        response = requests.post(backend_url, json=payload)
        tracked_data = response.json()
        tracked_balls = tracked_data["tracked_balls"]

        if calculate:
            for box in tracked_balls:
                ball_id = box["id"]
                x1, y1, x2, y2 = [box["x1"], box["y1"], box["x2"], box["y2"]]
                start = time.perf_counter()
                stored = ball_id in color_store
                color = color_store.get(ball_id, '?')
                if not stored:
                    color = predict_color(img, x1, y1, x2, y2)
                    if color.endswith("_str") and ball_id is not None:
                        color_store[ball_id] = color

                    box_number += 1
                end = time.perf_counter()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{color}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)

        if not calculate:
            frame = prevFrame
        else:
            prevFrame = frame
        cv2.imshow("YOLOv8 Detection", frame)
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    video = Path("videos", "1.MOV")
    color_identification(str(video))
