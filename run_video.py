import time
from pathlib import Path
import requests

import PIL
import torch

from kolya_color import device, data_transforms_val, padding, best_model_wts
import cv2
import sys
from ultralytics import YOLO

backend_url = "http://localhost:5321/upload_frame/afros_test"

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
    frame_skip = 2
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = str(int(fps))

        # Encode the frame as JPEG
        _, encoded_image = cv2.imencode('.jpg', frame)

        # Convert the encoded image to bytes
        frame_bytes = encoded_image.tobytes()

        # Prepare the multipart form data
        files = {
            'frame': ('frame.jpg', frame_bytes, 'image/jpeg')
        }

        # handled_balls = response.json()
        calculate = frame_count % frame_skip == 0

        if calculate:
            # Send the frame to the FastAPI backend
            response = requests.post(backend_url, files=files)

            # Check response
            if response.status_code != 200:
                print(f"Failed to send frame. Status code: {response.status_code}")
                continue
            handled_balls = response.json()
            for box in handled_balls:
                x1, y1, x2, y2 = [box["x1"], box["y1"], box["x2"], box["y2"]]
                color = box["color"]
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
