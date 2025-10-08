import cv2, time, csv
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from queue import Queue
from threading import Thread
from datetime import datetime


# list of COCO classes
COCO_LABELS = [
    "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "street sign",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "hat", "backpack",
    "umbrella", "shoe", "eye glasses", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "plate", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "mirror", "dining table", "window", "desk",
    "toilet", "door", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "blender", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# time
time_last = time.time()


def load_model(model_url):
    # load MobileNet V2 model
    model = hub.load(model_url)
    return model


def object_detector(model, frame):
    # detect object in frame
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = tf.convert_to_tensor(img, dtype=tf.uint8)
    img = tf.expand_dims(img, 0)
    result = model(img)
    return {key:np.squeeze(value.numpy()) for key,value in result.items()}


def display_result(frame, result, threshold=0.5):
    # draw bounding boxes and labels
    height, width, _ = frame.shape
    boxes   = result['detection_boxes']
    classes = result['detection_classes'].astype(int)
    scores  = result['detection_scores']
    object_counts = {}

    for i in range(len(scores)):
        if scores[i] >= threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            x1, y1, x2, y2 = int(xmin*width), int(ymin*height), int(xmax*width), int(ymax*height)
            
            # draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

            # class label
            class_id = classes[i]
            label = COCO_LABELS[class_id] if class_id < len(COCO_LABELS) else f"Class {class_id}"
            text = f"{label}: {scores[i]:.2f}"

            confidence = scores[i]
            logging(label, confidence)
            
            # text box
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
            cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            object_counts[label] = object_counts.get(label, 0) + 1

        return height, object_counts


def calculate_fps(time_now, time_last):
    # calculate FPS
    return 1/(time_now -time_last)


def panel(frame, height, object_counts):
    global time_last
    scale = height/480

    panel_width = int(200*scale)
    panel = np.zeros((height, panel_width, 3), dtype = np.uint8)

    # get FPS
    time_now  = time.time()
    fps       = calculate_fps(time_now, time_last)
    time_last = time_now

    font_scale = 0.6 * scale
    thickness  = max(1, int(2 * scale))
    spacing    = int(30 * scale)

    # title
    cv2.putText(panel, "INFO PANEL", (int(10*scale), int(30*scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.8*scale, (0,255,0), thickness)

    # FPS
    cv2.putText(panel, f"FPS: {fps:.1f}", (int(10*scale), int(70*scale)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), max(1,int(scale)))

    # counting
    total_objects = sum(object_counts.values())
    cv2.putText(panel, f"Total objects: {total_objects}", (int(10*scale), int(100*scale)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), max(1, int(scale)))

    # lista por classe
    y = int(140*scale)
    for obj, count in object_counts.items():
        cv2.putText(panel, f"{obj}: {count}", (int(10*scale), y), cv2.FONT_HERSHEY_SIMPLEX, 0.5*scale, (200,200,200), max(1,int(scale)))
        y += spacing

    frame = np.hstack((frame, panel))
    return frame


def logging(label, confidence):
    current_time = datetime.now()
    
    with open('logs/detections.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([current_time, label, confidence])