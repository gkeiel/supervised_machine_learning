import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


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


def load_model(model_url):
    # load model MobileNet V2
    model = hub.load(model_url)
    return model


def object_detector(model, frame):
    # detect object in frame
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = tf.convert_to_tensor(img, dtype=tf.uint8)
    img = tf.expand_dims(img, 0)
    result = model(img)
    return {key:np.squeeze(value.numpy()) for key,value in result.items()}


def image_processing(frame):
    img = cv2.resize(frame, (224, 224))
    img = img/255.0                     # normalization
    img = np.expand_dims(img, axis=0)
    return img


def image_classification(img, model, labels):
    predictions = model(img)
    class_id = np.argmax(predictions)
    class_name = labels[class_id]
    return class_name


#def display_result(frame, class_name):
#    cv2.putText(frame, class_name, (10, 30),
#                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#    cv2.imshow('Object Classifier', frame)


def display_result(frame, result, threshold=0.5):
    # draw bounding boxes and labels
    height, width, _ = frame.shape
    boxes = result['detection_boxes']
    classes = result['detection_classes'].astype(int)
    scores = result['detection_scores']

    for i in range(len(scores)):
        if scores[i] >= threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            x1, y1, x2, y2 = int(xmin*width), int(ymin*height), int(xmax*width), int(ymax*height)
            
            # desenhar bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

            # rótulo da classe
            class_id = classes[i]
            label = COCO_LABELS[class_id] if class_id < len(COCO_LABELS) else f"Class {class_id}"
            text = f"{label}: {scores[i]:.2f}"

            # caixa de fundo para o texto
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
            cv2.putText(frame, text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv2.imshow("TensorFlow Object Detection", frame)