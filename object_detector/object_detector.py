import cv2
import object_detector_functions as odf


# load detector model MobileNet V2
model = odf.load_model("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

# video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # image pre-processing
    # img = odf.image_processing(frame)

    # classification
    # result = odf.image_classification(img, model, labels)

    # object detector
    result = odf.object_detector(model, frame)

    # display result on screen
    odf.display_result(frame, result)

    # 'q' for exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()