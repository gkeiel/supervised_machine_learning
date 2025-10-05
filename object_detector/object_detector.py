import os, cv2
import object_detector_functions as odf
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# load detector model MobileNet V2
model = odf.load_model("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

# video capture
cap = cv2.VideoCapture(0)

frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    if frame_id % 5 != 0:
        continue

    # resize frame
    frame = cv2.resize(frame, (224, 224))
    
    # object detector
    result = odf.object_detector(model, frame)

    # display result on screen
    height, object_counts = odf.display_result(frame, result)
    frame = odf.panel(frame, height, object_counts)
    cv2.imshow("TensorFlow Object Detection", frame)

    # pressing 'q' for exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()