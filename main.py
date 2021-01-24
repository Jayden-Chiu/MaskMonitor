from flask import Flask, render_template, Response
import numpy as np
import cv2
import imutils
import os
import io
from scipy.spatial import distance
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

app = Flask(__name__)

cap = cv2.VideoCapture(0)

models_dir = "models"
prototxt = os.path.sep.join([models_dir, "deploy.prototxt"])
weights = os.path.sep.join(
    [models_dir, "res10_300x300_ssd_iter_140000.caffemodel"])
mask_detector_path = os.path.sep.join([models_dir, "mask_detector.model"])

face_net = cv2.dnn.readNet(prototxt, weights)
mask_net = load_model(mask_detector_path)


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def gen():
    """Video streaming generator function."""
    while True:
        ret, frame = cap.read()
        label = None

        (frame, num, label) = detect_mask(frame)

        encode_return_code, image_buffer = cv2.imencode('.jpg', frame)
        io_buf = io.BytesIO(image_buffer)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + io_buf.read() + b'\r\n')


def detect_mask(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    face_net.setInput(blob)
    detections = face_net.forward()

    rois = []
    coords = []
    preds = []

    label = None

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (start_x, start_y, end_x, end_y) = box.astype("int")

            (start_x, start_y) = (max(0, start_x), max(0, start_y))
            (end_x, end_y) = (min(w - 1, end_x), min(h - 1, end_y))

            # get roi from frame
            face = frame[start_y:end_y, start_x:end_x]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            rois.append(face)
            coords.append((start_x, start_y, end_x, end_y))

    num = 0
    
    if len(rois) > 0:
        rois = np.array(rois, dtype="float32")
        preds = mask_net.predict(rois, batch_size=32)

        for (box, pred) in zip(coords, preds):
            (start_x, start_y, end_x, end_y) = box
            (incorrect_mask, mask, no_mask) = pred

            if mask > no_mask and mask > incorrect_mask:
                num += 1
                label = "Mask"
                color = (0, 255, 0)

            elif no_mask > mask and no_mask > incorrect_mask:
                label = "No Mask"
                color = (0, 0, 255)
            elif incorrect_mask > mask and incorrect_mask > no_mask:
                label = "Incorrect Mask"
                color = (255, 0, 0)

            cv2.rectangle(frame, (start_x, start_y - 25),
                          (end_x, start_y), color, -1)
            cv2.putText(frame, label, (start_x, start_y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, 2)

    return (frame, num, label)


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(
        gen(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=True)
