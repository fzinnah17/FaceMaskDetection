from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from scipy.special import softmax


BASE_DIR = Path(__file__).resolve().parent

FACE_DETECTOR_CONFIG = BASE_DIR / "models" / "deploy.prototxt.txt"
FACE_DETECTOR_WEIGHTS = (
    BASE_DIR / "models" / "res10_300x300_ssd_iter_140000_fp16.caffemodel"
)
MASK_MODEL_PATH = BASE_DIR / "face_cnn_model"


# OpenCV face detector.
face_detection_model = cv2.dnn.readNetFromCaffe(
    str(FACE_DETECTOR_CONFIG),
    str(FACE_DETECTOR_WEIGHTS),
)

# TensorFlow/Keras mask classifier.
model = tf.keras.models.load_model(str(MASK_MODEL_PATH))


LABELS = [
    "Mask",
    "No Mask",
    "Covered Mouth Chin",
    "Covered Nose Mouth",
]


def get_color(label):
    """Return the bounding-box color for a prediction label."""

    if label == "Mask":
        return 0, 255, 0

    if label == "No Mask":
        return 0, 0, 255

    if label == "Covered Mouth Chin":
        return 0, 255, 255

    return 255, 255, 0


def face_mask_prediction(img):
    """
    Detect faces in a frame and classify mask status.

    The pipeline:
      1. Detect faces with OpenCV DNN.
      2. Crop each detected face.
      3. Normalize and reshape the crop.
      4. Run the TensorFlow/Keras classifier.
      5. Draw the predicted class and confidence.
    """

    if img is None:
        raise ValueError("Input frame cannot be None.")

    image = img.copy()
    height, width = image.shape[:2]

    blob = cv2.dnn.blobFromImage(
        image,
        1,
        (300, 300),
        (104, 117, 123),
        swapRB=True,
    )

    face_detection_model.setInput(blob)
    detections = face_detection_model.forward()

    detected_faces = 0

    for index in range(detections.shape[2]):
        confidence = detections[0, 0, index, 2]

        if confidence <= 0.5:
            continue

        box = detections[0, 0, index, 3:7] * np.array(
            [width, height, width, height]
        )

        start_x, start_y, end_x, end_y = box.astype(int)

        start_x = max(0, start_x)
        start_y = max(0, start_y)
        end_x = min(width, end_x)
        end_y = min(height, end_y)

        if end_x <= start_x or end_y <= start_y:
            continue

        face = image[start_y:end_y, start_x:end_x]

        if face.size == 0:
            continue

        face_blob = cv2.dnn.blobFromImage(
            face,
            1,
            (100, 100),
            (104, 117, 123),
            swapRB=True,
        )

        face_blob_squeeze = np.squeeze(face_blob).T
        face_blob_rotate = cv2.rotate(
            face_blob_squeeze,
            cv2.ROTATE_90_CLOCKWISE,
        )
        face_blob_flip = cv2.flip(face_blob_rotate, 1)

        max_value = face_blob_flip.max()

        if max_value == 0:
            continue

        normalized_face = np.maximum(face_blob_flip, 0) / max_value
        model_input = normalized_face.reshape(1, 100, 100, 3)

        prediction = model.predict(model_input, verbose=0)
        probabilities = softmax(prediction, axis=-1)[0]

        predicted_index = int(probabilities.argmax())
        confidence_score = float(probabilities[predicted_index])
        label = LABELS[predicted_index]

        label_text = f"{label}: {confidence_score * 100:.0f}%"
        color = get_color(label)

        cv2.rectangle(
            image,
            (start_x, start_y),
            (end_x, end_y),
            color,
            2,
        )

        text_y = max(start_y - 10, 20)

        cv2.putText(
            image,
            label_text,
            (start_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

        detected_faces += 1

    cv2.putText(
        image,
        f"Faces detected: {detected_faces}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    return image
