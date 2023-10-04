from io import BytesIO

import numpy as np
import tensorflow as tf
import cv2
import face_recognition
from tensorflow import keras
from PIL import Image
from keras.applications.imagenet_utils import decode_predictions

model = None


def load_model():
    model = tf.keras.applications.MobileNetV2(weights="imagenet")
    print("Model loaded")
    return model


def predict(image: Image.Image):
    global model
    if model is None:
        model = load_model()

    image = np.asarray(image.resize((224, 224)))[..., :3]
    image = np.expand_dims(image, 0)
    image = image / 127.5 - 1.0

    result = decode_predictions(model.predict(image), 2)[0]

    response = []
    for i, res in enumerate(result):
        resp = {}
        resp["class"] = res[1]
        resp["confidence"] = f"{res[2]*100:0.2f} %"

        response.append(resp)

    return response


def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image

def compare_two_faces(image1: Image.Image, image2: Image.Image):
    image1 = np.asarray(image1.resize((224, 224)))[..., :3]
    image2 = np.asarray(image2.resize((224, 224)))[..., :3]

    face1_encodings = face_recognition.face_encodings(image1)
    face2_encodings = face_recognition.face_encodings(image2)

    if not face1_encodings or not face2_encodings:
        return "No face detected in one of the images"

    face1 = face1_encodings[0]
    face2 = face2_encodings[0]

    result = face_recognition.compare_faces([face1], face2)

    return result[0] and "Same person" or "Not same person"
