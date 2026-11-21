import cv2
import numpy as np
import tensorflow as tf
from collections import deque

MODEL_PATH = "../models"
IMG_SIZE = (96, 96)
EMOTIONS = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]

model = tf.keras.models.load_model(MODEL_PATH)

facecascade = cv2.CascadeClassifier(cv2.data.haarcascade + "haarcascade_frontalface_default.xml")

