import sys
import cv2, numpy as np
import tensorflow as tf

EMOTIONS = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
model = tf.keras.models.load_model("models/best.keras", safe_mode=False, compile=False)
img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_CUBIC)
x = np.stack([img]*3, axis=-1)[None].astype("uint8")
p = model(x, training=False).numpy()[0]
print({e: round(float(v), 3) for e, v in zip(EMOTIONS, p)})
print("Predict:", EMOTIONS[int(p.argmax())])