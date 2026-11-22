import sys, json
import cv2, numpy as np, onnxruntime as ort

LABELS = json.load(open("models/labels.json"))
sess = ort.InferenceSession("models/emotion.onnx", providers=["CPUExecutionProvider"])
name = sess.get_inputs()[0].name

img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
g48 = cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA)
g96 = cv2.resize(g48, (96, 96), interpolation=cv2.INTER_CUBIC)
x = np.stack([g96]*3, axis=-1).astype(np.uint8)[None]

probs = sess.run(None, {name: x})[0][0]
for l, p in sorted(zip(LABELS, probs), key=lambda t:-t[1]):
    print(f'{l:10s} {p:1%}')