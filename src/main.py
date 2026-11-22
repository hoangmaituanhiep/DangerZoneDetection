from preprocess import process, draw_bars
import cv2, time, json, onnxruntime as ort

MODEL_PATH = "models/emotion.onnx"
LABELS = json.load(open("models/labels.json"))
IMG_SIZE = 96
MARGIN = 0.1
EMA_ALPHA = 0.2

sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def main():
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cam.isOpened():
        raise RuntimeError("Camera Error")
    emma, prev = None, time.time()

    while True:
        ok, frame = cam.read()
        if not ok:
            break;
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

        if len(faces):
            x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
            m = int(MARGIN * w)
            x0, y0, x1, y1 = max(0, x-m), max(0, y-m), min(gray.shape[1], x+w+m), min(gray.shape[0], y+h+m)

            probs = sess.run(None, {input_name: process(gray[y0:y1, x0:x1])})[0][0]
            emma = probs if emma is None else EMA_ALPHA*probs + (1-EMA_ALPHA)*emma

            k = int(emma.argmax())
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(frame, f'{LABELS[k]} {emma[k]:.0%}', (x0, max(y0-8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            draw_bars(LABELS, frame, emma)
        else:
            emma = None

        now = time.time()
        cv2.putText(frame, f'FPS:{1/max((now-prev), 1e-6):.0f}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        prev = now
        cv2.imshow("Emotion Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()