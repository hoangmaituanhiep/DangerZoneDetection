import os, sys
try:
    from preprocess import process, draw_bars, draw_result_panel, draw_confidence_overlay
except ImportError:
    from src.preprocess import process, draw_bars, draw_result_panel, draw_confidence_overlay
import cv2, time, json, numpy as np, onnxruntime as ort

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "emotion.onnx") if not os.path.exists("models/emotion.onnx") else "models/emotion.onnx"
LABELS_PATH = os.path.join(BASE_DIR, "models", "labels.json") if not os.path.exists("models/labels.json") else "models/labels.json"
LABELS = json.load(open(LABELS_PATH))
IMG_SIZE = 96
MARGIN = 0.1
TIMEOUT_DURATION = 5.0  # 5 seconds cooldown after model predicts
WINDOW_NAME = "Emotion Recognition"

sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def main():
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cam.isOpened():
        cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise RuntimeError("Camera Error: Unable to open webcam.")

    # Enable scalable / resizable window
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    last_predict_time = None
    cached_result = None
    prev = time.time()
    initialized_window_size = False

    while True:
        ok, frame = cam.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Set initial window size on first frame
        if not initialized_window_size:
            default_total_w = int(round(w / 0.70))
            cv2.resizeWindow(WINDOW_NAME, default_total_w, h)
            initialized_window_size = True

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

        now = time.time()
        time_since_pred = (now - last_predict_time) if last_predict_time is not None else float('inf')
        is_timed_out = time_since_pred < TIMEOUT_DURATION
        remaining_timeout = max(0.0, TIMEOUT_DURATION - time_since_pred) if is_timed_out else 0.0

        # Query live window dimensions for dynamic 70/30 scaling
        rect = cv2.getWindowImageRect(WINDOW_NAME)
        if rect[2] > 100 and rect[3] > 100:
            win_w, win_h = rect[2], rect[3]
        else:
            win_w = int(round(w / 0.70))
            win_h = h

        left_w = int(round(win_w * 0.70))
        right_w = max(50, win_w - left_w)

        if len(faces):
            x, y, fw, fh = max(faces, key=lambda f: f[2]*f[3])
            m = int(MARGIN * fw)
            x0, y0, x1, y1 = max(0, x-m), max(0, y-m), min(w, x+fw+m), min(h, y+fh+m)

            # Predict only when timeout is over
            if not is_timed_out:
                face_crop = gray[y0:y1, x0:x1]
                if face_crop.size > 0:
                    probs = sess.run(None, {input_name: process(face_crop)})[0][0]
                    k = int(probs.argmax())
                    cached_result = {
                        'label': LABELS[k],
                        'conf': float(probs[k]),
                        'probs': probs.copy(),
                        'top_idx': k
                    }
                    last_predict_time = now
                    is_timed_out = True
                    remaining_timeout = TIMEOUT_DURATION

            # Draw clean face detection box
            box_color = (0, 180, 255) if is_timed_out else (0, 255, 100)
            cv2.rectangle(frame, (x0, y0), (x1, y1), box_color, 2)
            if cached_result is not None:
                tag = f"{cached_result['label'].capitalize()} {cached_result['conf']:.0%}"
                if is_timed_out:
                    tag += f" [{remaining_timeout:.1f}s]"
                cv2.putText(frame, tag, (x0, max(y0-8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, box_color, 2, cv2.LINE_AA)

        if (frame.shape[1], frame.shape[0]) != (left_w, win_h):
            webcam_display = cv2.resize(frame, (left_w, win_h), interpolation=cv2.INTER_LINEAR)
        else:
            webcam_display = frame

        top_idx = cached_result.get('top_idx') if cached_result else None
        probs = cached_result.get('probs') if cached_result else None
        draw_confidence_overlay(webcam_display, LABELS, probs, top_idx=top_idx, alpha=0.3)

        fps = 1.0 / max(now - prev, 1e-6)
        prev = now
        cv2.putText(webcam_display, f'{fps:.0f} FPS', (left_w - 75, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

        panel = np.zeros((win_h, right_w, 3), dtype=np.uint8)
        draw_result_panel(panel, cached_result, is_timed_out, remaining_timeout, TIMEOUT_DURATION)

        canvas = np.hstack([webcam_display, panel])
        divider_thick = max(1, int(round(win_h / 480.0)))
        cv2.line(canvas, (left_w, 0), (left_w, win_h), (55, 55, 65), divider_thick)

        cv2.imshow(WINDOW_NAME, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
