import cv2, numpy as np
def __init__(self, LABELS):
    self.LABELS = LABELS

def process(grayface):
    g48 = cv2.resize(grayface, (48, 48), interpolation=cv2.INTER_AREA)
    g96 = cv2.resize(g48, (96, 96), interpolation=cv2.INTER_CUBIC)

    x = np.stack([g96]*3, axis=-1).astype(np.unit8)

    return x[None]

def draw_bars(LABELS, frame, probs, x0=10, y0=60):
    for i, (name, p) in enumerate(zip(LABELS, probs)):
        y = y0 + i*22
        cv2.rectangle(frame, (x0+90, y-12), (x0+90+int(p*150), y), (0, 200, 0), -1)
        cv2.putText(frame, f'{name[:9]:9s} {p:4.0%}', (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)