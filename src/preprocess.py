import cv2
import numpy as np

try:
    from load_image import load_image
except ImportError:
    from src.load_image import load_image

def process(grayface):
    g48 = cv2.resize(grayface, (48, 48), interpolation=cv2.INTER_AREA)
    g96 = cv2.resize(g48, (96, 96), interpolation=cv2.INTER_CUBIC)

    x = np.stack([g96]*3, axis=-1).astype(np.uint8)

    return x[None]

def draw_bars(LABELS, frame, probs, x0=10, y0=60):
    for i, (name, p) in enumerate(zip(LABELS, probs)):
        y = y0 + i*22
        cv2.rectangle(frame, (x0+90, y-12), (x0+90+int(p*150), y), (0, 200, 0), -1)
        cv2.putText(frame, f'{name[:9]:9s} {p:4.0%}', (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

def draw_confidence_overlay(frame, labels, probs, top_idx=None, alpha=0.55):
    """
    Renders the confidence breakdown as a semi-transparent HUD overlay on the webcam feed.
    """
    h, w = frame.shape[:2]
    scale = max(0.65, min(1.6, h / 480.0))

    pad = int(12 * scale)
    row_h = int(19 * scale)
    box_w = int(220 * scale)
    box_h = int((30 + len(labels) * 19 + 8) * scale)

    x0, y0 = int(16 * scale), int(16 * scale)
    x1, y1 = x0 + box_w, y0 + box_h

    if x1 > w or y1 > h:
        return

    roi = frame[y0:y1, x0:x1]
    overlay_bg = np.full_like(roi, (18, 18, 22), dtype=np.uint8)
    blended = cv2.addWeighted(overlay_bg, alpha, roi, 1.0 - alpha, 0)

    cv2.putText(blended, 'confidence', (pad, int(20 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.40 * scale, (210, 210, 210), 1, cv2.LINE_AA)

    name_w = int(68 * scale)
    pct_w = int(36 * scale)
    bar_x = pad + name_w + int(4 * scale)
    bar_max_w = box_w - pad - bar_x - pct_w - int(4 * scale)

    start_y = int(38 * scale)
    prob_list = probs if probs is not None else [0.0] * len(labels)

    for i, (name, p) in enumerate(zip(labels, prob_list)):
        y = start_y + i * row_h
        is_top = (i == top_idx) if top_idx is not None else False
        text_color = (0, 255, 130) if is_top else (190, 190, 190)
        bar_color = (0, 210, 100) if is_top else (100, 95, 85)

        cv2.putText(blended, f'{name[:8]:<8s}', (pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35 * scale, text_color, 1, cv2.LINE_AA)
        cv2.rectangle(blended, (bar_x, y - int(8 * scale)), (bar_x + bar_max_w, y), (42, 42, 48), -1)
        fill_w = int(p * bar_max_w)
        if fill_w > 0:
            cv2.rectangle(blended, (bar_x, y - int(8 * scale)), (bar_x + fill_w, y), bar_color, -1)
        cv2.putText(blended, f'{p:3.0%}', (bar_x + bar_max_w + int(4 * scale), y), cv2.FONT_HERSHEY_SIMPLEX, 0.32 * scale, text_color, 1, cv2.LINE_AA)

    frame[y0:y1, x0:x1] = blended

def draw_result_panel(panel, cached_result, is_timed_out, remaining_timeout, timeout_duration=5.0):
    """
    Renders the primary emotion image and prediction caption in the right 30% placeholder.
    """
    h, w = panel.shape[:2]
    panel[:] = (22, 22, 26)  
    scale = max(0.6, min(2.2, min(h / 480.0, w / 274.0)))

    if cached_result is not None:
        label = cached_result['label']
        try:
            img = load_image(label)
        except Exception:
            img = None

        if img is not None:
            ih, iw = img.shape[:2]
            pad_x = int(w * 0.08)
            max_w = max(10, w - 2 * pad_x)
            max_h = max(10, int(h * 0.62))

            img_scale = min(max_w / iw, max_h / ih)
            nw = max(1, int(iw * img_scale))
            nh = max(1, int(ih * img_scale))
            resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

            ox = (w - nw) // 2
            oy = int(h * 0.08)

            if resized.ndim == 3 and resized.shape[2] == 4:
                alpha = resized[:, :, 3] / 255.0
                for c in range(3):
                    panel[oy:oy+nh, ox:ox+nw, c] = (
                        alpha * resized[:, :, c] + (1.0 - alpha) * panel[oy:oy+nh, ox:ox+nw, c]
                    ).astype(np.uint8)
            else:
                panel[oy:oy+nh, ox:ox+nw] = resized[:, :, :3]

            cv2.rectangle(panel, (ox, oy), (ox + nw, oy + nh), (55, 55, 65), 1)

            caption = f"{label.upper()}  {cached_result['conf']:.0%}"
            caption_scale = max(0.55, 0.72 * scale)
            thick = max(1, int(round(1.8 * scale)))
            text_y = oy + nh + int(32 * scale)

            (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, caption_scale, thick)
            cx = max(10, (w - tw) // 2)
            cv2.putText(panel, caption, (cx, text_y), cv2.FONT_HERSHEY_SIMPLEX, caption_scale, (0, 255, 120), thick, cv2.LINE_AA)
            
    else:
        # Minimal awaiting state
        placeholder_text = "Awaiting Face"
        font_scale = max(0.5, 0.6 * scale)
        (tw, th), _ = cv2.getTextSize(placeholder_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cx = max(10, (w - tw) // 2)
        cy = h // 2
        cv2.putText(panel, placeholder_text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (120, 120, 130), 1, cv2.LINE_AA)
