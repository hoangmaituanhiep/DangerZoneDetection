import cv2
import numpy as np

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

def draw_result_panel(panel, labels, cached_result, is_timed_out, remaining_timeout, timeout_duration=5.0):
    """
    Renders the model results and cooldown status in the 30% right placeholder panel.
    Scales dynamically to fit any panel dimensions.
    """
    h, w = panel.shape[:2]
    panel[:] = (26, 26, 30)  

    scale = max(0.5, min(2.5, min(h / 480.0, w / 274.0)))
    pad = int(14 * scale)
    thick_bold = max(1, int(round(1.8 * scale)))
    thick_norm = max(1, int(round(scale)))

    cv2.putText(panel, 'MODEL RESULT', (pad, int(30 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.62 * scale, (255, 255, 255), thick_bold, cv2.LINE_AA)
    cv2.putText(panel, 'Result Placeholder (30%)', (pad, int(48 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.38 * scale, (150, 150, 150), thick_norm, cv2.LINE_AA)
    cv2.line(panel, (pad, int(56 * scale)), (w - pad, int(56 * scale)), (65, 65, 75), thick_norm)

    if is_timed_out:
        status_text = f'TIMEOUT: {remaining_timeout:.1f}s'
        status_color = (0, 180, 255)  
        sub_status = 'Holding result...'
    else:
        status_text = 'STATUS: READY'
        status_color = (0, 255, 100)  
        sub_status = 'Scanning face...'

    cv2.putText(panel, status_text, (pad, int(76 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.46 * scale, status_color, thick_norm, cv2.LINE_AA)
    sub_w = int(110 * scale)
    if w - pad - sub_w > pad + int(100 * scale):
        cv2.putText(panel, sub_status, (w - pad - sub_w, int(76 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.35 * scale, (150, 150, 150), thick_norm, cv2.LINE_AA)

    bar_top = int(84 * scale)
    bar_bot = int(93 * scale)
    cv2.rectangle(panel, (pad, bar_top), (w - pad, bar_bot), (48, 48, 55), -1)
    if is_timed_out and timeout_duration > 0:
        ratio = max(0.0, min(1.0, remaining_timeout / timeout_duration))
        fill_w = int((w - 2 * pad) * ratio)
        cv2.rectangle(panel, (pad, bar_top), (pad + fill_w, bar_bot), status_color, -1)
    elif not is_timed_out:
        cv2.rectangle(panel, (pad, bar_top), (w - pad, bar_bot), (0, 180, 80), -1)

    card_top = int(104 * scale)
    card_bot = int(172 * scale)
    cv2.rectangle(panel, (pad, card_top), (w - pad, card_bot), (38, 38, 46), -1)
    cv2.rectangle(panel, (pad, card_top), (w - pad, card_bot), (65, 65, 80), thick_norm)

    if cached_result is not None:
        cv2.putText(panel, 'CURRENT PREDICTION:', (pad + int(10 * scale), int(122 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.38 * scale, (160, 160, 160), thick_norm, cv2.LINE_AA)
        label_text = cached_result['label'].upper()
        conf_val = cached_result['conf']
        cv2.putText(panel, label_text, (pad + int(10 * scale), int(155 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.72 * scale, (0, 255, 120), thick_bold, cv2.LINE_AA)
        cv2.putText(panel, f'{conf_val:.1%}', (w - pad - int(65 * scale), int(155 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.62 * scale, (0, 255, 120), thick_bold, cv2.LINE_AA)
    else:
        cv2.putText(panel, 'NO PREDICTION YET', (pad + int(10 * scale), int(135 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale, (140, 140, 140), thick_norm, cv2.LINE_AA)
        cv2.putText(panel, 'Awaiting face detection', (pad + int(10 * scale), int(155 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.36 * scale, (100, 100, 100), thick_norm, cv2.LINE_AA)

    cv2.putText(panel, 'Confidence Breakdown:', (pad, int(194 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.42 * scale, (200, 200, 200), thick_norm, cv2.LINE_AA)

    y0 = int(216 * scale)
    row_h = int(22 * scale)
    top_k = cached_result.get('top_idx', -1) if cached_result else -1
    probs = cached_result['probs'] if cached_result else [0.0] * len(labels)

    name_w = int(72 * scale)
    pct_w = int(40 * scale)
    bar_start_x = pad + name_w + int(4 * scale)
    bar_max_w = max(10, (w - pad) - bar_start_x - pct_w - int(6 * scale))

    for i, (name, p) in enumerate(zip(labels, probs)):
        y = y0 + i * row_h
        is_top = (i == top_k)
        text_color = (0, 255, 120) if is_top else (180, 180, 180)
        bar_color = (0, 200, 100) if is_top else (100, 95, 85)

        cv2.putText(panel, f'{name[:8]:<8s}', (pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38 * scale, text_color, thick_norm, cv2.LINE_AA)
        cv2.rectangle(panel, (bar_start_x, y - int(10 * scale)), (bar_start_x + bar_max_w, y), (42, 42, 48), -1)
        fill_w = int(p * bar_max_w)
        if fill_w > 0:
            cv2.rectangle(panel, (bar_start_x, y - int(10 * scale)), (bar_start_x + fill_w, y), bar_color, -1)
        cv2.putText(panel, f'{p:4.0%}', (bar_start_x + bar_max_w + int(6 * scale), y), cv2.FONT_HERSHEY_SIMPLEX, 0.35 * scale, text_color, thick_norm, cv2.LINE_AA)

    cv2.putText(panel, f'Timeout interval: {timeout_duration:.1f}s', (pad, int(425 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.36 * scale, (120, 120, 120), thick_norm, cv2.LINE_AA)
    cv2.putText(panel, 'Press [q] to exit', (pad, int(445 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.36 * scale, (90, 90, 90), thick_norm, cv2.LINE_AA)
