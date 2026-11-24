import os
import cv2

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

def _read_asset(base_name):
    for ext in ['.jpg', '.png', '.jpeg', '.webp']:
        path = os.path.join(ASSETS_DIR, f"{base_name}{ext}")
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                return img
        # Also try relative to current working directory
        rel_path = os.path.join("assets", f"{base_name}{ext}")
        if os.path.exists(rel_path):
            img = cv2.imread(rel_path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                return img
    return None

anger = _read_asset('anger')
contempt = _read_asset('contempt')
disgust = _read_asset('disgust')
fear = _read_asset('fear')
happiness = _read_asset('happiness')
happy = happiness
neutral = _read_asset('neutral')
sadness = _read_asset('sadness')
surprise = _read_asset('surprise')

_IMAGES = {
    'anger': anger,
    'contempt': contempt,
    'disgust': disgust,
    'fear': fear,
    'happiness': happiness,
    'happy': happiness,
    'neutral': neutral,
    'sadness': sadness,
    'surprise': surprise
}

def load_image(label):
    if not label:
        return None
    key = str(label).lower().strip()
    if key in _IMAGES and _IMAGES[key] is not None:
        return _IMAGES[key]
    # Fallback to direct read attempt
    img = _read_asset(key)
    if img is not None:
        _IMAGES[key] = img
        return img
    raise ValueError(f"Unknown label: {label}")
