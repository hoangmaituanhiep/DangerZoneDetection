import cv2

anger = cv2.imread('assets/anger.png', cv2.IMREAD_UNCHANGED)
contempt = cv2.imread('assets/contempt.png', cv2.IMREAD_UNCHANGED)
disgust = cv2.imread('assets/disgust.png', cv2.IMREAD_UNCHANGED)
fear = cv2.imread('assets/fear.png', cv2.IMREAD_UNCHANGED)
happy = cv2.imread('assets/happy.png', cv2.IMREAD_UNCHANGED)
neutral = cv2.imread('assets/neutral.png', cv2.IMREAD_UNCHANGED)
sadness = cv2.imread('assets/sadness.png', cv2.IMREAD_UNCHANGED)
surprise = cv2.imread('assets/surprise.png', cv2.IMREAD_UNCHANGED)

def load_image(label):
    if label == 'anger':
        return anger
    elif label == 'contempt':
        return contempt
    elif label == 'disgust':
        return disgust
    elif label == 'fear':
        return fear
    elif label == 'happy':
        return happy
    elif label == 'neutral':
        return neutral
    elif label == 'sadness':
        return sadness
    elif label == 'surprise':
        return surprise
    else:
        raise ValueError(f"Unknown label: {label}")