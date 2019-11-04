import numpy as np
import random
import cv2

def random_mask(height, width, channel=3):
    img = np.zeros((height, width, channel), dtype=np.uint8)

    size = int((width + height) * 0.03)
    if width < 64 or height < 64:
        raise Exception('width and height of mask must be at least 64!')\

    # Draw random lines
    for _ in range(random.randint(1, 20)):
        x1, x2 = random.randint(1, width), random.randint(1, width)
        y1, y2 = random.randint(1, height), random.randint(1, width)
        thickness = random.randint(3, size)
        cv2.line(img, (x1, y1), (x2, y2), (1, 1, 1), thickness)

    # Draw random circle
    for _ in range(random.randint(1, 20)):
        x1, y1 = random.randint(1, width), random.randint(1, height)
        radius = random.randint(3, size)
        cv2.circle(img, (x1, y1), radius, (1, 1, 1), -1)

    # Draw random ellipses
    for _ in range(random.randint(1, 20)):
        x1, y1 = random.randint(1, width), random.randint(1, height)
        s1, s2 = random.randint(1, width), random.randint(1, height)
        a1, a2, a3 = random.randint(3, 180), random.randint(3, 180), random.randint(3, 180)
        thickness = random.randint(3, size)
        cv2.ellipse(img, (x1, y1), (s1, s2), a1, a2, a3, (1, 1, 1), thickness)

    return 1-img
