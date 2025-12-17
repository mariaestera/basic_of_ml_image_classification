import os
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np

def resize(img, new_height=48):

    width, height = img.size

    scale = new_height / height
    new_width = int(width * scale)

    return img.resize((new_width, new_height), Image.BILINEAR)

def split_to_digits(img):
    # to grey scale
    img_grey = np.array(img.convert('L'))
    # to black-white
    _, binary = cv2.threshold(img_grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_inv = 255 - binary

    #looking for split points
    projection = np.sum(binary_inv, axis=0)

    segments = []
    in_char = False
    start = 0

    for i, val in enumerate(projection):
        if val > 0 and not in_char:
            start = i
            in_char = True
        elif val == 0 and in_char:
            segments.append((start, i))
            in_char = False

    if in_char:
        segments.append((start, len(projection)))

    digits = []

    for x1, x2 in segments:
        digit = binary_inv[:, x1:x2]

        rows = np.where(np.sum(digit, axis=1) > 0)[0]

        if len(rows) == 0:
            continue

        y1, y2 = rows[0], rows[-1]
        #digit = img_grey[y1:y2 + 1, x1:x2]
        digit = 255-digit[y1:y2 + 1, :]

        digits.append(digit)

    return digits


def prepare_digits_for_mnist(digits, blur_ksize=3, margin=3,n=20):
    prepared = []

    # zabezpieczenie kernela blura
    if blur_ksize % 2 == 0:
        blur_ksize += 1

    for digit in digits:
        # odwrócenie kolorów
        digit = 255 - digit

        # dodanie marginesów PRZED blurem
        digit = cv2.copyMakeBorder(
            digit,
            top=margin,
            bottom=margin,
            left=margin,
            right=margin,
            borderType=cv2.BORDER_CONSTANT,
            value=0
        )

        # blur
        digit = cv2.GaussianBlur(digit, (blur_ksize, blur_ksize), 0)

        h, w = digit.shape

        # skalowanie do 20x20 (jak MNIST)
        if h > w:
            new_h = n
            new_w = int(w * (new_h  / h))
        else:
            new_w = n
            new_h = int(h * (new_w / w))

        digit_resized = cv2.resize(
            digit, (new_w, new_h), interpolation=cv2.INTER_AREA
        )

        # padding do 28x28
        digit_padded = np.zeros((28, 28), dtype=np.uint8)
        top = (28 - new_h) // 2
        left = (28 - new_w) // 2
        digit_padded[top:top + new_h, left:left + new_w] = digit_resized

        # normalizacja
        digit_normalized = digit_padded.astype(np.float32) / 255.0
        prepared.append(digit_normalized)

    return np.array(prepared)[..., np.newaxis]