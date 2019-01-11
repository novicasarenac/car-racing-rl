import PIL
import numpy as np


def to_grayscale(img):
    return np.dot(img, [0.299, 0.587, 0.144])


def zero_center(img):
    return img - 127.0


def crop(img, bottom=12, left=6, right=6):
    height, width = img.shape
    return img[0: height - bottom, left: width - right]


def save(img, path):
    pil_img = PIL.Image.fromarray(img)
    pil_img.save(path)
