import cv2
from cv2.typing import MatLike
from fastapi import UploadFile
import numpy as np


def resize_image(image: MatLike, _size=2000) -> MatLike:
    h, w = image.shape[:2]
    aspect_ratio = w / h

    if h > _size or w > _size:
        # If width is larger
        if aspect_ratio > 1:
            new_w = _size
            new_h = int(new_w / aspect_ratio)
        # If height is larger
        else:
            new_h = _size
            new_w = int(new_h * aspect_ratio)

        # Resize the image
        image = cv2.resize(image, (new_w, new_h))
    return image


async def handle_image_from_upload_file(image: UploadFile) -> np.ndarray:
    byte_arr = await image.read()
    np_arr = np.frombuffer(byte_arr, np.uint8)
    # Decode the numpy array into an image
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    img = resize_image(img)
    assert img is not None, ValueError({"message": "Expected image"})
    return img
