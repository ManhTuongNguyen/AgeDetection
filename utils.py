import cv2
from fastapi import UploadFile
import numpy as np
from age_detection import resize_image


async def handle_image_from_uploadfile(image: UploadFile) -> np.ndarray:
    byte_arr = await image.read()
    np_arr = np.frombuffer(byte_arr, np.uint8)
    # Decode the numpy array into an image
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    img = resize_image(img)
    assert img is not None, ValueError({"message": "Expected image"})
    return img
