import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, Request, APIRouter
from fastapi.params import File
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from age_detection import detect_age, resize_image
from config import BASE_URL, SERVER_HOST, SERVER_PORT

app = FastAPI()
router = APIRouter(prefix="/api")

templates = Jinja2Templates(directory="templates")
app.mount("/images", StaticFiles(directory="images"), name="images")

IMAGEDIR = "images/"


@router.post("/detect-age/")
async def detect_age_api(image: UploadFile = File(...)):
    """Detect age from uploaded image

    Returns:
        dict -- {"file": file_path, "labels": labels}
    """
    byte_arr = await image.read()
    np_arr = np.frombuffer(byte_arr, np.uint8)
    # Decode the numpy array into an image
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    img = resize_image(img)
    assert img is not None, ValueError({"message": "Expected image"})
    image_detected, labels = detect_age(img)
    if image_detected is not None:
        cv2.imwrite(f"{IMAGEDIR}{image.filename}", image_detected)
        return {
            "labels": labels,
            "code": "success",
            "image": BASE_URL + f'{IMAGEDIR}{image.filename}',
        }
    else:
        return {
            "message": "Không nhận diện được!",
            "code": "failed"
        }


@app.get('/', response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("detect_age.html", {"request": request, "is_show_result": False})


@app.post("/")
async def create_upload_files(request: Request, image: UploadFile = File(...)):
    byte_arr = await image.read()
    np_arr = np.frombuffer(byte_arr, np.uint8)
    # Decode the numpy array into an image
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    img = resize_image(img)
    assert img is not None, ValueError({"message": "Expected image"})
    image_detected, labels = detect_age(img)
    if image_detected is not None:
        cv2.imwrite(f"{IMAGEDIR}{image.filename}", image_detected)
    else:
        cv2.imwrite(f"{IMAGEDIR}{image.filename}", img)
    return templates.TemplateResponse(
        "detect_age.html", 
        {
            "request": request,
            "image": image.filename,
            "labels": labels,
            "is_show_result": True
        }
    )


app.include_router(router)


if __name__ == "__main__":
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
