import cv2
import uvicorn
from fastapi import FastAPI, UploadFile, Request, APIRouter
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.params import File
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from age_detection import detect_age
from config import BASE_URL, SERVER_HOST, SERVER_PORT
from utils import handle_image_from_upload_file

app = FastAPI()
router = APIRouter(prefix="/api")

templates = Jinja2Templates(directory="templates")
app.mount("/images", StaticFiles(directory="images"), name="images")

IMAGE_DIR = "images/"


@router.post("/detect-age/")
async def detect_age_api(image: UploadFile = File(...)):
    """
    Detect age from uploaded image
    """
    handled_image = await handle_image_from_upload_file(image)
    image_detected, labels = detect_age(handled_image)
    if image_detected is not None:
        cv2.imwrite(f"{IMAGE_DIR}{image.filename}", image_detected)
        return {
            "labels": labels,
            "code": "success",
            "image": BASE_URL + f'{IMAGE_DIR}{image.filename}',
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
    handled_image = await handle_image_from_upload_file(image)
    image_detected, labels = detect_age(handled_image)
    if image_detected is not None:
        cv2.imwrite(f"{IMAGE_DIR}{image.filename}", image_detected)
    else:
        cv2.imwrite(f"{IMAGE_DIR}{image.filename}", handled_image)
    return templates.TemplateResponse(
        "detect_age.html", 
        {
            "request": request,
            "image": image.filename,
            "labels": labels,
            "is_show_result": True
        }
    )


@app.get("/docs")
def read_docs():
    return get_swagger_ui_html(openapi_url="/openapi.json", title="API doc")


app.include_router(router)


if __name__ == "__main__":
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
