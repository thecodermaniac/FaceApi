import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from .components.serve_model import predict, read_imagefile, compare_two_faces
import os

load_dotenv(".env")
app = FastAPI()


@app.get("/")
async def index():
    return {"message": "Hello World"}


@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    prediction = predict(image)

    return prediction

@app.post("/compare")
async def compare_api(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    extension1 = file1.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    extension2 = file2.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension1 or not extension2:
        return "Image must be jpg or png format!"
    image1 = read_imagefile(await file1.read())
    image2 = read_imagefile(await file2.read())
    result = compare_two_faces(image1, image2)

    return result


if __name__ == "__main__":
    print(os.getenv("PORT"))
    uvicorn.run("app:app", port=int(os.getenv("PORT")), reload=True)
