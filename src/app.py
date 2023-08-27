import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from components.serve_model import predict, read_imagefile
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


if __name__ == "__main__":
    print(os.getenv("PORT"))
    uvicorn.run("app:app", port=int(os.getenv("PORT")), reload=True)
