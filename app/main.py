from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
from .inference import predict

app = FastAPI()

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    result = predict(image)
    return result