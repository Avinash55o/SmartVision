from fastapi import FastAPI,File,UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import clip
import openvino.runtime as ov
import io
app = FastAPI()

device="cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess=clip.load("ViT-B/32",device=device)

@app.post("/generate-caption/")
async def generate_caption(file:uploadFile =File(...)):
    try:
        image=Image.open(io.BytesIO(await file.read()))
        image_input=preprocess(image).unsqueeze(0).to(device)


        with torch.no_grand():
            image_features= clip_model.encode_image(image_input).float()