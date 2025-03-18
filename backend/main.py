# main.py

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Zero-Shot Image Classifier Backend is running!"}
