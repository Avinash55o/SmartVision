from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import clip
import openvino.runtime as ov
import io
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP for image-question matching
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Load BLIP for image captioning
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Load Gemma for enhancing captions
gemma_model_name = "google/gemma-2b-it"
gemma_tokenizer = AutoTokenizer.from_pretrained(gemma_model_name)
gemma_model = AutoModelForCausalLM.from_pretrained(gemma_model_name).to(device)

@app.post("/generate-caption/")
async def generate_caption(file: UploadFile = File(...)):
    try:
        # Open and preprocess the image
        image = Image.open(io.BytesIO(await file.read()))
        
        # Generate a caption using BLIP
        inputs = blip_processor(image, return_tensors="pt").to(device)
        output_ids = blip_model.generate(**inputs)
        caption = blip_processor.decode(output_ids[0], skip_special_tokens=True)

        # Enhance the caption using Gemma
        prompt = f"Enhance the caption: {caption}"
        gemma_inputs = gemma_tokenizer(prompt, return_tensors="pt").to(device)
        gemma_output = gemma_model.generate(**gemma_inputs, max_length=50, num_return_sequences=1)
        enhanced_caption = gemma_tokenizer.decode(gemma_output[0], skip_special_tokens=True)

        return JSONResponse({"caption": enhanced_caption})

    except Exception as e:
        return JSONResponse({"error": str(e)})

@app.post("/ask-question/")
async def ask_question(file: UploadFile = File(...), question: str = Form(...)):
    try:
        # Open and preprocess the image
        image = Image.open(io.BytesIO(await file.read()))
        image_input = preprocess(image).unsqueeze(0).to(device)

        # Tokenize the question
        question_token = clip.tokenize([question]).to(device)

        # Get CLIP embeddings
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            text_features = clip_model.encode_text(question_token)

        # Calculate similarity
        similarity = torch.cosine_similarity(image_features, text_features)
        similarity_score = similarity.item()

        # Check similarity threshold
        if similarity_score > 0.3:  # You can fine-tune this threshold
            answer = "Yes, the image matches the description."
        else:
            answer = "No, the image does not match the description."

        return JSONResponse({"question": question, "answer": answer, "similarity_score": similarity_score})

    except Exception as e:
        return JSONResponse({"error": str(e)})


@app.post("/classify-image/")
async def classify_image(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read()))
        image_input = preprocess(image).unsqueeze(0).to(device)

        # Generate classification using CLIP
        text_descriptions = ["a beautiful sunset", "a busy street", "a cat on a sofa"]
        text_tokens = clip.tokenize(text_descriptions).to(device)
        text_features = clip_model.encode_text(text_tokens).float()

        similarity = (image_input @ text_features.T).squeeze(0)
        best_match_idx = similarity.argmax().item()
        classification = text_descriptions[best_match_idx]

        return JSONResponse({"classification": classification})

    except Exception as e:
        return JSONResponse({"error": str(e)})
