from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import clip
import io
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer
from functools import lru_cache
import asyncio

app = FastAPI()

# Set device (CPU or CUDA) with lazy initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cache model loading to avoid redundant initialization
@lru_cache(maxsize=1)
def load_clip_model():
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)  # Disable JIT for compatibility
    return model, preprocess

@lru_cache(maxsize=1)
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    return processor, model

@lru_cache(maxsize=1)
def load_gemma_model():
    model_name = "google/gemma-2b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    return tokenizer, model

# Load models once at startup
clip_model, preprocess = load_clip_model()
blip_processor, blip_model = load_blip_model()
gemma_tokenizer, gemma_model = load_gemma_model()

# Medical-specific labels for CLIP
MEDICAL_LABELS = [
    "normal tissue", "abnormal tissue", "tumor", "fracture", "infection", "healthy organ"
]

# ✅ Endpoint 1: Generate and Enhance Captions
@app.post("/generate-caption/")
async def generate_caption(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Step 1: CLIP for Zero-Shot Classification (optimized for medical use)
        image_input = preprocess(image).unsqueeze(0).to(device)
        text_inputs = clip.tokenize(MEDICAL_LABELS).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            text_features = clip_model.encode_text(text_inputs)
            similarity = (image_features @ text_features.T).softmax(dim=-1)
            best_match_idx = similarity.argmax().item()
            classification = MEDICAL_LABELS[best_match_idx]

        # Step 2: Generate raw caption using BLIP (async processing)
        inputs = blip_processor(image, return_tensors="pt").to(device)
        output_ids = await asyncio.to_thread(blip_model.generate, **inputs)
        caption = blip_processor.decode(output_ids[0], skip_special_tokens=True)

        # Step 3: Enhance caption using Gemma
        prompt = f"Medical Image Classification: {classification}\nRaw Caption: {caption}\nEnhance for medical report:"
        gemma_inputs = gemma_tokenizer(prompt, return_tensors="pt").to(device)
        gemma_output = await asyncio.to_thread(gemma_model.generate, **gemma_inputs, max_new_tokens=50)
        enhanced_caption = gemma_tokenizer.decode(gemma_output[0], skip_special_tokens=True)

        return JSONResponse({"caption": enhanced_caption, "classification": classification})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# ✅ Endpoint 2: Ask Questions About an Image (VQA)
@app.post("/ask-image/")
async def ask_image(file: UploadFile = File(...), question: str = ""):
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")

    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Step 1: CLIP classification
        image_input = preprocess(image).unsqueeze(0).to(device)
        text_inputs = clip.tokenize(MEDICAL_LABELS).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            text_features = clip_model.encode_text(text_inputs)
            similarity = (image_features @ text_features.T).softmax(dim=-1)
            best_match_idx = similarity.argmax().item()
            classification = MEDICAL_LABELS[best_match_idx]

        # Step 2: Generate caption
        inputs = blip_processor(image, return_tensors="pt").to(device)
        output_ids = await asyncio.to_thread(blip_model.generate, **inputs)
        caption = blip_processor.decode(output_ids[0], skip_special_tokens=True)

        # Step 3: Answer question with Gemma
        prompt = f"Medical Classification: {classification}\nDescription: {caption}\nQuestion: {question}\nAnswer:"
        gemma_inputs = gemma_tokenizer(prompt, return_tensors="pt").to(device)
        gemma_output = await asyncio.to_thread(gemma_model.generate, **gemma_inputs, max_new_tokens=50)
        answer = gemma_tokenizer.decode(gemma_output[0], skip_special_tokens=True)

        return JSONResponse({"question": question, "answer": answer, "classification": classification})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# ✅ Endpoint 3: Simplify Medical Reports
@app.post("/simplify-report/")
async def simplify_report(report: str, audience: str = "both"):
    """
    Simplify a medical report for either 'patient', 'doctor', or 'both'.
    - 'patient': Simple, non-technical language.
    - 'doctor': Concise, technical summary.
    - 'both': Returns both versions.
    """
    if not report:
        raise HTTPException(status_code=400, detail="Report text is required")

    if audience not in ["patient", "doctor", "both"]:
        raise HTTPException(status_code=400, detail="Audience must be 'patient', 'doctor', or 'both'")

    try:
        result = {}

        # Generate patient-friendly version
        if audience in ["patient", "both"]:
            patient_prompt = (
                f"Simplify this medical report for a patient in clear, non-technical language:\n{report}"
            )
            patient_inputs = gemma_tokenizer(patient_prompt, return_tensors="pt").to(device)
            patient_output = await asyncio.to_thread(
                gemma_model.generate, **patient_inputs, max_new_tokens=100
            )
            result["patient_version"] = gemma_tokenizer.decode(patient_output[0], skip_special_tokens=True)

        # Generate doctor-friendly version
        if audience in ["doctor", "both"]:
            doctor_prompt = (
                f"Summarize this medical report for a doctor, keeping it concise and using technical terms:\n{report}"
            )
            doctor_inputs = gemma_tokenizer(doctor_prompt, return_tensors="pt").to(device)
            doctor_output = await asyncio.to_thread(
                gemma_model.generate, **doctor_inputs, max_new_tokens=100
            )
            result["doctor_version"] = gemma_tokenizer.decode(doctor_output[0], skip_special_tokens=True)

        return JSONResponse(result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing report: {str(e)}")
    if not report:
        raise HTTPException(status_code=400, detail="Report text is required")

    try:
        prompt = f"Simplify this medical report for a patient:\n{report}"
        gemma_inputs = gemma_tokenizer(prompt, return_tensors="pt").to(device)
        gemma_output = await asyncio.to_thread(gemma_model.generate, **gemma_inputs, max_new_tokens=100)
        simple_report = gemma_tokenizer.decode(gemma_output[0], skip_special_tokens=True)

        return JSONResponse({"simplified_report": simple_report})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error simplifying report: {str(e)}")

# ✅ Endpoint 4: Recommend Follow-up Actions
@app.post("/recommend-followup/")
async def recommend_followup(report: str):
    if not report:
        raise HTTPException(status_code=400, detail="Report text is required")

    try:
        prompt = f"Based on this medical report, suggest follow-up actions:\n{report}"
        gemma_inputs = gemma_tokenizer(prompt, return_tensors="pt").to(device)
        gemma_output = await asyncio.to_thread(gemma_model.generate, **gemma_inputs, max_new_tokens=100)
        recommendations = gemma_tokenizer.decode(gemma_output[0], skip_special_tokens=True)

        return JSONResponse({"recommendations": recommendations})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")
    
