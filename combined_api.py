import os
import torch
import clip
import easyocr
from PIL import Image
from transformers import pipeline
from scipy.stats import hmean
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import uuid

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load models
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
nlp_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0 if torch.cuda.is_available() else -1)
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# Common candidate labels
candidate_labels = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
    "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose"
]

app = FastAPI()
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

#######################
# Helper Functions
#######################

def get_text_easyocr(image_path):
    results = reader.readtext(image_path, detail=0)
    return " ".join(results).strip()

def get_nlp_scores(text, labels=candidate_labels):
    if not text.strip():
        return {label: 0.0 for label in labels}
    result = nlp_classifier(text, labels, multi_label=True)
    return {label: score for label, score in zip(result['labels'], result['scores'])}

def get_clip_scores(image_path, labels=candidate_labels):
    image = Image.open(image_path).convert("RGB")
    image_input = clip_preprocess(image).unsqueeze(0).to(device)
    text_inputs = clip.tokenize(labels).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text_inputs)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    logits = 100.0 * image_features @ text_features.T
    probs = torch.softmax(logits, dim=-1)[0]
    return {label: prob.item() for label, prob in zip(labels, probs)}

def combine_scores_max(scores):
    return max(scores.values())

def classify_image(image_path, method="auto"):
    text = get_text_easyocr(image_path)
    if method == "ocr" or (method == "auto" and len(text.strip()) > 10):
        scores = get_nlp_scores(text)
        source = "Text-based image"
    else:
        scores = get_clip_scores(image_path)
        source = "Non-text-based image"

    max_score = combine_scores_max(scores)

    return {
        "filename": os.path.basename(image_path),
        "type": source,
        "prediction_score": f"{max_score * 100:.2f}%"

    }

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    try:
        temp_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{file.filename}")
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = classify_image(temp_path)
        os.remove(temp_path)
        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
