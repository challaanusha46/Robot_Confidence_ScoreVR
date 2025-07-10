# combined_api_V1.py  –  FastAPI + upload form + conditional fusion
# ----------------------------------------------------------------
import os, uuid, shutil, torch, clip, easyocr
from PIL import Image
from transformers import pipeline
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse

# ── 1.  Model setup ────────────────────────────────────────────
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[info] device: {device}")

reader  = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
nlp     = pipeline("zero-shot-classification",
                   model="facebook/bart-large-mnli",
                   device=0 if torch.cuda.is_available() else -1)
clip_model, clip_pre = clip.load("ViT-B/32", device=device)

CANDIDATE_LABELS = [
    # original 10
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
    "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose",
    # extra 7
    "Seattle", "Portland", "Detroit", "San Francisco",
    "Washington", "Denver", "Atlanta"
]

# ── 2.  FastAPI / file system scaffolding ──────────────────────
app = FastAPI()
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── 3.  Helper functions ───────────────────────────────────────
def ocr_text(path: str) -> str:
    return " ".join(reader.readtext(path, detail=0)).strip()

def nlp_like(text: str) -> dict[str, float]:
    res = nlp(text, CANDIDATE_LABELS, multi_label=True)
    return {l: s for l, s in zip(res["labels"], res["scores"])}

def clip_like(path: str) -> dict[str, float]:
    img = Image.open(path).convert("RGB")
    img_t = clip_pre(img).unsqueeze(0).to(device)
    txt_t = clip.tokenize(CANDIDATE_LABELS).to(device)
    with torch.no_grad():
        i_f = clip_model.encode_image(img_t)
        t_f = clip_model.encode_text(txt_t)
    i_f /= i_f.norm(dim=-1, keepdim=True); t_f /= t_f.norm(dim=-1, keepdim=True)
    probs = (100 * i_f @ t_f.T).softmax(dim=-1)[0]
    return {l: probs[i].item() for i, l in enumerate(CANDIDATE_LABELS)}

def bayes(prior: dict[str, float], like: dict[str, float]) -> dict[str, float]:
    post = {l: prior[l] * like[l] for l in prior}
    z = sum(post.values()) or 1e-12
    return {l: v / z for l, v in post.items()}

def top2_scores(post: dict[str, float]) -> tuple[tuple[str,str], tuple[str,str]]:
    ordered = sorted(post, key=post.get, reverse=True)[:2]
    return tuple((lbl, f"{post[lbl]*100:.2f}%") for lbl in ordered)

# ── 4.  Upload form (GET /) ────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def upload_form():
    return """
    <html><head><title>City-Classifier</title></head>
    <body style='font-family:Arial;margin:40px'>
      <h2>Upload clue image(s)</h2>
      <form action="/upload" method="post" enctype="multipart/form-data">
        <input type="file" name="files" accept="image/*" multiple required><br><br>
        <button type="submit">Analyze</button>
      </form>
    </body></html>
    """

# ── 5.  Form handler (POST /upload) ────────────────────────────
@app.post("/upload", response_class=HTMLResponse)
async def handle_upload(files: list[UploadFile] = File(...)):
    json_result = await analyze_images(files)
    return f"""
    <html><body><h3>Result</h3><pre>{json_result.body.decode()}</pre>
    <a href="/">⬅️ Upload another</a></body></html>
    """

# ── 6.  Core API (POST /analyze-images) ────────────────────────
@app.post("/analyze-images")
async def analyze_images(files: list[UploadFile] = File(...)):
    paths = []
    try:
        for f in files:
            p = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{f.filename}")
            with open(p, "wb") as buf: shutil.copyfileobj(f.file, buf)
            paths.append(p)

        # uniform prior
        posterior = {l: 1/len(CANDIDATE_LABELS) for l in CANDIDATE_LABELS}

        # process each image independently
        for p in paths:
            text = ocr_text(p)
            if len(text) > 10:                       # treat as text-based clue
                likelihood = nlp_like(text)
            else:                                    # treat as non-text clue
                likelihood = clip_like(p)
            posterior = bayes(posterior, likelihood)

        (c1, s1), (c2, s2) = top2_scores(posterior)
        return JSONResponse({
            "filename": os.path.basename(paths[0]),
            "type": "Conditional Bayes (text vs visual)",
            "top1_city":  c1, "top1_conf":  s1,
            "top2_city":  c2, "top2_conf":  s2
        })
    finally:
        for p in paths: os.remove(p)

