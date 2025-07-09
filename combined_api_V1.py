# combined_api_V1.py  –  FastAPI + upload form + Bayesian fusion
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
    if not text:
        return {l: 1/len(CANDIDATE_LABELS) for l in CANDIDATE_LABELS}
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

def top2_scores(post: dict[str, float]) -> tuple[str, str]:
    vals = sorted(post.values(), reverse=True)[:2]
    return f"{vals[0]*100:.2f}%", f"{vals[1]*100:.2f}%"

# ── 4.  Browser-friendly upload form (GET /) ───────────────────
@app.get("/", response_class=HTMLResponse)
async def upload_form():
    return """
    <html>
      <head>
        <title>City-Classifier Demo</title>
        <style>
          body{font-family:Arial,Helvetica,sans-serif;margin:40px}
          .box{border:1px solid #ccc;padding:20px;width:320px}
        </style>
      </head>
      <body>
        <h2>Upload clue image(s)</h2>
        <div class="box">
          <form action="/upload" method="post" enctype="multipart/form-data">
            <input type="file" name="files" accept="image/*" multiple required><br><br>
            <button type="submit">Analyze</button>
          </form>
        </div>
      </body>
    </html>
    """

# ── 5.  Handle form submit → show HTML result (POST /upload) ──
@app.post("/upload", response_class=HTMLResponse)
async def handle_upload(files: list[UploadFile] = File(...)):
    json_result = await analyze_images(files)          # reuse API below
    data = json_result.body.decode()
    return f"""
    <html>
      <head><title>Result</title></head>
      <body>
        <h3>Analysis Result</h3>
        <pre>{data}</pre>
        <a href="/">⬅️ Upload another</a>
      </body>
    </html>
    """

# ── 6.  Core API endpoint (JSON) ───────────────────────────────
@app.post("/analyze-images")
async def analyze_images(files: list[UploadFile] = File(...)):
    paths = []
    try:
        for f in files:
            p = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{f.filename}")
            with open(p, "wb") as buf: shutil.copyfileobj(f.file, buf)
            paths.append(p)

        # 6-A  Bayesian fusion
        post = {l: 1/len(CANDIDATE_LABELS) for l in CANDIDATE_LABELS}
        post = bayes(post, nlp_like(" ".join(ocr_text(p) for p in paths)))
        for p in paths:
            post = bayes(post, clip_like(p))

        score1, score2 = top2_scores(post)
        return JSONResponse({
            "filename": os.path.basename(paths[0]),
            "type": "Clues Image_BayesianLogic",
            "max_confidence":    score1,
            "second_confidence": score2
        })

    finally:
        for p in paths:
            os.remove(p)
