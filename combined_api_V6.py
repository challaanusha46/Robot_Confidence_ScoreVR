# combined_api_V14.py – Simple rule + Tempered Bayes (NO duplicate check)
# - If OCR text length > 10 → BART; else → CLIP
# - Server-side sessions (auto, no client token needed)
# - No duplicate guard: the same image counted again will update belief again
# - Tempered Bayes + light forgetting to avoid saturation
# - Response: final top-2 only
# ----------------------------------------------------------------
import os, uuid, shutil, threading, unicodedata
from collections import OrderedDict
from typing import Tuple, Dict, List

import torch, clip, easyocr
from PIL import Image
from transformers import pipeline
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, HTMLResponse

# ---------------- 1) Models ----------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[info] device: {device}")

reader  = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
nlp     = pipeline("zero-shot-classification",
                   model="facebook/bart-large-mnli",
                   device=0 if torch.cuda.is_available() else -1)
clip_model, clip_pre = clip.load("ViT-B/32", device=device)

# Top 20 U.S. cities
CANDIDATE_LABELS: List[str] = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
    "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose",
    "Austin", "Jacksonville", "Fort Worth", "Columbus", "San Francisco",
    "Charlotte", "Indianapolis", "Seattle", "Denver", "Washington, D.C."
]

# ---------------- 2) FastAPI & sessions ---------------
app = FastAPI()
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAX_SESSIONS = 200
LOCK = threading.Lock()
# session_id -> {"post": posterior_dict}
SESSION: "OrderedDict[str, dict]" = OrderedDict()

def _uniform_prior() -> Dict[str, float]:
    return {l: 1/len(CANDIDATE_LABELS) for l in CANDIDATE_LABELS}

def _evict_if_needed() -> None:
    while len(SESSION) > MAX_SESSIONS:
        SESSION.popitem(last=False)

def get_or_create_prior(session_id: str) -> Dict[str, float]:
    with LOCK:
        if session_id in SESSION:
            SESSION.move_to_end(session_id)
            return SESSION[session_id]["post"]
        prior = _uniform_prior()
        SESSION[session_id] = {"post": prior}
        SESSION.move_to_end(session_id); _evict_if_needed()
        return prior

def save_posterior(session_id: str, post: Dict[str, float]) -> None:
    with LOCK:
        SESSION[session_id] = {"post": post}
        SESSION.move_to_end(session_id); _evict_if_needed()

def reset_session(session_id: str) -> None:
    with LOCK:
        SESSION[session_id] = {"post": _uniform_prior()}
        SESSION.move_to_end(session_id)

def delete_session(session_id: str) -> None:
    with LOCK:
        if session_id in SESSION:
            del SESSION[session_id]

def _derive_session_id(request: Request) -> Tuple[str, str]:
    cookie_sid = request.cookies.get("session_id")
    if cookie_sid: return cookie_sid, "cookie"
    dev = request.headers.get("X-Device-ID")
    if dev: return uuid.uuid5(uuid.NAMESPACE_DNS, f"device|{dev}").hex, "device"
    fwd = request.headers.get("X-Forwarded-For")
    ip  = fwd.split(",")[0].strip() if fwd else (request.client.host or "0.0.0.0")
    ua  = request.headers.get("User-Agent", "")
    return uuid.uuid5(uuid.NAMESPACE_DNS, f"{ip}|{ua}").hex, "derived"

# ---------------- 3) Helpers ----------------
def _clean(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "").strip()

def ocr_text(path: str) -> str:
    return " ".join(reader.readtext(path, detail=0)).strip()

def nlp_like(text: str) -> Dict[str, float]:
    res = nlp(text, CANDIDATE_LABELS, multi_label=True)
    return {l: float(s) for l, s in zip(res["labels"], res["scores"])}

def clip_like(path: str) -> Dict[str, float]:
    img = Image.open(path).convert("RGB")
    img_t = clip_pre(img).unsqueeze(0).to(device)
    txt_t = clip.tokenize(CANDIDATE_LABELS).to(device)
    with torch.no_grad():
        i_f = clip_model.encode_image(img_t)
        t_f = clip_model.encode_text(txt_t)
    i_f /= i_f.norm(dim=-1, keepdim=True); t_f /= t_f.norm(dim=-1, keepdim=True)
    probs = (100 * i_f @ t_f.T).softmax(dim=-1)[0]
    return {l: float(probs[i].item()) for i, l in enumerate(CANDIDATE_LABELS)}

# ---------------- 4) Tempered Bayes / anti-saturation ----------------
TAU = 0.7         # temper likelihoods (soften overconfident clues)
MIX_GAMMA = 0.02  # light forgetting each step: 2% blend with uniform
SAT_THRESH = 0.98 # if posterior max ≥ this…
SAT_GAMMA = 0.10  # …blend an extra 10% uniform (unsaturate)

def bayes(prior: Dict[str, float], like: Dict[str, float]) -> Dict[str, float]:
    like_t = {k: max(like[k], 1e-12) ** TAU for k in prior}
    post_raw = {k: prior[k] * like_t[k] for k in prior}
    Z = sum(post_raw.values()) or 1e-12
    post = {k: v / Z for k, v in post_raw.items()}

    if MIX_GAMMA > 0:
        u = 1.0 / len(post)
        post = {k: (1 - MIX_GAMMA) * post[k] + MIX_GAMMA * u for k in post}

    if max(post.values()) >= SAT_THRESH and SAT_GAMMA > 0:
        u = 1.0 / len(post)
        post = {k: (1 - SAT_GAMMA) * post[k] + SAT_GAMMA * u for k in post}

    return post

def top2(post: Dict[str, float]):
    ordered = sorted(post, key=post.get, reverse=True)[:2]
    return [(lbl, f"{post[lbl]*100:.2f}%") for lbl in ordered]

# ---------------- 5) Simple chooser ----------------
TEXT_LEN_THRESHOLD = 10  # requested rule

def choose_likelihood_simple(path: str, text: str) -> Dict[str, float]:
    """If text length > 10 → BART; else → CLIP."""
    if len(_clean(text)) > TEXT_LEN_THRESHOLD:
        return nlp_like(text)
    else:
        return clip_like(path)

# ---------------- 6) HTML (manual test) -------------
@app.get("/", response_class=HTMLResponse)
async def form():
    return """
    <html><body style='font-family:Arial;margin:40px'>
      <h2>Upload clue image(s)</h2>
      <form action="/upload" method="post" enctype="multipart/form-data">
        <input type="file" name="files" accept="image/*" multiple required><br><br>
        <button type="submit">Analyze</button>
      </form>
      <p style="margin-top:20px">
        <form action="/reset-session" method="post"><button type="submit">Reset This Session</button></form>
      </p>
    </body></html>
    """

@app.post("/upload", response_class=HTMLResponse)
async def handle_upload(request: Request, files: List[UploadFile] = File(...)):
    res = await analyze_images(request=request, files=files)
    html = f"<pre>{res.body.decode()}</pre><a href='/'>⬅️ back</a>"
    page = HTMLResponse(html)
    sid, src = _derive_session_id(request)
    if src != "cookie":
        page.set_cookie(key="session_id", value=sid, httponly=True, samesite="Lax")
    return page

# ---------------- 7) Session management --------------
@app.post("/reset-session")
async def api_reset_session(request: Request):
    sid, _ = _derive_session_id(request)
    reset_session(sid)
    resp = JSONResponse({"session_id": sid, "status": "reset"})
    resp.set_cookie(key="session_id", value=sid, httponly=True, samesite="Lax")
    return resp

@app.delete("/session")
async def api_delete_session(request: Request):
    sid, _ = _derive_session_id(request)
    delete_session(sid)
    resp = JSONResponse({"session_id": sid, "status": "deleted"})
    resp.delete_cookie("session_id")
    return resp

# ---------------- 8) Core API (final-only) -----------
@app.post("/analyze-images")
async def analyze_images(request: Request, files: List[UploadFile] = File(...)):
    session_id, _ = _derive_session_id(request)

    paths: List[str] = []
    try:
        for f in files:
            p = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{f.filename}")
            with open(p, "wb") as buf:
                shutil.copyfileobj(f.file, buf)
            paths.append(p)

        posterior = get_or_create_prior(session_id)

        for p in paths:
            text = ocr_text(p)
            likelihood = choose_likelihood_simple(p, text)
            posterior = bayes(posterior, likelihood)
            with LOCK:
                SESSION[session_id]["post"] = posterior

        save_posterior(session_id, posterior)

        (fc1, fs1), (fc2, fs2) = top2(posterior)
        payload = {
            "session_id": session_id,
            "final_top1_city": fc1, "final_top1_conf": fs1,
            "final_top2_city": fc2, "final_top2_conf": fs2
        }
        resp = JSONResponse(payload)
        resp.set_cookie(key="session_id", value=session_id, httponly=True, samesite="Lax")
        return resp

    finally:
        for p in paths:
            try: os.remove(p)
            except: pass

