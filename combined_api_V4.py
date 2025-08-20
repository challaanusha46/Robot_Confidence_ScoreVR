# combined_api_V6.py – FastAPI + conditional fusion + server-side sessions
# Auto session: prefer X-Device-ID header; else IP+User-Agent; cookie for browsers
# --------------------------------------------------------------------------------
import os, uuid, shutil, threading
from collections import OrderedDict
from typing import Optional, Tuple
import torch, clip, easyocr
from PIL import Image
from transformers import pipeline
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, HTMLResponse

# 1) Model setup -------------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[info] device: {device}")

reader  = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
nlp     = pipeline("zero-shot-classification",
                   model="facebook/bart-large-mnli",
                   device=0 if torch.cuda.is_available() else -1)
clip_model, clip_pre = clip.load("ViT-B/32", device=device)

# Top 20 U.S. cities
CANDIDATE_LABELS = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
    "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose",
    "Austin", "Jacksonville", "Fort Worth", "Columbus", "San Francisco",
    "Charlotte", "Indianapolis", "Seattle", "Denver", "Washington, D.C."
]

# 2) FastAPI scaffolding -------------------------------------------------------------------
app = FastAPI()
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 3) Server-side sessions (no TTL; LRU capped) --------------------------------------------
MAX_SESSIONS = 200
LOCK = threading.Lock()
SESSION: "OrderedDict[str, dict]" = OrderedDict()  # session_id -> {"post": posterior_dict}

def _uniform_prior():
    return {l: 1/len(CANDIDATE_LABELS) for l in CANDIDATE_LABELS}

def _evict_if_needed():
    while len(SESSION) > MAX_SESSIONS:
        SESSION.popitem(last=False)  # evict least-recently-used

def get_or_create_prior(session_id: str):
    with LOCK:
        if session_id in SESSION:
            SESSION.move_to_end(session_id)
            return SESSION[session_id]["post"]
        prior = _uniform_prior()
        SESSION[session_id] = {"post": prior}
        SESSION.move_to_end(session_id)
        _evict_if_needed()
        return prior

def save_posterior(session_id: str, post: dict):
    with LOCK:
        SESSION[session_id] = {"post": post}
        SESSION.move_to_end(session_id)
        _evict_if_needed()

def reset_session(session_id: str):
    with LOCK:
        SESSION[session_id] = {"post": _uniform_prior()}
        SESSION.move_to_end(session_id)

def delete_session(session_id: str):
    with LOCK:
        if session_id in SESSION:
            del SESSION[session_id]

# 4) Session key derivation ----------------------------------------------------------------
def _derive_session_id(request: Request) -> Tuple[str, str]:
    """
    Returns (session_id, source): source in {"cookie","device","derived"}.
    Priority:
      1) cookie 'session_id' (browsers)
      2) header 'X-Device-ID' (Unity recommended: SystemInfo.deviceUniqueIdentifier)
      3) derived from client IP + User-Agent (works out-of-the-box)
    """
    # 1) Cookie (browsers)
    cookie_sid = request.cookies.get("session_id")
    if cookie_sid:
        return cookie_sid, "cookie"

    # 2) Explicit device header (preferred for Unity)
    dev = request.headers.get("X-Device-ID")
    if dev:
        sid = uuid.uuid5(uuid.NAMESPACE_DNS, f"device|{dev}").hex
        return sid, "device"

    # 3) Fallback: IP + UA (enable --proxy-headers so IP is real behind proxies)
    fwd = request.headers.get("X-Forwarded-For")
    ip  = fwd.split(",")[0].strip() if fwd else (request.client.host or "0.0.0.0")
    ua  = request.headers.get("User-Agent", "")
    sid = uuid.uuid5(uuid.NAMESPACE_DNS, f"{ip}|{ua}").hex
    return sid, "derived"

# 5) ML helpers ----------------------------------------------------------------------------
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

def top2(post: dict[str, float]):
    ordered = sorted(post, key=post.get, reverse=True)[:2]
    return [(lbl, f"{post[lbl]*100:.2f}%") for lbl in ordered]

# 6) HTML form (manual test; sets cookie so browsers stick to a session) -------------------
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
        <form action="/reset-session" method="post">
          <button type="submit">Reset This Session</button>
        </form>
      </p>
    </body></html>
    """

@app.post("/upload", response_class=HTMLResponse)
async def handle_upload(request: Request, files: list[UploadFile] = File(...)):
    res = await analyze_images(request=request, files=files)
    html = f"<pre>{res.body.decode()}</pre><a href='/'>⬅️ back</a>"
    page = HTMLResponse(html)
    # ensure browser gets a cookie so it stays in the same session
    sid, source = _derive_session_id(request)
    if source != "cookie":
        page.set_cookie(key="session_id", value=sid, httponly=True, samesite="Lax")
    return page

# 7) Session management endpoints ----------------------------------------------------------
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

# 8) Core API: conditional fusion + server-side session -----------------------------------
@app.post("/analyze-images")
async def analyze_images(request: Request, files: list[UploadFile] = File(...)):
    # choose/derive a session id automatically
    session_id, _ = _derive_session_id(request)

    paths = []
    try:
        # Save uploads (no lock; I/O should not block session store)
        for f in files:
            p = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{f.filename}")
            with open(p, "wb") as buf: shutil.copyfileobj(f.file, buf)
            paths.append(p)

        # Pull/update posterior for this session
        posterior = get_or_create_prior(session_id)

        # Per-image: OCR+NLP if text, else CLIP; then Bayes update
        for p in paths:
            text = ocr_text(p)
            likelihood = nlp_like(text) if len(text) > 10 else clip_like(p)
            posterior = bayes(posterior, likelihood)

        save_posterior(session_id, posterior)

        (c1, s1), (c2, s2) = top2(posterior)
        payload = {
            "session_id": session_id,
            "top1_city": c1, "top1_conf": s1,
            "top2_city": c2, "top2_conf": s2
        }
        resp = JSONResponse(payload)
        # For browsers: set cookie so subsequent calls reuse the same session
        resp.set_cookie(key="session_id", value=session_id, httponly=True, samesite="Lax")
        return resp

    finally:
        for p in paths:
            os.remove(p)
