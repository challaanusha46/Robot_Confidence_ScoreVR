# combined_api_V10.py – FastAPI + OCR/NLP/CLIP Bayesian fusion
#  - Server-side sessions (no client session id needed)
#  - Document gating: skip text-heavy pages with no city signal (coverage/words/keywords)
#  - Stricter CLIP gating (min top-1 and margin)
#  - Text semantics (FROM/TO + mentions)
#  - Duplicate guard (same file not re-counted)
#  - Response = final top-2 only (no per-image steps)
# -----------------------------------------------------------------------------------------
import os, uuid, shutil, threading, math, re, unicodedata, hashlib
from collections import OrderedDict
from typing import Optional, Tuple, Dict, List

import torch, clip, easyocr
from PIL import Image
from transformers import pipeline
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, HTMLResponse

# --------------------------- 1) Model setup ----------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[info] device: {device}")

reader  = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
nlp     = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0 if torch.cuda.is_available() else -1
)
clip_model, clip_pre = clip.load("ViT-B/32", device=device)

# Top 20 U.S. cities
CANDIDATE_LABELS: List[str] = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
    "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose",
    "Austin", "Jacksonville", "Fort Worth", "Columbus", "San Francisco",
    "Charlotte", "Indianapolis", "Seattle", "Denver", "Washington, D.C."
]

# --------------------------- 2) Heuristics (no data tuning needed) -----------------------
CONFIG = dict(
    H_gate=0.97,       # global entropy gate (very flat distributions → skip)
    w_to=2.0,          # text: boost city after "TO"
    w_from=0.7,        # text: damp city after "FROM"
    w_mention=1.3      # text: boost any direct city mention
)

# Document gating (prevents posters/blueprints from drifting posterior)
TEXTY_MIN_CHARS = 120    # OCR chars ≥ this AND no city signal → treat as document
TEXT_AREA_MIN   = 0.08   # ≥8% of image covered by OCR boxes → document
WORD_COUNT_MIN  = 30     # ≥30 words → document
POSTER_KEYS     = ["wanted", "reward"]  # doc keywords that don't imply a city

# CLIP must be decisive if text gives no signal
CLIP_MIN_TOP1   = 0.40   # winner prob must be at least 0.40
CLIP_MIN_MARGIN = 0.15   # winner must beat runner-up by ≥ 0.15

# --------------------------- 3) FastAPI & sessions ---------------------------------------
app = FastAPI()
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAX_SESSIONS = 200
LOCK = threading.Lock()
# session_id -> {"post": posterior_dict, "seen": set(file_hashes)}
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
        SESSION[session_id] = {"post": prior, "seen": set()}
        SESSION.move_to_end(session_id)
        _evict_if_needed()
        return prior

def save_posterior(session_id: str, post: Dict[str, float]) -> None:
    with LOCK:
        seen = SESSION.get(session_id, {}).get("seen", set())
        SESSION[session_id] = {"post": post, "seen": seen}
        SESSION.move_to_end(session_id)
        _evict_if_needed()

def reset_session(session_id: str) -> None:
    with LOCK:
        SESSION[session_id] = {"post": _uniform_prior(), "seen": set()}
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

# --------------------------- 4) ML helpers ------------------------------------------------
def _clean(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "").lower()

def ocr_text_and_coverage(path: str) -> Tuple[str, float]:
    """OCR text + % image area covered by OCR boxes (document-ness)."""
    results = reader.readtext(path, detail=1)  # [([pts], text, conf), ...]
    text = " ".join([r[1] for r in results]).strip()

    try:
        W, H = Image.open(path).size
        img_area = float(W * H) if W and H else 1.0
    except Exception:
        img_area = 1.0

    text_area = 0.0
    for r in results:
        pts = r[0]
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        w = max(xs) - min(xs); h = max(ys) - min(ys)
        if w > 0 and h > 0:
            text_area += w * h

    coverage = min(text_area / img_area, 1.0)
    return text, coverage

def nlp_like(text: str) -> Dict[str, float]:
    res = nlp(text, CANDIDATE_LABELS, multi_label=True)
    return {l: s for l, s in zip(res["labels"], res["scores"])}

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

def bayes(prior: Dict[str, float], like: Dict[str, float]) -> Dict[str, float]:
    post = {l: prior[l] * like[l] for l in prior}
    z = sum(post.values()) or 1e-12
    return {l: v / z for l, v in post.items()}

def top2(post: Dict[str, float]):
    ordered = sorted(post, key=post.get, reverse=True)[:2]
    return [(lbl, f"{post[lbl]*100:.2f}%") for lbl in ordered]

def entropy(dist: Dict[str, float]) -> float:
    ps = list(dist.values()); s = sum(ps) or 1e-12
    ps = [p/s for p in ps]
    H = -sum(p * math.log(p + 1e-12) for p in ps)
    return H / math.log(len(ps))

CITY_ALIASES = {
    "New York":        [r"\bnew york\b", r"\bnyc\b", r"\bnew york city\b"],
    "Los Angeles":     [r"\blos angeles\b", r"\bla\b(?!\w)"],
    "Chicago":         [r"\bchicago\b"],
    "Houston":         [r"\bhouston\b"],
    "Phoenix":         [r"\bphoenix\b"],
    "Philadelphia":    [r"\bphiladelphia\b", r"\bphilly\b"],
    "San Antonio":     [r"\bsan antonio\b"],
    "San Diego":       [r"\bsan diego\b"],
    "Dallas":          [r"\bdallas\b"],
    "San Jose":        [r"\bsan jose\b"],
    "Austin":          [r"\baustin\b"],
    "Jacksonville":    [r"\bjacksonville\b"],
    "Fort Worth":      [r"\bfort worth\b"],
    "Columbus":        [r"\bcolumbus\b"],
    "San Francisco":   [r"\bsan francisco\b", r"\bsf\b(?!\w)"],
    "Charlotte":       [r"\bcharlotte\b"],
    "Indianapolis":    [r"\bindianapolis\b"],
    "Seattle":         [r"\bseattle\b"],
    "Denver":          [r"\bdenver\b"],
    "Washington, D.C.":[r"\bwashington\b(?:,?\s*d\.?c\.?)*"],
}
CITY_REGEX = {c: re.compile("|".join(p), re.I) for c, p in CITY_ALIASES.items()}
ROUTE_RE = re.compile(
    r"\bfrom\s+(?P<from>[^,\n]+?)(?:,?\s*[A-Z]{2})?\b.*?\bto\s+(?P<to>[^,\n]+?)(?:,?\s*[A-Z]{2})?\b",
    re.I | re.S
)

def _match_city_fragment(fragment: str) -> Optional[str]:
    frag = _clean(fragment)
    for city, rgx in CITY_REGEX.items():
        if rgx.search(frag): return city
    return None

def nlp_like_semantic(text: str) -> Dict[str, float]:
    base = nlp_like(text)
    w = {c: 1.0 for c in CANDIDATE_LABELS}
    t = _clean(text)

    m = ROUTE_RE.search(t)
    if m:
        c_from = _match_city_fragment(m.group("from") or "")
        c_to   = _match_city_fragment(m.group("to")   or "")
        if c_to: w[c_to]   *= CONFIG["w_to"]
        if c_from and c_from != c_to: w[c_from] *= CONFIG["w_from"]

    for city, rgx in CITY_REGEX.items():
        if rgx.search(t): w[city] *= CONFIG["w_mention"]

    adj = {c: base[c] * w[c] for c in CANDIDATE_LABELS}
    s = sum(adj.values()) or 1e-12
    return {c: v / s for c, v in adj.items()}

def choose_likelihood(path: str, text: str, coverage: float | None = None) -> Tuple[Optional[Dict[str, float]], str, float]:
    """
    Return (likelihood or None, modality_used, entropy_of_chosen). None => skip update.
      - If text has explicit city signal (mentions or FROM/TO): compare text vs image by entropy.
      - If document-like (coverage/words/chars OR poster keywords) with no city signal: skip.
      - Otherwise rely on CLIP but require top1 and margin thresholds.
    """
    like_img = clip_like(path);  H_img = entropy(like_img)

    t = _clean(text)
    has_route   = bool(ROUTE_RE.search(t))
    has_mention = any(r.search(t) for r in CITY_REGEX.values())

    # document heuristics
    word_count  = len(re.findall(r"[a-zA-Z]+", t))
    is_poster   = any(k in t for k in POSTER_KEYS)
    is_texty    = (len(t) >= TEXTY_MIN_CHARS) or \
                  (coverage is not None and coverage >= TEXT_AREA_MIN) or \
                  (word_count >= WORD_COUNT_MIN)

    # 1) explicit city text → use sharper modality
    if has_route or has_mention:
        like_text = nlp_like_semantic(text); H_text = entropy(like_text)
        chosen, H, modality = (like_text, H_text, "text") if H_text <= H_img else (like_img, H_img, "image")
        return (None, modality, H) if H >= CONFIG["H_gate"] else (chosen, modality, H)

    # 2) document/poster with no city signal → skip
    if is_texty or is_poster:
        return None, "text", 1.0

    # 3) image-only: require decisive CLIP
    ordered = sorted(like_img.values(), reverse=True)
    top1 = ordered[0]; top2 = ordered[1] if len(ordered) > 1 else 0.0
    if top1 < CLIP_MIN_TOP1 or (top1 - top2) < CLIP_MIN_MARGIN:
        return None, "image", H_img

    return (None, "image", H_img) if H_img >= CONFIG["H_gate"] else (like_img, "image", H_img)

# --------------------------- 5) Duplicate guard ------------------------------------------
def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

# --------------------------- 6) HTML form (manual test) ----------------------------------
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
async def handle_upload(request: Request, files: List[UploadFile] = File(...)):
    res = await analyze_images(request=request, files=files)
    html = f"<pre>{res.body.decode()}</pre><a href='/'>⬅️ back</a>"
    page = HTMLResponse(html)
    sid, src = _derive_session_id(request)
    if src != "cookie":
        page.set_cookie(key="session_id", value=sid, httponly=True, samesite="Lax")
    return page

# --------------------------- 7) Session management ---------------------------------------
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

# --------------------------- 8) Core API (final-only response) ---------------------------
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
            # duplicate guard
            try:
                h = _file_sha256(p)
                with LOCK:
                    seen = SESSION[session_id].setdefault("seen", set())
                    if h in seen: 
                        continue
                    seen.add(h)
            except Exception:
                pass

            text, coverage = ocr_text_and_coverage(p)
            like, modality, H = choose_likelihood(p, text, coverage)
            if like is None:
                # print(f"[skip] {os.path.basename(p)} mod={modality} H={H:.3f} cov={coverage:.3f}")
                continue

            posterior = bayes(posterior, like)
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("combined_api_V10:app", host="127.0.0.1", port=8000, reload=True)
