# City-Classifier FastAPI (V14)

FastAPI service that fuses **EasyOCR + BART-MNLI + CLIP (ViT-B/32)** with **tempered Bayes** to infer a U.S. city from one or more “clue” images.
Sessions are tracked **server-side** (no client session id required).

---

## Quick start

```bash
# optional venv
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
            openai-clip easyocr transformers pillow fastapi uvicorn[standard]

uvicorn combined_api_V14:app --reload --host 0.0.0.0 --port 8000
```

> GPU optional — if CUDA is present, the app auto-uses `cuda:0`.

---

## Endpoints

| URL               | Verb   | Description                                            |
| ----------------- | ------ | ------------------------------------------------------ |
| `/`               | GET    | Minimal HTML upload form (manual testing)              |
| `/upload`         | POST   | Handles the form and prints the JSON result            |
| `/analyze-images` | POST   | **Main JSON API** — upload 1..N images as `files=@...` |
| `/reset-session`  | POST   | Clear the server-side posterior (start a fresh run)    |
| `/session`        | DELETE | Delete the server-side session                         |
| `/docs`           | GET    | Swagger UI                                             |

---

## Input / Output

### Input (to `POST /analyze-images`)

* **Content-Type:** `multipart/form-data`
* **Field name:** `files` (repeatable; 1..N images per call)
* **Accepted types:** `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`
* **Session:** no token needed. The server derives a session (cookie → `X-Device-ID` → IP+UA).
  For VR/Unity, send header `X-Device-ID: <stable device id>` to keep runs separate.

**cURL example**

```bash
curl -F "files=@Positive_Imgs/Clue_01.png" \
     -F "files=@Positive_Imgs/Clue_02.jpg" \
     http://127.0.0.1:8000/analyze-images
```

### Output (JSON)

```json
{
  "session_id": "97afc0eca1cb5770941cbb20c5deafeb",
  "final_top1_city": "New York",   "final_top1_conf": "62.10%",
  "final_top2_city": "Seattle",    "final_top2_conf": "28.40%"
}
```

* Values reflect the **posterior after this call**, including earlier clues in the same session.
* V14 returns **final top-2 only** (no per-image breakdown) for simple VR integration.

---

## How a clue is processed

1. **Choose model**

   * `len(OCR_text) > 10` → score cities with **BART-MNLI** (text-based clue).
   * otherwise → score cities with **CLIP** (visual clue).

2. **Bayesian update (tempered)**
   For each city `c` among `K=20` labels:

   ```
   like_t[c]  = max(like[c], 1e-12) ** 0.7          # temper overconfident clues
   post_raw[c] = prior[c] * like_t[c]
   post[c]     = post_raw[c] / sum_k post_raw[k]
   post[c]     = (1 - 0.02) * post[c] + 0.02 / K    # light forgetting
   if max(post) >= 0.98:
       post[c] = (1 - 0.10) * post[c] + 0.10 / K    # avoid 100% lock-in
   ```

   The resulting `post` becomes the next `prior` within the session.

---

## Unity usage (one-liner)

* POST images as multipart form-data to `/analyze-images`.
* Send `X-Device-ID` header (e.g., `SystemInfo.deviceUniqueIdentifier`).
* Call `POST /reset-session` once before a new VR run.

---

## Customize

* **Cities:** edit `CANDIDATE_LABELS` in `combined_api_V14.py`.
* **Text/visual switch:** change `TEXT_LEN_THRESHOLD` (default `10`).
* **Saturation knobs:** `TAU`, `MIX_GAMMA`, `SAT_THRESH`, `SAT_GAMMA`.

---

## Daily update

<!-- DAILY:START -->
_Last updated: (automation will fill this)_
<!-- DAILY:END -->
