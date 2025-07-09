# City‑Classifier FastAPI

Tiny FastAPI service that Bayes‑fuses **EasyOCR + BART‑MNLI + CLIP ViT‑B/32** to guess a U.S. city from 1‑N clue images.

---

## Run in 3 lines

```bash
python -m venv .venv && . .venv/bin/activate     # optional venv
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
            openai-clip easyocr transformers pillow fastapi uvicorn[standard]
uvicorn combined_api_V1:app --reload             # opens on :8000
```

---

## Endpoints

| URL               | Verb     | What it does                 |
| ----------------- | -------- | ---------------------------- |
| `/`               | **GET**  | Browser form → upload images |
| `/upload`         | **POST** | Shows result page            |
| `/analyze-images` | **POST** | JSON API (`files=@img1` …)   |
| `/docs`           | **GET**  | Swagger UI                   |

### Example JSON response

```json
{"max_confidence":"37.51%","second_confidence":"25.71%"}
```

---

## Customise

- **Cities** → edit `CANDIDATE_LABELS` in the script.
- **GPU** → install CUDA wheels & script auto‑selects `cuda:0`.
- **OCR languages** → `easyocr.Reader(["en","es",…])`.


