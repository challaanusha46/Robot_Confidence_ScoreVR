import torch, clip
from PIL import Image
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def ny_confidence(image_path, prompt="a photo of the New York City skyline"):
    # 1. encode image
    img = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        img_feat = model.encode_image(img)
        img_feat /= img_feat.norm(dim=-1, keepdim=True)

    # 2. encode text prompt
    txt = clip.tokenize([prompt]).to(device)
    with torch.no_grad():
        txt_feat = model.encode_text(txt)
        txt_feat /= txt_feat.norm(dim=-1, keepdim=True)

    # 3. cosine similarity → scaled logistic confidence
    sim = (img_feat @ txt_feat.T).item()          # value in [‑1,1]
    # OpenAI usually multiplies by 100 then does soft‑max for multi‑label.
    # With one label we can map similarity to [0,1] via a logistic:
    conf = 1 / (1 + np.exp(-10 * sim))            # temperature=0.1
    return sim, conf                              # raw similarity & [0‑1] confidence

image = "C:/Users/anush/OneDrive - University of Georgia/Thesis/Python_GeneralFiles/Positive_Imgs/Clue- NonText.jpg"   # your skyline
sim, conf = ny_confidence(image)
print(f"Cosine sim = {sim:+.3f}   →   confidence ≈ {conf*100:.1f}%")
