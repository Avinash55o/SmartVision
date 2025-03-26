import torch
import clip
from PIL import Image
import os
import shutil

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

for _ in range(3):
    try:
        model, preprocess = clip.load("/app/models/ViT-B-32.pt", device=device)
        break
    except RuntimeError as e:
        if "checksum does not not match" in str(e):
            print(f"Checksum error: {e}. Clearing cache and retrying...")
            shutil.rmtree(os.path.expanduser("~/.cache/clip"), ignore_errors=True)
        else:
            raise e
else:
    raise RuntimeError("Failed to load base model after 3 attempts")

# Load fine-tuned weights with weights_only=True for safety
model.load_state_dict(torch.load("/app/models/fine_tuned_clip.pt", weights_only=True))
model.eval()

img_path = "/app/datasets/chest_xray/images/6585_006.png"
if not os.path.exists(img_path):
    print(f"Image not found: {img_path}")
    exit(1)

img = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
labels = ["normal lung", "pneumonia", "lung tumor"]
text = clip.tokenize(labels).to(device)

with torch.no_grad():
    logits_per_image, _ = model(img, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

for label, prob in zip(labels, probs[0]):
    print(f"{label}: {prob:.4f}")