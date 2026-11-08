import clip
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Using device: {device}")

model, preprocess = clip.load("ViT-B/32", device = device)

text = clip.tokenize([
    "a photograph of a dog",
    "a photograph of a cat"
]).to(device)

with torch.no_grad():
    features = model.encode_text(text)

print(features.shape)
print("CLIP loaded successfully")