import torch
import open_clip
from PIL import Image
import numpy as np

# Load the CLIP model (ViT-B-32)
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# Load any outfit photo in your folder
image = preprocess(Image.open("test_outfit.jpg")).unsqueeze(0)

# Define labels (you can add more later)
labels = ["casual outfit", "formal outfit", "party wear", "office wear", "jeans", "t-shirt", "dress", "shoes"]
text_tokens = tokenizer(labels)

# Encode
with torch.no_grad():
    img_feat = model.encode_image(image)
    txt_feat = model.encode_text(text_tokens)

# Normalize + compute cosine similarity
img_feat /= img_feat.norm(dim=-1, keepdim=True)
txt_feat /= txt_feat.norm(dim=-1, keepdim=True)

sims = (img_feat @ txt_feat.T).squeeze(0)
best = labels[sims.argmax().item()]

print(f"👗 Best Match: {best}")
