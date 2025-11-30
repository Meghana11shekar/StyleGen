import os
import numpy as np
import torch
from dotenv import load_dotenv
from supabase import create_client
from fashion_clip.fashion_clip import FashionCLIP

# ---- env + supabase ----
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---- force CPU, no GPU ----
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ---- load FashionCLIP once ----
model = FashionCLIP("fashion-clip")

# fix meta-tensor issue: convert any meta params to real CPU tensors
for name, p in model.model.named_parameters():
    if p.device.type == "meta":
        p.data = torch.zeros_like(p, device="cpu")

model.model = model.model.to("cpu")


# ---- shared labels ----
labels = [
    "casual outfit","streetwear outfit","party outfit","formal outfit",
    "t-shirt","crop top","hoodie","sweatshirt","shirt",
    "jeans","skirt","dress","trousers","shorts",
    "heels","sneakers","boots"
]

# ---- helper encoders ----
def encode_image(img):
    """Return image embedding as 1D numpy-like vector."""
    return model.encode_images([img], batch_size=1)[0]

def encode_text(text):
    """Return text embedding for a single style string."""
    return model.encode_text([text], batch_size=1)[0]
