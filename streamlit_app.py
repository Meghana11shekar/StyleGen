import streamlit as st
from PIL import Image, ImageOps
import torch
import numpy as np
import sys
import os

# ---- fake annoy ----
sys.modules['annoy'] = __import__('fake_annoy')

# ---- fashion clip ----
from fashion_clip.fashion_clip import FashionCLIP

model = FashionCLIP("fashion-clip")

from supabase import create_client
from dotenv import load_dotenv
from utils.colour_detect import detect_dominant_color

# ---- load env
load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# ---- Streamlit UI
st.set_page_config(page_title="StyleGen", layout="wide")
st.title("🧥 StyleGen — FashionCLIP version")

labels = [
    "casual outfit", "streetwear outfit", "party outfit", "formal outfit",
    "t-shirt", "crop top", "hoodie", "sweatshirt", "shirt",
    "jeans", "skirt", "dress", "trousers", "shorts",
    "heels", "sneakers", "boots"
]

uploaded_file = st.file_uploader("Upload an outfit", ["png", "jpg", "jpeg"])

def predict_outfit(image):
    img = ImageOps.exif_transpose(image).convert("RGB")
    img = img.resize((224, 224))

    # embedding image
    img_feats = model.encode_images([img], batch_size=1)[0]
    text_feats = model.encode_text(labels, batch_size=len(labels))

    img_tensor = torch.tensor(img_feats)
    text_tensor = torch.tensor(text_feats)

    sims = torch.matmul(text_tensor, img_tensor)
    probs = torch.softmax(sims, dim=0)

    idx = int(torch.argmax(probs))
    return labels[idx], float(probs[idx]), img_feats.tolist()


# -----------------------------------
# PROCESS UPLOAD
# -----------------------------------
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)

    best_label, conf, embedding = predict_outfit(image)

    # dominant color
    color = detect_dominant_color(uploaded_file)

    st.success(f"Detected: {best_label} ({color}) — {conf:.2f}")

    # save metadata
    supabase.table("items").insert({
        "item_name": uploaded_file.name,
        "style": best_label,
        "color": color,
        "embedding": embedding
    }).execute()

# -----------------------------------
# WARDROBE VIEW
st.header("👗 Your Wardrobe Collection")

try:
    response = supabase.table("items").select("*").order("id", desc=True).execute()
    items = response.data

    if items:
        cols = st.columns(3)
        for i, item in enumerate(items):
            with cols[i % 3]:
                st.markdown(
                    f"""
                    <div style="
                        background:white;
                        padding:15px;
                        border-radius:12px;
                        margin:5px;
                        box-shadow:0 4px 10px rgba(0,0,0,0.08);
                        text-align:center;
                    ">
                        <h4>{item['style']}</h4>
                        <p>Color: {item['color']}</p>
                        <p style="font-size:13px;color:gray">{item['item_name']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    else:
        st.info("🧺 No outfits saved yet — upload one!")
except Exception as e:
    st.error(f"⚠️ Error loading wardrobe: {e}")

# -----------------------------------
# Recommendations
# -----------------------------------
st.header("🧩 Outfit Recommendations")

def recommend(style):
    response = supabase.table("items").select("*").execute()
    items = response.data

    target = next((i for i in items if i["style"] == style), None)
    if not target:
        return []

    target_vec = np.array(target["embedding"])

    recs = []
    for item in items:
        if not item.get("embedding"):
            continue
        v = np.array(item["embedding"])
        denom = np.linalg.norm(target_vec) * np.linalg.norm(v)
        if denom == 0:
            continue
        sim = float(np.dot(target_vec, v) / denom)
        recs.append((item, sim))

    recs.sort(key=lambda x: x[1], reverse=True)
    return [i[0] for i in recs[:3]]

choice = st.selectbox("Pick item to match", labels)

if st.button("Suggest"):
    outfits = recommend(choice)

    if outfits:
        st.success("Here are best matches:")
        for o in outfits:
            st.write(o["style"], " - ", o["color"])
    else:
        st.warning("Upload more items!")
