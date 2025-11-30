import streamlit as st
from PIL import Image, ImageOps
import torch
from utils.core import supabase, encode_image, encode_text, labels
from utils.colour_detect import detect_dominant_color

st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Upload a Clothing Item 👇</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", ["png", "jpg", "jpeg"])


def predict_outfit(image):
    img = ImageOps.exif_transpose(image).convert("RGB")
    img = img.resize((224, 224))

    img_feats = encode_image(img)
    text_feats = [encode_text(lbl) for lbl in labels]

    sims = torch.matmul(torch.tensor(text_feats), torch.tensor(img_feats))
    probs = torch.softmax(sims, dim=0)

    idx = int(torch.argmax(probs))
    return labels[idx], float(probs[idx]), img_feats.tolist()


if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)

    best_label, conf, embedding = predict_outfit(image)
    color = detect_dominant_color(uploaded_file)

    st.success(f"Detected: {best_label} ({color}) — {conf:.2f}")

    supabase.table("items").insert({
        "item_name": uploaded_file.name,
        "style": best_label,
        "color": color,
        "embedding": embedding
    }).execute()

st.markdown("</div>", unsafe_allow_html=True)
