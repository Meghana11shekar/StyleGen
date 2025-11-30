import streamlit as st
from PIL import Image, ImageOps
import torch
import numpy as np
import sys
import os

# PAGE SETTINGS
st.set_page_config(page_title="StyleGen", layout="wide")

# remove streamlit default header so title is not cut
st.markdown("""
<style>
[data-testid="stHeader"] {
    background-color: transparent;
    height: 0px;
}
</style>
""", unsafe_allow_html=True)

# TITLE
st.markdown("""
<div style='text-align:center; padding-top:15px; padding-bottom:0px;'>
    <h1 style='color:#8c5fbf; font-size:56px; font-weight:900;'>
        🧥 StyleGen — FashionCLIP
    </h1>
    <p style='font-size:22px; color:#6b5a76; margin-top:-8px;'>
        Your personal AI stylist powered by CLIP + Supabase
    </p>
</div>
""", unsafe_allow_html=True)


# ---- pastel theme CSS ----
st.markdown("""
<style>

:root {
    --bg-color: #f9f2ff;
    --card-bg: #ffffff;
    --accent: #b381f2;
    --accent2: #f2baff;
    --text-dark: #362e3d;
}

/* APP BACKGROUND */
[data-testid="stAppViewContainer"] {
    background-color: var(--bg-color) !important;
}

/* SECTION */
.section {
    padding: 20px;
    border-radius: 20px;
    background: var(--card-bg);
    margin-top: 25px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.1);
}

/* CARDS */
.card {
    background: var(--card-bg);
    padding: 18px;
    border-radius: 15px;
    margin: 10px;
    box-shadow: 0 6px 16px rgba(0,0,0,0.12);
    text-align:center;
    transition: transform 0.2s ease;
    border: 2px solid #e8d6ff;
    font-weight:500;
}
.card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 30px rgba(0,0,0,0.15);
}

/* TITLES */
.section-title {
    font-size: 28px;
    font-weight: 700;
    color: var(--accent);
    margin-bottom: 15px;
}

/* BUTTON */
.stButton>button {
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    color: white;
    border-radius: 14px;
    padding: 12px 22px;
    font-weight: 700;
    width: 100%;
    border: none;
    font-size: 17px;
}

/* FONT */
html, body, p, div, span, label, h4 {
    font-family: "Poppins", sans-serif;
    color: var(--text-dark) !important;
}

img { border-radius: 14px; }

</style>
""", unsafe_allow_html=True)


# ---- fake annoy ----
sys.modules['annoy'] = __import__('fake_annoy')

# ---- FashionCLIP ----
from fashion_clip.fashion_clip import FashionCLIP
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU

model = FashionCLIP("fashion-clip")

# IMPORTANT: move model safely off meta tensors
for name, param in model.model.named_parameters():
    if param.device.type == "meta":
        param.data = torch.zeros_like(param, device="cpu")

model.model = model.model.to("cpu")



from supabase import create_client
from dotenv import load_dotenv
from utils.colour_detect import detect_dominant_color

# ---- load env
load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

labels = [
    "casual outfit", "streetwear outfit", "party outfit", "formal outfit",
    "t-shirt", "crop top", "hoodie", "sweatshirt", "shirt",
    "jeans", "skirt", "dress", "trousers", "shorts",
    "heels", "sneakers", "boots"
]


# -----------------------------------
# UPLOAD SECTION
# -----------------------------------
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Upload a Clothing Item 👇</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", ["png", "jpg", "jpeg"])


def predict_outfit(image):
    img = ImageOps.exif_transpose(image).convert("RGB")
    img = img.resize((224, 224))

    img_feats = model.encode_images([img], batch_size=1)[0]
    text_feats = model.encode_text(labels, batch_size=len(labels))

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


# -----------------------------------
# WARDROBE SECTION
# -----------------------------------
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>👗 Your Wardrobe Collection</div>", unsafe_allow_html=True)

try:
    response = supabase.table("items").select("*").order("id", desc=True).execute()
    items = response.data

    if items:
        cols = st.columns(3)
        for i, item in enumerate(items):
            with cols[i % 3]:
                st.markdown(
                    f"""
                    <div class="card">
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

st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------
# RECOMMENDATIONS FIXED VERSION
# -----------------------------------
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>🧩 Outfit Recommendations</div>", unsafe_allow_html=True)

def recommend(style):
    response = supabase.table("items").select("*").execute()
    items = response.data

    # compute virtual text embedding
    text_embed = model.encode_text([style], batch_size=1)[0]

    recs = []
    for item in items:
        if not item.get("embedding"):
            continue

        v = np.array(item["embedding"])
        sim = float(np.dot(text_embed, v) /
                    (np.linalg.norm(text_embed) * np.linalg.norm(v)))
        recs.append((item, sim))

    recs.sort(key=lambda x: x[1], reverse=True)
    return [i[0] for i in recs[:3]]


choice = st.selectbox("Pick item to match", labels)

if st.button("Suggest"):
    outfits = recommend(choice)

    if outfits:
        st.success("Best Matches:")
        for o in outfits:
            st.write(o["style"], " - ", o["color"])
    else:
        st.warning("Upload more items!")

st.markdown("</div>", unsafe_allow_html=True)
