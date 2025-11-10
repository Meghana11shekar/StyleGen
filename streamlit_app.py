import streamlit as st
from PIL import Image
import torch
import open_clip
import numpy as np
from supabase import create_client, Client
from dotenv import load_dotenv
import os
from utils.colour_detect import detect_dominant_color

# --- Load .env variables ---
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Load CLIP model ---
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# --- Streamlit UI ---
st.set_page_config(page_title="StyleGen Wardrobe", page_icon="🪄", layout="wide")

st.markdown(
    """
    <style>
    .main {
        background-color: #f7f8fc;
        font-family: 'Poppins', sans-serif;
    }
    .stButton>button {
        background-color: #7d5fff;
        color: white;
        border-radius: 8px;
        padding: 0.5em 1.2em;
        border: none;
    }
    .stButton>button:hover {
        background-color: #5b3cc4;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🧥 StyleGen — Your AI Wardrobe")
st.subheader("Upload your clothes, auto-tag them, and view your wardrobe ✨")
st.divider()

uploaded_file = st.file_uploader("Upload a clothing image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Show preview
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Outfit", use_container_width=True)

    # Save temp file
    temp_path = "temp.jpg"
    image = image.convert("RGB")
    image.save(temp_path)

    # Detect color
    color = detect_dominant_color(temp_path)

    # Predict outfit style
    labels = [
    # General styles
    "a casual outfit", "a formal outfit", "a streetwear outfit", "a party outfit",
    "a sporty outfit", "a summer outfit", "a winter outfit",

    # Upper wear
    "a t-shirt", "a crop top", "a hoodie", "a sweatshirt", "a jacket", "a shirt", "a blouse",

    # Lower wear
    "a pair of jeans", "a denim skirt", "a long skirt", "a mini skirt", "a pair of trousers", "a pair of shorts",

    # Full-body
    "a dress", "a jumpsuit",

    # Footwear
    "a pair of sneakers", "a pair of heels", "a pair of boots", "a pair of sandals"
]

    text_tokens = tokenizer(labels)

    with torch.no_grad():
        image_tensor = preprocess(image).unsqueeze(0)
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text_tokens)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        sims = (image_features @ text_features.T).squeeze(0)
        best_label = labels[sims.argmax().item()]

    st.success(f"👕 Detected: {best_label} ({color})")

    # Extract embedding for recommendation engine
    embedding = image_features.squeeze(0).tolist()

    # --- Store metadata + embedding in Supabase ---
    data = {
        "item_name": uploaded_file.name,
        "style": best_label,
        "color": color,
        "embedding": embedding
    }

    try:
        supabase.table("items").insert(data).execute()
        st.info("✅ Saved to wardrobe database.")
    except Exception as e:
        st.error(f"Database error: {e}")

# --- Wardrobe Viewer Section ---
st.markdown("---")
st.header("👗 Your Wardrobe Collection")

try:
    response = supabase.table("items").select("*").execute()

    if response.data:
        cols = st.columns(3)
        for idx, item in enumerate(response.data):
            with cols[idx % 3]:
                st.markdown(
                    f"""
                    <div style="background-color:white; padding:15px; border-radius:12px; 
                    box-shadow:0 4px 10px rgba(0,0,0,0.1); margin-bottom:15px; text-align:center;">
                        <h4 style="color:#7d5fff;">{item['style'].capitalize()}</h4>
                        <p><b>Color:</b> {item['color'].capitalize()}</p>
                        <p style="font-size:0.9em; color:gray;">{item['item_name']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    else:
        st.info("🧺 No outfits saved yet — upload your first one above!")
except Exception as e:
    st.error(f"⚠️ Could not fetch wardrobe: {e}")

# --- Recommendation Engine Section ---
st.markdown("---")
st.header("🧩 Outfit Recommendations")

def recommend_outfits(target_style="jeans", top_k=3):
    response = supabase.table("items").select("*").execute()
    items = response.data
    if not items:
        return []

    target = next((i for i in items if i["style"] == target_style), None)
    if not target:
        return []

    target_vec = np.array(target["embedding"])
    recs = []
    for item in items:
     if item["item_name"] != target["item_name"] and item.get("embedding"):
        emb = np.array(item["embedding"])
        if emb is None or len(emb) == 0:
            continue
        sim = np.dot(target_vec, emb) / (np.linalg.norm(target_vec) * np.linalg.norm(emb))
        recs.append((item, sim))


    recs.sort(key=lambda x: x[1], reverse=True)
    return [r[0] for r in recs[:top_k]]

selected_item = st.selectbox("Pick an item to match:", ["jeans", "t-shirt", "hoodie", "dress", "skirt", "shoes"])

if st.button("Suggest Outfits"):
    recs = recommend_outfits(selected_item)
    if recs:
        cols = st.columns(3)
        for idx, item in enumerate(recs):
            with cols[idx % 3]:
                st.markdown(
                    f"""
                    <div style="background-color:white; padding:15px; border-radius:12px; 
                    box-shadow:0 4px 10px rgba(0,0,0,0.1); margin-bottom:15px; text-align:center;">
                        <h4 style="color:#7d5fff;">{item['style'].capitalize()}</h4>
                        <p><b>Color:</b> {item['color'].capitalize()}</p>
                        <p style="font-size:0.9em; color:gray;">{item['item_name']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    else:
        st.warning("No similar outfits found yet — upload more items!")
