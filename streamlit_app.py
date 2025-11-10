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
    # General style
    "casual outfit", "formal outfit", "party outfit", "sporty outfit", 
    "winter wear", "summer wear", "street style outfit",

    # Category-based
    "jeans", "denim pants", "t-shirt", "hoodie", "dress", "jacket", 
    "sweatshirt", "trousers", "skirt", "sneakers", "heels", "boots"
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

    # --- Store metadata in Supabase ---
    data = {
        "item_name": uploaded_file.name,
        "style": best_label,
        "color": color
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
        cols = st.columns(3)  # 3 cards per row
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
