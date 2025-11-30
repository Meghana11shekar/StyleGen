import streamlit as st
import os
from supabase import create_client
from dotenv import load_dotenv

# --- minimal pastel for this page ---
st.markdown("""
<style>
:root {
    --bg-color: #f9f2ff;
    --card-bg: #ffffff;
}
[data-testid="stAppViewContainer"] {
    background-color: var(--bg-color) !important;
}
.card {
    background: var(--card-bg);
    padding: 18px;
    border-radius: 15px;
    margin: 10px;
    box-shadow: 0 6px 16px rgba(0,0,0,0.12);
    text-align:center;
    border: 2px solid #e8d6ff;
    font-family: "Poppins", sans-serif;
}
</style>
""", unsafe_allow_html=True)

# --- Supabase (read only) ---
load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

st.markdown("## 📦 Your Wardrobe")

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
        st.info("🧺 No outfits saved yet — upload one from the main page!")
except Exception as e:
    st.error(f"⚠️ Error loading wardrobe: {e}")
