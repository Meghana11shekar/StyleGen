import streamlit as st
import numpy as np
from utils.core import supabase, encode_text, labels

st.markdown("## 🧠 Outfit Recommender")

choice = st.selectbox("Pick item type:", labels)

def recommend(style):
    items = supabase.table("items").select("*").execute().data

    t = encode_text(style)
    ranked = []

    for item in items:
        emb = np.array(item["embedding"])
        sim = float(np.dot(t, emb) /
                    (np.linalg.norm(t) * np.linalg.norm(emb)))
        ranked.append((sim, item))

    ranked.sort(key=lambda x: x[0], reverse=True)

    return ranked[:3]


if st.button("Suggest"):
    results = recommend(choice)

    if not results:
        st.warning("Upload more outfits first!")
    else:
        st.success("Top matches:")

        for sim, item in results:
            st.write(f"{item['style']} - {item['color']}")
