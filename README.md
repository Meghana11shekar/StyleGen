# 🧥 StyleGen — Personal AI Stylist

An AI-powered smart wardrobe and outfit recommendation system using **FashionCLIP + Streamlit + Supabase**.

---

## 🚀 Tech Stack
- Python  
- Streamlit  
- Supabase (Database + Storage)  
- FashionCLIP / CLIP  
- Pillow  
- NumPy  

---

# 📅 Progress Log

## ✅ Day 0 — Project Setup
- Created Supabase project
- Added `.env`, `.gitignore` and `requirements.txt`
- Verified Supabase connectivity from Python
- Project ready for CLIP integration

---

## ✅ Day 1 — CLIP Model Setup
- Installed `open_clip_torch` and FashionCLIP
- Tested image–text similarity
- Verified CLIP inference working
- First style detection successful

➡️ Next: Build Upload UI + Supabase wardrobe storage

---

## ✅ Day 2 — Upload UI + Auto Tagging
- Added Streamlit uploader
- Color detection using `utils/colour_detect.py`
- Auto-tag outfit style, color, category using FashionCLIP
- Stored item metadata + embeddings in Supabase

➡️ Next: Wardrobe Dashboard UI

---

## ✅ Day 3 — Wardrobe Dashboard + Better Tags
- Added wardrobe grid view
- Improved prompt list (hoodie, denim, streetwear, dress, etc.)
- Soft pastel Streamlit UI theme
- Improved color detection + removed warnings

➡️ Next: Outfit matching recommendation engine

---

## ✅ Day 4 — Recommendation Engine
- Stored FashionCLIP embeddings in Supabase
- Cosine similarity based outfit matching
- UI for selecting clothing and generating combos
- Faster and more accurate styling

➡️ Next: Improve prediction model

---

## ✅ Day 5 — FashionCLIP Upgrade
- Switched to `fashion-clip` for better fashion understanding
- Better tagging accuracy
- Wardrobe loading stable
- Detects styles like crop top, hoodie, denim, dress, etc.

➡️ Next: Multi-page App & Architecture

---

## ⭐ Day 6 — Multi-Page UI + Architecture Update
- Converted app into a clean **multi-page Streamlit application**
- New folder structure:
streamlit_app.py
pages/
1_Upload.py
2_Wardrobe.py
3_Recommend.py
utils/
core.py

yaml
Copy code
- Shared model + Supabase connection using `utils/core.py`
- No duplication of code anymore
- Recommendation, Upload and Wardrobe independent pages

➡️ Next: Store images in Supabase Storage + Pinterest style UI

---

# 📌 Roadmap (upcoming)
- Show images in wardrobe and recommendation
- Pinterest style grid layout
- Shop-the-look integration
- User profiles + favorites
- Better styling & multi-tag matches

---

# 🧵 Run locally
streamlit run streamlit_app.py

---

# ⭐ Author
Made by Meghana ✨
