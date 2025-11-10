# StyleGen 👗  
Personal AI Stylist using CLIP + Supabase + Streamlit  

### Day 0 Progress
- Setup Supabase project  
- Created `.env` and tested connection  
- Added `requirements.txt` and `.gitignore`  
- Project ready for Day 1 (CLIP model integration)

### Tech Stack
Python, Supabase, Streamlit, CLIP, OpenAI API

### ✅ Day 1 — CLIP Model Setup 

- Installed `open_clip_torch` for zero-dependency CLIP usage.
- Tested image vs. text similarity with outfit images.
- Confirmed working inference and label detection.
- Next up → Day 2: Streamlit upload UI + Supabase wardrobe storage.

✅ Day 2 — Streamlit Upload UI + Auto Tagging + Supabase Storage  
Built interactive Streamlit interface for users to upload clothing images.  
Integrated color detection (`utils/color_detect.py`) to extract dominant color.  
Used CLIP to auto-tag each item by style, color, and clothing category.  
Stored metadata and embeddings in Supabase (`items` table).  
Next up → Day 3: UI revamp + AI Wardrobe Dashboard.


✅ Day 3 — Improved CLIP Prompts + Wardrobe Dashboard UI  
Enhanced prompt list for precise tagging (denim, hoodie, streetwear, etc.).  
Redesigned Streamlit UI with soft theme and responsive layout.  
Added wardrobe grid dashboard that fetches and displays uploaded items from Supabase.  
Improved color detection + removed deprecated warnings.  
Next up → Day 4: AI Outfit Recommendation Engine (using embeddings + style rules).

✅ Day 4 — AI Outfit Recommendation Engine  
Extracted and stored CLIP embeddings in Supabase.  
Implemented cosine similarity–based outfit matching system.  
Fixed embedding errors and improved fashion prompts for better accuracy.  
Added recommendation UI where users can select an item and get matching outfit suggestions.  




