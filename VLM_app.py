import streamlit as st
import clip
import torch
from PIL import Image
import numpy as np
import os
import uuid
import faiss
import pickle

# App config
st.set_page_config(page_title="CLIP Image Search")
st.title("🔍 Visual Language Search with CLIP")

# Constants
DB_PATH = "clip_db"
INDEX_FILE = os.path.join(DB_PATH, "index.faiss")
METADATA_FILE = os.path.join(DB_PATH, "metadata.pkl")
MAX_CHARS = 300
IMAGE_SIZE = (512, 512)

# Ensure storage directory exists
os.makedirs(DB_PATH, exist_ok=True)

# Cache model and preprocessing
@st.cache_resource
def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

# Cache index and metadata
@st.cache_resource
def load_index_and_metadata():
    if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
        index = faiss.read_index(INDEX_FILE)
        with open(METADATA_FILE, "rb") as f:
            metadata = pickle.load(f)
    else:
        index = faiss.IndexFlatIP(512)
        metadata = []
    return index, metadata

# Save updated index and metadata
def save_index_and_metadata(index, metadata):
    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(metadata, f)

# Load everything
model, preprocess, device = load_clip_model()
index, metadata = load_index_and_metadata()

# Upload section
st.subheader("📤 Upload Image & (Optional) Description")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
description = st.text_area("Enter an optional description for the image")

if uploaded_file:
    if st.button("Submit"):
        # Save and resize image
        img = Image.open(uploaded_file).convert("RGB")
        img = img.resize(IMAGE_SIZE)
        image_id = str(uuid.uuid4())
        image_path = os.path.join(DB_PATH, f"{image_id}.jpg")
        img.save(image_path)

        # Use fallback description if empty
        if not description.strip():
            description = "An uploaded image"
        safe_description = description[:MAX_CHARS]

        # Generate embedding
        image_input = preprocess(img).unsqueeze(0).to(device)
        text_input = clip.tokenize([safe_description]).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_input)
            embedding = (image_features + text_features) / 2
            embedding /= embedding.norm(dim=-1, keepdim=True)

        # Add to index
        index.add(embedding.cpu().numpy())
        metadata.append({
            "path": image_path,
            "description": description
        })

        # Save everything
        save_index_and_metadata(index, metadata)

        st.success("✅ Image and description saved!")
        st.image(image_path, caption=description)

# Query section
st.subheader("🔎 Query Image by Description")
query = st.text_input("Enter your query")

if st.button("Search") and query:
    safe_query = query[:MAX_CHARS]
    with torch.no_grad():
        query_token = clip.tokenize([safe_query]).to(device)
        query_features = model.encode_text(query_token)
        query_features /= query_features.norm(dim=-1, keepdim=True)

    # Search
    if index.ntotal > 0:
        D, I = index.search(query_features.cpu().numpy(), k=1)
        match = metadata[I[0][0]]
        st.image(match["path"], caption=match["description"])
        with open(match["path"], "rb") as file:
            st.download_button(
                label="⬇️ Download Image",
                data=file,
                file_name=os.path.basename(match["path"]),
                mime="image/jpeg"
            )
    else:
        st.warning("⚠️ No images stored yet.")
