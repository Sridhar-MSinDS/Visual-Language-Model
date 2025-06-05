---
title: VLM
emoji: 📉
colorFrom: red
colorTo: pink
sdk: streamlit
app_file: app.py
pinned: false
sdk_version: 1.45.1
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference


# CLIP-based Visual Language Search

A Streamlit app that allows users to upload images with descriptions and later search for them using text queries powered by OpenAI's CLIP model.

## Features
- Upload image + text and store it with CLIP embedding
- Text-based search for the most relevant image
- Fast search using FAISS

## Usage
1. Upload image + description
2. Search by entering a query
3. The app finds and displays the most semantically similar image
