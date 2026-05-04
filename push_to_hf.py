#!/usr/bin/env python
"""
Hugging Face Model Uploader
Run this on your AMD server to safely push your trained weights to Hugging Face.
"""
from huggingface_hub import HfApi, login
import os
import sys

# ==============================================================================
# CONFIGURATION - CHANGE THESE!
# ==============================================================================
# 1. Get a Write token from huggingface.co/settings/tokens
HF_TOKEN = "YOUR_HF_TOKEN_HERE" 

# 2. Set your Hugging Face username and the name of the model repo you want to create
# Example: "merolavtechnology/dinov2-skin-disease-80pct"
REPO_ID = "YOUR_USERNAME/YOUR_MODEL_NAME" 

if HF_TOKEN == "YOUR_HF_TOKEN_HERE":
    print("🚨 Error: Please edit this script and paste your real HF_TOKEN.")
    sys.exit(1)

print("Logging into Hugging Face...")
login(token=HF_TOKEN)

api = HfApi()

print(f"Creating repository '{REPO_ID}' (if it doesn't already exist)...")
api.create_repo(repo_id=REPO_ID, repo_type="model", private=False, exist_ok=True)

print("Uploading 'best_model.pt' to Hugging Face. This might take a minute...")
try:
    api.upload_file(
        path_or_fileobj="best_model.pt",
        path_in_repo="best_model.pt",
        repo_id=REPO_ID,
        repo_type="model"
    )
    print(f"✅ Success! Your model is safely stored at: https://huggingface.co/{REPO_ID}")
except Exception as e:
    print(f"🚨 Upload failed: {e}")
