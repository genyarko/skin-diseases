#!/usr/bin/env python
"""
Aggressive Kaggle Dataset Cleaner
Run this in your Kaggle notebook.
"""
import os
import hashlib
import glob
from collections import defaultdict

def get_file_hash(filepath, block_size=65536):
    hasher = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            buf = f.read(block_size)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(block_size)
        return hasher.hexdigest()
    except Exception:
        return None

def aggressive_cleanup(target_dir):
    print(f"Scanning all images in {target_dir}...")
    files = glob.glob(f"{target_dir}/**/*.jpg", recursive=True) + glob.glob(f"{target_dir}/**/*.png", recursive=True)
    
    hash_map = defaultdict(list)
    for f in files:
        h = get_file_hash(f)
        if h: hash_map[h].append(f)
            
    deleted_count = 0
    for h, paths in hash_map.items():
        folders = set([os.path.dirname(p) for p in paths])
        if len(folders) > 1:
            print(f" 🚨 Contamination Detected! Deleting all {len(paths)} copies from: {folders}")
            for p in paths:
                try:
                    os.remove(p)
                    deleted_count += 1
                except Exception as e: pass
                    
    print(f"\n✅ Aggressive Cleanup Complete! Permanently deleted {deleted_count} contaminated images.")

working_dir = '/kaggle/working/Merged_Dermnet_Skin40'
if os.path.exists(working_dir):
    aggressive_cleanup(working_dir)
