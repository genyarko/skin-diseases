#!/usr/bin/env python
"""
Kaggle Dataset Smart Merger (V2 - Imbalance Fixed)
Run this in a cell within your Kaggle notebook.
"""
import os
import shutil

LABEL_MAP = {
    "Acne_Vulgaris": "Acne and Rosacea Photos",
    "Perioral_Dermatitis": "Acne and Rosacea Photos",
    "Rhinophyma": "Acne and Rosacea Photos",
    "Malignant_Melanoma": "Melanoma Skin Cancer Nevi and Moles",
    "Blue_Nevus": "Melanoma Skin Cancer Nevi and Moles",
    "Compound_Nevus": "Melanoma Skin Cancer Nevi and Moles",
    "Congenital_Nevus": "Melanoma Skin Cancer Nevi and Moles",
    "Dysplastic_Nevus": "Melanoma Skin Cancer Nevi and Moles",
    "Nevus_Incipiens": "Melanoma Skin Cancer Nevi and Moles",
    "Basal_Cell_Carcinoma": "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
    "Actinic_solar_Damage(Actinic_Keratosis)": "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
    "Actinic_solar_Damage(Solar_Elastosis)": "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
    "Cutaneous_Horn": "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
    "Actinic_solar_Damage(Pigmentation)": "Light Diseases and Disorders of Pigmentation",
    "Eczema": "Eczema Photos",
    "Dyshidrosiform_Eczema": "Eczema Photos",
    "Seborrheic_Dermatitis": "Eczema Photos",
    "Stasis_Dermatitis": "Eczema Photos",
    "Steroid_Use_abusemisuse_Dermatitis": "Eczema Photos",
    "Tinea_Corporis": "Tinea Ringworm Candidiasis and other Fungal Infections",
    "Tinea_Faciale": "Tinea Ringworm Candidiasis and other Fungal Infections",
    "Tinea_Manus": "Tinea Ringworm Candidiasis and other Fungal Infections",
    "Tinea_Pedis": "Tinea Ringworm Candidiasis and other Fungal Infections",
    "Tinea_Versicolor": "Tinea Ringworm Candidiasis and other Fungal Infections",
    "Pityrosporum_Folliculitis": "Tinea Ringworm Candidiasis and other Fungal Infections",
    "Seborrheic_Keratosis": "Seborrheic Keratoses and other Benign Tumors",
    "Keratoacanthoma": "Seborrheic Keratoses and other Benign Tumors",
    "Dermatofibroma": "Seborrheic Keratoses and other Benign Tumors",
    "Epidermoid_Cyst": "Seborrheic Keratoses and other Benign Tumors",
    "Pyogenic_Granuloma": "Seborrheic Keratoses and other Benign Tumors",
    "Sebaceous_Gland_Hyperplasia": "Seborrheic Keratoses and other Benign Tumors",
    "Skin_Tag": "Seborrheic Keratoses and other Benign Tumors",
    "Psoriasis": "Psoriasis pictures Lichen Planus and related diseases",
    "Inverse_Psoriasis": "Psoriasis pictures Lichen Planus and related diseases",
    "Allergic_Contact_Dermatitis": "Poison Ivy Photos and other Contact Dermatitis",
    "Onychomycosis": "Nail Fungus and other Nail Disease",
    "Alopecia_Areata": "Hair Loss Photos Alopecia and other Hair Diseases"
}

def smart_merge(skin40_base, dermnet_base, target_base):
    print(f"Creating merged dataset at: {target_base}")
    print("Copying base Dermnet dataset...")
    shutil.copytree(dermnet_base, target_base, dirs_exist_ok=True)
    
    print("\nMerging Skin40 images using the comprehensive LABEL_MAP...")
    for split in ['train', 'test']:
        skin40_split_dir = os.path.join(skin40_base, split)
        target_split_dir = os.path.join(target_base, split)
        
        if not os.path.exists(skin40_split_dir): continue
            
        for skin40_class in os.listdir(skin40_split_dir):
            skin40_class_path = os.path.join(skin40_split_dir, skin40_class)
            if not os.path.isdir(skin40_class_path): continue
                
            if skin40_class in LABEL_MAP:
                target_class = LABEL_MAP[skin40_class]
                target_class_path = os.path.join(target_split_dir, target_class)
                os.makedirs(target_class_path, exist_ok=True)
                
                images = [f for f in os.listdir(skin40_class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                copied_count = 0
                for img in images:
                    shutil.copy2(os.path.join(skin40_class_path, img), os.path.join(target_class_path, f"skin40_{img}"))
                    copied_count += 1
                print(f"  Mapped [{split}] {skin40_class} -> {target_class} ({copied_count} images)")
            else:
                shutil.copytree(skin40_class_path, os.path.join(target_split_dir, skin40_class), dirs_exist_ok=True)
                print(f"  Copied [{split}] {skin40_class} (No mapping, kept original name)")

skin40_dir = '/kaggle/input/datasets/dubietbay/skin40-dermnet-isic2019/Skin40'
dermnet_dir = '/kaggle/input/datasets/dubietbay/skin40-dermnet-isic2019/Dermnet'
merged_dir = '/kaggle/working/Merged_Dermnet_Skin40'

if os.path.exists(merged_dir):
    print("Deleting old merge directory...")
    shutil.rmtree(merged_dir)
smart_merge(skin40_dir, dermnet_dir, merged_dir)
print(f"\n✅ Merge Complete! {merged_dir}")
