#!/usr/bin/env python
"""
Kaggle Dataset Distribution Checker
Run this in a cell within your Kaggle notebook.
"""
import os
from collections import defaultdict

def check_distribution(dataset_dir):
    print(f"Analyzing distribution for: {dataset_dir}\n")
    if not os.path.exists(dataset_dir): return

    distribution = defaultdict(lambda: {'train': 0, 'test': 0})
    for split in ['train', 'test']:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.exists(split_dir): continue
        for class_name in os.listdir(split_dir):
            class_path = os.path.join(split_dir, class_name)
            if os.path.isdir(class_path):
                distribution[class_name][split] = len([f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    print(f"{'Disease Class':<60} | {'Train':<8} | {'Test':<8} | {'Total':<8}")
    print("-" * 90)
    
    total_train = 0; total_test = 0
    for class_name in sorted(distribution.keys()):
        train_count = distribution[class_name]['train']
        test_count = distribution[class_name]['test']
        total_train += train_count
        total_test += test_count
        print(f"{class_name:<60} | {train_count:<8} | {test_count:<8} | {train_count + test_count:<8}")
        
    print("-" * 90)
    print(f"{'OVERALL TOTALS':<60} | {total_train:<8} | {total_test:<8} | {total_train + total_test:<8}")

final_dataset = '/kaggle/working/Merged_Dermnet_Skin40'
check_distribution(final_dataset)
