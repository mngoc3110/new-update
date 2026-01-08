import numpy as np
from collections import Counter
import os

def analyze_annotation_file(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return

    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                # Label is the 3rd part
                labels.append(int(parts[2]))

    counts = Counter(labels)
    
    print(f"\n--- Analysis for: {os.path.basename(file_path)} ---")
    print(f"Total samples: {len(labels)}")
    
    if not counts:
        print("No labels found.")
        return

    min_label = min(counts.keys())
    max_label = max(counts.keys())
    print(f"Label range: {min_label} to {max_label}")
    
    print("\nClass Counts:")
    sorted_counts = sorted(counts.items())
    for label, count in sorted_counts:
        print(f"  Class {label}: {count} samples")
        
    # Assuming 1-based labels for binary split
    if 1 in counts:
        neutral_count = counts.get(1, 0)
        non_neutral_count = sum(c for l, c in counts.items() if l > 1)
        print("\nBinary Split (assuming class 1 is 'Neutral'):")
        print(f"  Neutral (Class 1) samples: {neutral_count}")
        print(f"  Non-Neutral (Classes >1) samples: {non_neutral_count}")
    print("------------------------------------")


if __name__ == "__main__":
    annot_dir = "./RAER/annotation"
    train_file = os.path.join(annot_dir, "train_80.txt")
    val_file = os.path.join(annot_dir, "val_20.txt")
    
    analyze_annotation_file(train_file)
    analyze_annotation_file(val_file)