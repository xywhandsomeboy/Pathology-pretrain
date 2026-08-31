#!/usr/bin/env python3
"""
Split TCGA dataset by FIGO_major with 7:3 train/val ratio.
Ensure samples with _0 and _1 suffixes (same base name) stay together.
"""

import csv
import os
from collections import defaultdict
import random

def extract_base_name(file_name):
    """Extract base name by removing _0 or _1 suffix."""
    if file_name.endswith('_0') or file_name.endswith('_1'):
        return file_name[:-2]
    return file_name

def read_csv_file(file_path):
    """Read CSV file and return header and rows."""
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    return header, rows

def group_samples_by_base_name(rows, file_name_idx):
    """Group rows by base name of file_name_HE."""
    sample_groups = defaultdict(list)
    for row in rows:
        file_name = row[file_name_idx]
        base_name = extract_base_name(file_name)
        sample_groups[base_name].append(row)
    return sample_groups

def get_figo_major(row, figo_idx):
    """Get FIGO_major value from row."""
    return row[figo_idx] if figo_idx < len(row) else ''

def stratified_split(sample_groups, rows, file_name_idx, figo_idx, train_ratio=0.7, random_seed=42):
    """
    Perform stratified split by FIGO_major.
    Groups samples with same base name together.
    """
    random.seed(random_seed)
    
    # Group samples by FIGO_major
    figo_groups = defaultdict(list)
    for base_name, group_rows in sample_groups.items():
        # Get FIGO_major from first row (all rows in group should have same FIGO_major)
        figo = get_figo_major(group_rows[0], figo_idx)
        figo_groups[figo].append((base_name, group_rows))
    
    # Split each FIGO_major group
    train_samples = []
    val_samples = []
    
    for figo, groups in figo_groups.items():
        # Shuffle groups
        random.shuffle(groups)
        
        # Calculate split point
        total_groups = len(groups)
        train_count = int(total_groups * train_ratio)
        
        # Split groups
        train_groups = groups[:train_count]
        val_groups = groups[train_count:]
        
        # Add all rows from train groups
        for base_name, group_rows in train_groups:
            train_samples.extend(group_rows)
        
        # Add all rows from val groups
        for base_name, group_rows in val_groups:
            val_samples.extend(group_rows)
    
    return train_samples, val_samples

def write_csv_file(file_path, header, rows):
    """Write CSV file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

def main():
    input_file = '/home/li_yu/Proj04_he/Code/datasets/finetune_csv/hospital-TCGA/TCGA_train_set.csv'
    output_dir = '/home/li_yu/Proj04_he/Code/datasets/finetune_csv/hospital-TCGA'
    
    train_file = os.path.join(output_dir, 'TCGA_train_split.csv')
    val_file = os.path.join(output_dir, 'TCGA_val_split.csv')
    
    print(f"Reading CSV file: {input_file}")
    header, rows = read_csv_file(input_file)
    
    # Find column indices
    file_name_idx = header.index('file_name_HE')
    figo_idx = header.index('FIGO_major')
    
    print(f"Total rows: {len(rows)}")
    print(f"file_name_HE column index: {file_name_idx}")
    print(f"FIGO_major column index: {figo_idx}")
    
    # Group samples by base name
    print("\nGrouping samples by base name...")
    sample_groups = group_samples_by_base_name(rows, file_name_idx)
    print(f"Total unique samples (base names): {len(sample_groups)}")
    
    # Count samples per FIGO_major
    figo_counts = defaultdict(int)
    for base_name, group_rows in sample_groups.items():
        figo = get_figo_major(group_rows[0], figo_idx)
        figo_counts[figo] += 1
    
    print("\nFIGO_major distribution (by samples):")
    for figo, count in sorted(figo_counts.items()):
        print(f"  {figo}: {count} samples")
    
    # Perform stratified split
    print("\nPerforming stratified split (7:3)...")
    train_rows, val_rows = stratified_split(sample_groups, rows, file_name_idx, figo_idx, train_ratio=0.7)
    
    print(f"\nSplit results:")
    print(f"  Train samples: {len(train_rows)} rows")
    print(f"  Val samples: {len(val_rows)} rows")
    
    # Verify no sample split
    train_base_names = set()
    val_base_names = set()
    
    for row in train_rows:
        base_name = extract_base_name(row[file_name_idx])
        train_base_names.add(base_name)
    
    for row in val_rows:
        base_name = extract_base_name(row[file_name_idx])
        val_base_names.add(base_name)
    
    overlap = train_base_names & val_base_names
    if overlap:
        print(f"\nWARNING: Found {len(overlap)} samples in both train and val!")
        print(f"Overlapping samples: {list(overlap)[:5]}...")
    else:
        print(f"\n✓ Verification passed: No sample appears in both train and val")
    
    # Count FIGO_major in splits
    train_figo = defaultdict(int)
    val_figo = defaultdict(int)
    
    for row in train_rows:
        figo = get_figo_major(row, figo_idx)
        train_figo[figo] += 1
    
    for row in val_rows:
        figo = get_figo_major(row, figo_idx)
        val_figo[figo] += 1
    
    print("\nFIGO_major distribution in splits (by rows):")
    all_figos = set(train_figo.keys()) | set(val_figo.keys())
    for figo in sorted(all_figos):
        train_count = train_figo.get(figo, 0)
        val_count = val_figo.get(figo, 0)
        total = train_count + val_count
        if total > 0:
            train_pct = train_count / total * 100
            print(f"  {figo}: Train={train_count} ({train_pct:.1f}%), Val={val_count} ({100-train_pct:.1f}%)")
    
    # Write output files
    print(f"\nWriting train file: {train_file}")
    write_csv_file(train_file, header, train_rows)
    
    print(f"Writing val file: {val_file}")
    write_csv_file(val_file, header, val_rows)
    
    print("\nDone!")

if __name__ == '__main__':
    main()
