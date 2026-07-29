import numpy as np
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

FILE_PATH = r"d:\2 Personal Activities\Study Folder\University\Năm Ba\Aeternus\ViSign2Voice\data\class_0001.npz"

print("=" * 60)
print("PHÂN TÍCH FILE:", os.path.basename(FILE_PATH))
print(f"File size: {os.path.getsize(FILE_PATH) / (1024**2):.1f} MB")
print("=" * 60)

data = np.load(FILE_PATH, allow_pickle=True)

print(f"\n[1] Keys in NPZ file:")
for key in data.files:
    arr = data[key]
    print(f"  - '{key}': shape={arr.shape}, dtype={arr.dtype}")

print("\n" + "=" * 60)

for key in data.files:
    arr = data[key]
    print(f"\n[DETAIL] Key: '{key}'")
    print(f"  shape  : {arr.shape}")
    print(f"  dtype  : {arr.dtype}")
    print(f"  min    : {arr.min() if arr.dtype.kind in 'fiu' else 'N/A'}")
    print(f"  max    : {arr.max() if arr.dtype.kind in 'fiu' else 'N/A'}")
    if arr.dtype.kind in 'f':
        print(f"  mean   : {arr.mean():.4f}")

    # Sample data
    print(f"  Sample data (first 5 elements):")
    if arr.ndim == 1:
        print(f"    {arr[:5]}")
    elif arr.ndim == 2:
        print(f"    {arr[:3, :5]}")
    elif arr.ndim == 3:
        print(f"    arr[0] (frame đầu tiên, 10 feature đầu): {arr[0, 0, :10]}")
        print(f"    arr[0] shape (1 sample): {arr[0].shape}")
    
    # If object array (may contain strings/labels)
    if arr.dtype == object:
        print(f"  Object array detected. Sample values:")
        flat = arr.flatten()
        for i in range(min(5, len(flat))):
            print(f"    [{i}] = {flat[i]}")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
