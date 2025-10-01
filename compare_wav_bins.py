#!/usr/bin/env python3
"""Compare WAV binary dumps from C++ and Python implementations."""

import numpy as np
import struct

def load_binary_file(filepath):
    """Load binary file and parse as float32 array."""
    try:
        with open(filepath, 'rb') as f:
            data = f.read()
        
        # Parse as float32 (4 bytes per float)
        num_floats = len(data) // 4
        floats = struct.unpack(f'{num_floats}f', data)
        arr = np.array(floats, dtype=np.float32)
        
        return arr
    except Exception as e:
        print(f"❌ Error loading {filepath}: {e}")
        return None

def compare_arrays(arr1, arr2, name1, name2):
    """Compare two arrays and print statistics."""
    print(f"\n{'='*80}")
    print(f"Comparing: {name1} vs {name2}")
    print(f"{'='*80}")
    
    if arr1 is None or arr2 is None:
        print("⚠️  One or both arrays failed to load")
        return
    
    print(f"\n📊 Shape Information:")
    print(f"  {name1}: {arr1.shape} ({len(arr1)} samples)")
    print(f"  {name2}: {arr2.shape} ({len(arr2)} samples)")
    
    if arr1.shape != arr2.shape:
        print(f"\n❌ Shape mismatch!")
        print(f"  Length difference: {abs(len(arr1) - len(arr2))} samples")
        
        # Compare overlapping portion
        min_len = min(len(arr1), len(arr2))
        if min_len > 0:
            print(f"\n  Comparing first {min_len} samples...")
            arr1_truncated = arr1[:min_len]
            arr2_truncated = arr2[:min_len]
            compare_values(arr1_truncated, arr2_truncated, name1, name2)
        return
    
    compare_values(arr1, arr2, name1, name2)

def compare_values(arr1, arr2, name1, name2):
    """Compare values of two arrays with same shape."""
    print(f"\n📈 Statistics:")
    print(f"  {name1}:")
    print(f"    Mean: {arr1.mean():.6f}")
    print(f"    Std:  {arr1.std():.6f}")
    print(f"    Min:  {arr1.min():.6f}")
    print(f"    Max:  {arr1.max():.6f}")
    
    print(f"\n  {name2}:")
    print(f"    Mean: {arr2.mean():.6f}")
    print(f"    Std:  {arr2.std():.6f}")
    print(f"    Min:  {arr2.min():.6f}")
    print(f"    Max:  {arr2.max():.6f}")
    
    # Compute differences
    diff = arr1 - arr2
    abs_diff = np.abs(diff)
    
    print(f"\n🔍 Difference Analysis:")
    print(f"  Mean absolute difference: {abs_diff.mean():.8f}")
    print(f"  Max absolute difference:  {abs_diff.max():.8f}")
    print(f"  Std of differences:       {diff.std():.8f}")
    
    # Check if arrays are identical
    if np.array_equal(arr1, arr2):
        print(f"\n✅ Arrays are IDENTICAL (exact match)")
    elif np.allclose(arr1, arr2, rtol=1e-5, atol=1e-8):
        print(f"\n✅ Arrays are VERY CLOSE (within tolerance rtol=1e-5, atol=1e-8)")
    elif np.allclose(arr1, arr2, rtol=1e-3, atol=1e-6):
        print(f"\n⚠️  Arrays are CLOSE (within tolerance rtol=1e-3, atol=1e-6)")
    else:
        print(f"\n❌ Arrays are DIFFERENT")
        
        # Find indices with largest differences
        top_diff_indices = np.argsort(abs_diff)[-10:][::-1]
        print(f"\n  Top 10 largest differences:")
        for i, idx in enumerate(top_diff_indices, 1):
            print(f"    {i}. Index {idx}: {arr1[idx]:.6f} vs {arr2[idx]:.6f} (diff: {diff[idx]:.6f})")
    
    # Sample comparison (first 10 values)
    print(f"\n📝 Sample values (first 10):")
    print(f"  Index | {name1:>12s} | {name2:>12s} | Difference")
    print(f"  {'-'*55}")
    for i in range(min(10, len(arr1))):
        print(f"  {i:5d} | {arr1[i]:12.6f} | {arr2[i]:12.6f} | {diff[i]:12.8f}")

if __name__ == "__main__":
    # Load C++ binary files
    print("\n🔧 Loading C++ binary files...")
    cpp_wav1 = load_binary_file('/u02/libs/silero-vad/cc/debug_wav1.bin')
    cpp_wav2 = load_binary_file('/u02/libs/silero-vad/cc/debug_wav2.bin')
    
    # Load Python binary files
    print("🐍 Loading Python binary files...")
    py_wav1 = load_binary_file('/u02/libs/silero-vad/cc/python/pipeline/debug_wav1.bin')
    py_wav2 = load_binary_file('/u02/libs/silero-vad/cc/python/pipeline/debug_wav2.bin')
    
    # Compare wav1: C++ vs Python
    compare_arrays(cpp_wav1, py_wav1, "C++ wav1", "Python wav1")
    
    # Compare wav2: C++ vs Python
    compare_arrays(cpp_wav2, py_wav2, "C++ wav2", "Python wav2")
    
    print(f"\n{'='*80}")
    print("✅ Comparison complete!")
    print(f"{'='*80}\n")
