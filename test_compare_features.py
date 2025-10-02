#!/usr/bin/env python3
"""
Compare C++ AudioFeatureExtractor output with Python SpeechBrain processing.
"""

import torch
import numpy as np
from speechbrain.inference.speaker import EncoderClassifier

# Load SpeechBrain model
model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="python/pretrained_models/spkrec-ecapa-voxceleb"
)

# Create test waveform (same seed as C++)
torch.manual_seed(42)
np.random.seed(42)
waveform = torch.randn(16000) * 0.1  # 1 second, same as C++ test

print("=== Python SpeechBrain Processing ===\n")
print(f"Input: {waveform.shape} samples ({waveform.shape[0]/16000:.1f} seconds)")

# Process with SpeechBrain (matches the two lines in encode_batch)
# Line 110: feats = self.mods.compute_features(wavs)
feats = model.mods.compute_features(waveform.unsqueeze(0))
print(f"After compute_features: {feats.shape}")

# Line 111: feats = self.mods.mean_var_norm(feats, wav_lens)
wav_lens = torch.tensor([1.0])
feats_normalized = model.mods.mean_var_norm(feats, wav_lens)
print(f"After mean_var_norm: {feats_normalized.shape}")

# Compute statistics
feats_np = feats_normalized.squeeze(0).cpu().numpy()
print(f"\nStatistics:")
print(f"  Mean: {feats_np.mean():.6f}")
print(f"  Std:  {feats_np.std():.6f}")
print(f"  Min:  {feats_np.min():.6f}")
print(f"  Max:  {feats_np.max():.6f}")

print(f"\nFirst frame (first 5 values): {feats_np[0, :5]}")

# Save waveform for C++ comparison
print(f"\nSaving waveform to test_waveform.bin...")
waveform_np = waveform.numpy().astype(np.float32)
with open("test_waveform.bin", "wb") as f:
    f.write(waveform_np.tobytes())

# Save features for comparison
print(f"Saving features to test_features_python.bin...")
with open("test_features_python.bin", "wb") as f:
    f.write(feats_np.astype(np.float32).tobytes())

print(f"\n✅ Python processing complete!")
print(f"   Output shape: [{feats_np.shape[0]}, {feats_np.shape[1]}]")

