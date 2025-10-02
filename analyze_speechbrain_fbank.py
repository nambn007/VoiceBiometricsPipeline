#!/usr/bin/env python3
"""
Analyze SpeechBrain Fbank processing step-by-step
to understand exact implementation details.
"""

import torch
import numpy as np
from speechbrain.processing.features import STFT, Filterbank, spectral_magnitude

# Create test input (1 second, same seed as C++)
torch.manual_seed(42)
np.random.seed(42)
waveform = (torch.randn(16000) * 0.1).unsqueeze(0)  # [1, 16000]

print("=== SpeechBrain Fbank Processing Analysis ===\n")
print(f"Input shape: {waveform.shape}")
print(f"Input stats: mean={waveform.mean():.6f}, std={waveform.std():.6f}")
print(f"Input range: [{waveform.min():.6f}, {waveform.max():.6f}]\n")

# Step 1: STFT
print("Step 1: STFT")
print("-" * 50)
stft = STFT(sample_rate=16000, n_fft=400, win_length=25, hop_length=10)

print(f"Parameters:")
print(f"  sample_rate: {stft.sample_rate}")
print(f"  n_fft: {stft.n_fft}")
print(f"  win_length: {stft.win_length} samples (from {25}ms)")
print(f"  hop_length: {stft.hop_length} samples (from {10}ms)")
print(f"  center: {stft.center}")
print(f"  pad_mode: {stft.pad_mode}")
print(f"  normalized_stft: {stft.normalized_stft}")

stft_out = stft(waveform)
print(f"\nSTFT output shape: {stft_out.shape}")  # [batch, frames, freqs, 2]
print(f"STFT output dtype: {stft_out.dtype}")

# Check window
window = stft.window.numpy()
print(f"\nWindow (first 10): {window[:10]}")
print(f"Window (last 10): {window[-10:]}")
print(f"Window sum: {window.sum():.2f}")

# Step 2: Spectral Magnitude
print(f"\nStep 2: Spectral Magnitude")
print("-" * 50)
mag = spectral_magnitude(stft_out, power=1, log=False)  # power=1 for power spectrum
print(f"Magnitude shape: {mag.shape}")  # [batch, frames, freqs]
print(f"Magnitude dtype: {mag.dtype}")
print(f"Magnitude stats: mean={mag.mean():.6f}, std={mag.std():.6f}")
print(f"Magnitude range: [{mag.min():.6f}, {mag.max():.6f}]")

# Show first frame
mag_frame0 = mag[0, 0, :].numpy()
print(f"\nFirst frame magnitude (first 10 bins): {mag_frame0[:10]}")

# Step 3: Filterbank
print(f"\nStep 3: Mel Filterbank")
print("-" * 50)
fbank = Filterbank(
    sample_rate=16000,
    n_fft=400,
    n_mels=80,
    f_min=0,
    f_max=8000,
    log_mel=True,  # Apply log
    power_spectrogram=2  # Default
)

print(f"Parameters:")
print(f"  n_mels: {fbank.n_mels}")
print(f"  f_min: {fbank.f_min} Hz")
print(f"  f_max: {fbank.f_max} Hz")
print(f"  log_mel: {fbank.log_mel}")
print(f"  power_spectrogram: {fbank.power_spectrogram}")
print(f"  amin: {fbank.amin}")

fbanks = fbank(mag)
print(f"\nFbank output shape: {fbanks.shape}")  # [batch, frames, n_mels]
print(f"Fbank dtype: {fbanks.dtype}")
print(f"Fbank stats: mean={fbanks.mean():.6f}, std={fbanks.std():.6f}")
print(f"Fbank range: [{fbanks.min():.6f}, {fbanks.max():.6f}]")

# Show first frame
fbank_frame0 = fbanks[0, 0, :].numpy()
print(f"\nFirst frame fbank (first 10 mels): {fbank_frame0[:10]}")

# Step 4: Mean-Var Norm (from SpeechBrain model)
print(f"\nStep 4: Mean-Var Norm")
print("-" * 50)
from speechbrain.inference.speaker import EncoderClassifier

model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="python/pretrained_models/spkrec-ecapa-voxceleb"
)

# Get mean_var_norm module
mean_var_norm = model.mods.mean_var_norm
print(f"norm_type: {mean_var_norm.norm_type}")
print(f"mean_norm: {mean_var_norm.mean_norm}")
print(f"std_norm: {mean_var_norm.std_norm}")

# Apply normalization
wav_lens = torch.ones(1)
fbanks_normalized = mean_var_norm(fbanks, wav_lens)
print(f"\nNormalized output shape: {fbanks_normalized.shape}")
print(f"Normalized dtype: {fbanks_normalized.dtype}")
print(f"Normalized stats: mean={fbanks_normalized.mean():.6f}, std={fbanks_normalized.std():.6f}")
print(f"Normalized range: [{fbanks_normalized.min():.6f}, {fbanks_normalized.max():.6f}]")

# Show first frame
norm_frame0 = fbanks_normalized[0, 0, :].numpy()
print(f"\nFirst frame normalized (first 10 mels): {norm_frame0[:10]}")

# Save all intermediate results
print(f"\n=== Saving intermediate results ===")
np.save("debug_stft.npy", stft_out.squeeze(0).numpy())
np.save("debug_magnitude.npy", mag.squeeze(0).numpy())
np.save("debug_fbank.npy", fbanks.squeeze(0).numpy())
np.save("debug_normalized.npy", fbanks_normalized.squeeze(0).numpy())
np.save("debug_window.npy", window)

# Also save mel filterbank weights
mel_fb = fbank.fbank_matrix.detach().cpu().numpy()
np.save("debug_mel_filterbank.npy", mel_fb)
print(f"Mel filterbank shape: {mel_fb.shape}")

print(f"\n✅ Analysis complete!")
print(f"Key findings:")
print(f"  - Window: Hamming, size={stft.win_length}")
print(f"  - STFT: center=True with constant padding")
print(f"  - Power: magnitude^2 (power=1 in spectral_magnitude)")
print(f"  - Mel: log applied after filterbank")
print(f"  - Norm: sentence-level mean subtraction only (std_norm=False)")

