"""
Demo: How ECAPA-TDNN processes waveform into mel-spectrogram features

This script demonstrates in detail the processing pipeline from raw audio waveform
to the mel-spectrogram features used by ECAPA-TDNN.
"""

import torch
import numpy as np
from speechbrain.inference.speaker import EncoderClassifier

def demo_compute_features():
    print("=== DEMO: ECAPA-TDNN COMPUTE_FEATURES ===\n")
    
    # Load the ECAPA-TDNN model
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb"
    )
    
    compute_features = model.mods.compute_features
    
    print("1. MODEL INFO:")
    print(f"   - Module type: {type(compute_features).__name__}")
    print(f"   - STFT n_fft: {compute_features.compute_STFT.n_fft}")
    print(f"   - STFT hop_length: {compute_features.compute_STFT.hop_length}")
    print(f"   - Mel filters: {compute_features.compute_fbanks.n_mels}")
    print(f"   - Frequency range: {compute_features.compute_fbanks.f_min}-{compute_features.compute_fbanks.f_max} Hz")
    
    print("\n2. PROCESSING PIPELINE:")
    
    # Create a dummy waveform (2 seconds of audio)
    sample_rate = 16000
    duration = 2.0
    waveform = torch.randn(1, int(sample_rate * duration))
    
    print(f"   Input: waveform {waveform.shape} ({duration}s audio)")
    
    # Step 1: STFT
    stft_module = compute_features.compute_STFT
    stft_output = stft_module(waveform)
    print(f"   STFT: {stft_output.shape} (complex spectrogram)")
    
    # Step 2: Mel-filterbank
    filterbank_module = compute_features.compute_fbanks
    mel_output = filterbank_module(stft_output)
    print(f"   Mel-filterbank: {mel_output.shape} (mel-spectrogram)")
    
    # Step 3: Log and normalization
    final_features = compute_features(waveform)
    print(f"   Final: {final_features.shape} (log mel-spectrogram)")
    
    print("\n3. TIME CALCULATION:")
    hop_length = compute_features.compute_STFT.hop_length
    time_per_frame = hop_length / sample_rate
    total_frames = final_features.shape[1]
    total_time = total_frames * time_per_frame
    
    print(f"   - Hop length: {hop_length} samples")
    print(f"   - Time per frame: {time_per_frame:.4f} seconds")
    print(f"   - Total frames: {total_frames}")
    print(f"   - Total time: {total_time:.2f} seconds")
    
    print("\n4. FEATURE STATISTICS:")
    print(f"   - Min value: {final_features.min():.4f}")
    print(f"   - Max value: {final_features.max():.4f}")
    print(f"   - Mean: {final_features.mean():.4f}")
    print(f"   - Std: {final_features.std():.4f}")
    
    print("\n5. USAGE NOTES:")
    print("   - Input: waveform (batch, time)")
    print("   - Output: mel-spectrogram (batch, time_frames, mel_features)")
    print("   - ECAPA-TDNN expects mel-spectrogram as input, not raw audio")
    
    return final_features

if __name__ == "__main__":
    features = demo_compute_features()
    print(f"\n✅ Demo complete! Output shape: {features.shape}")
