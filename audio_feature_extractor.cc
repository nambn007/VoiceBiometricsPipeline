#include "audio_feature_extractor.h"
#include "mel_extractor.h"
#include <iostream>
#include <stdexcept>
#include <cmath>

AudioFeatureExtractor::AudioFeatureExtractor(
    int sample_rate,
    int n_fft,
    int hop_length,
    int n_mels,
    float f_min,
    float f_max
) : sample_rate_(sample_rate) {
    // Initialize mel extractor with kaldi-native-fbank
    mel_extractor_ = std::make_unique<MelExtractor>(
        sample_rate,
        n_fft,
        hop_length,
        n_mels,
        f_min,
        f_max
    );
    
    std::cout << "✅ AudioFeatureExtractor initialized:" << std::endl;
    std::cout << "   Sample rate: " << sample_rate << " Hz" << std::endl;
    std::cout << "   n_fft: " << n_fft << ", hop_length: " << hop_length << std::endl;
    std::cout << "   n_mels: " << n_mels << ", f_min: " << f_min << ", f_max: " << f_max << std::endl;
}

AudioFeatureExtractor::~AudioFeatureExtractor() = default;

std::vector<std::vector<float>> AudioFeatureExtractor::compute_features(
    const std::vector<float>& waveform
) const {
    if (waveform.empty()) {
        std::cerr << "⚠️  Empty waveform in compute_features" << std::endl;
        return {};
    }
    
    // Use MelExtractor (kaldi-native-fbank) to compute log-mel features
    // This matches SpeechBrain's compute_features (Fbank class)
    auto features = mel_extractor_->extract(waveform);
    
    if (features.empty()) {
        std::cerr << "⚠️  Mel extraction failed" << std::endl;
        return {};
    }
    
    return features;
}

std::vector<std::vector<float>> AudioFeatureExtractor::mean_var_norm(
    const std::vector<std::vector<float>>& features
) const {
    /*
     * Implements SpeechBrain's InputNormalization with:
     * - norm_type="sentence" 
     * - std_norm=False
     * 
     * This computes:
     * 1. Mean across time dimension (dim=0 in torch, which is the frame axis)
     * 2. Subtract mean from each frame
     * 
     * Python code equivalent:
     *   current_mean = torch.mean(x, dim=0)  # [n_features]
     *   current_std = torch.tensor([1.0])    # No std normalization
     *   out = (x - current_mean) / current_std
     */
    
    if (features.empty() || features[0].empty()) {
        return features;
    }
    
    const size_t n_frames = features.size();
    const size_t n_features = features[0].size();
    
    // Step 1: Compute mean for each feature dimension across all frames
    // This corresponds to torch.mean(x, dim=0)
    std::vector<float> mean(n_features, 0.0f);
    
    for (const auto& frame : features) {
        for (size_t i = 0; i < n_features; ++i) {
            mean[i] += frame[i];
        }
    }
    
    // Divide by number of frames to get mean
    const float inv_n_frames = 1.0f / static_cast<float>(n_frames);
    for (size_t i = 0; i < n_features; ++i) {
        mean[i] *= inv_n_frames;
    }
    
    // Step 2: Subtract mean from each frame (with std=1.0, so no division)
    // This corresponds to (x - current_mean) / 1.0
    std::vector<std::vector<float>> normalized = features;  // Copy
    
    for (auto& frame : normalized) {
        for (size_t i = 0; i < n_features; ++i) {
            frame[i] -= mean[i];
        }
    }
    
    return normalized;
}

std::vector<std::vector<float>> AudioFeatureExtractor::process(
    const std::vector<float>& waveform
) const {
    // Step 1: Extract mel-spectrogram features
    auto features = compute_features(waveform);
    
    if (features.empty()) {
        return {};
    }
    
    // Step 2: Apply mean-variance normalization (sentence-level)
    auto normalized = mean_var_norm(features);
    
    return normalized;
}

std::vector<std::vector<float>> AudioFeatureExtractor::process_chunks(
    const std::vector<std::vector<float>>& audio_chunks
) const {
    if (audio_chunks.empty()) {
        std::cerr << "⚠️  No audio chunks to process" << std::endl;
        return {};
    }
    
    std::vector<std::vector<float>> all_features;
    
    // Process each chunk independently
    for (size_t i = 0; i < audio_chunks.size(); ++i) {
        const auto& chunk = audio_chunks[i];
        
        if (chunk.empty()) {
            std::cerr << "⚠️  Skipping empty chunk " << i << std::endl;
            continue;
        }
        
        // Extract and normalize features for this chunk
        auto chunk_features = process(chunk);
        
        if (chunk_features.empty()) {
            std::cerr << "⚠️  Failed to extract features for chunk " << i << std::endl;
            continue;
        }
        
        // Concatenate features from all chunks
        all_features.insert(
            all_features.end(),
            chunk_features.begin(),
            chunk_features.end()
        );
    }
    
    if (all_features.empty()) {
        std::cerr << "❌ No features extracted from any chunks" << std::endl;
        return {};
    }
    
    std::cout << "✅ Processed " << audio_chunks.size() << " chunks -> " 
              << all_features.size() << " frames x " 
              << all_features[0].size() << " features" << std::endl;
    
    return all_features;
}

