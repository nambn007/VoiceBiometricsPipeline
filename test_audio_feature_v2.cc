#include "audio_feature_extractor_v2.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <random>
#include <cstring>

int main() {
    std::cout << "=== Testing AudioFeatureExtractor V2 (SpeechBrain compatible) ===" << std::endl << std::endl;
    
    // Create extractor with exact SpeechBrain parameters
    AudioFeatureExtractor extractor(
        16000,  // sample_rate
        400,    // n_fft
        25.0f,  // win_length_ms
        10.0f,  // hop_length_ms
        80,     // n_mels
        0.0f,   // f_min
        8000.0f // f_max
    );
    
    std::cout << std::endl;
    
    // Test with same waveform as Python (load from binary if exists)
    std::vector<float> waveform;
    
    std::ifstream bin_file("test_waveform.bin", std::ios::binary);
    if (bin_file.is_open()) {
        // Load from Python-generated binary
        bin_file.seekg(0, std::ios::end);
        size_t file_size = bin_file.tellg();
        bin_file.seekg(0, std::ios::beg);
        
        size_t n_samples = file_size / sizeof(float);
        waveform.resize(n_samples);
        bin_file.read(reinterpret_cast<char*>(waveform.data()), file_size);
        bin_file.close();
        
        std::cout << "✅ Loaded waveform from test_waveform.bin: " << n_samples << " samples" << std::endl;
    } else {
        // Generate test waveform (same seed as Python)
        std::cout << "⚠️  test_waveform.bin not found, generating random waveform" << std::endl;
        waveform.resize(16000);
        std::mt19937 gen(42);
        std::normal_distribution<float> dist(0.0f, 0.1f);
        for (auto& sample : waveform) {
            sample = dist(gen);
        }
    }
    
    std::cout << "Input: " << waveform.size() << " samples (" 
              << (waveform.size() / 16000.0) << " seconds)" << std::endl;
    
    // Compute input statistics
    float sum = 0.0f, sum_sq = 0.0f;
    float min_val = waveform[0], max_val = waveform[0];
    for (float val : waveform) {
        sum += val;
        sum_sq += val * val;
        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
    }
    float mean = sum / waveform.size();
    float variance = (sum_sq / waveform.size()) - (mean * mean);
    float std_dev = std::sqrt(variance);
    
    std::cout << "Input stats: mean=" << mean << ", std=" << std_dev 
              << ", range=[" << min_val << ", " << max_val << "]" << std::endl << std::endl;
    
    // Process
    std::cout << "Processing..." << std::endl;
    auto features = extractor.process(waveform);
    
    if (features.empty()) {
        std::cerr << "❌ Feature extraction failed!" << std::endl;
        return 1;
    }
    
    std::cout << "✅ Feature extraction successful!" << std::endl << std::endl;
    
    // Compute output statistics
    std::cout << "Output shape: [" << features.size() << ", " << features[0].size() << "]" << std::endl;
    
    double feat_sum = 0.0, feat_sum_sq = 0.0;
    size_t count = 0;
    float feat_min = features[0][0], feat_max = features[0][0];
    
    for (const auto& frame : features) {
        for (float val : frame) {
            feat_sum += val;
            feat_sum_sq += val * val;
            feat_min = std::min(feat_min, val);
            feat_max = std::max(feat_max, val);
            ++count;
        }
    }
    
    double feat_mean = feat_sum / count;
    double feat_variance = (feat_sum_sq / count) - (feat_mean * feat_mean);
    double feat_std = std::sqrt(feat_variance);
    
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Output statistics:" << std::endl;
    std::cout << "  Mean: " << feat_mean << std::endl;
    std::cout << "  Std:  " << feat_std << std::endl;
    std::cout << "  Min:  " << feat_min << std::endl;
    std::cout << "  Max:  " << feat_max << std::endl << std::endl;
    
    // Show first frame
    std::cout << "First frame (first 10 values): [";
    for (size_t i = 0; i < std::min(size_t(10), features[0].size()); ++i) {
        std::cout << features[0][i];
        if (i < 9) std::cout << ", ";
    }
    std::cout << "]" << std::endl << std::endl;
    
    // Save features to binary for Python comparison
    std::ofstream out_file("test_features_cpp.bin", std::ios::binary);
    if (out_file.is_open()) {
        for (const auto& frame : features) {
            out_file.write(reinterpret_cast<const char*>(frame.data()), 
                          frame.size() * sizeof(float));
        }
        out_file.close();
        std::cout << "✅ Saved features to test_features_cpp.bin" << std::endl;
    }
    
    // Expected values from Python SpeechBrain
    std::cout << std::endl << "=== Comparison with Python SpeechBrain ===" << std::endl;
    std::cout << "Expected output shape: [101, 80]" << std::endl;
    std::cout << "Expected mean: ~0.0" << std::endl;
    std::cout << "Expected std: ~3.6" << std::endl;
    std::cout << "Expected range: ~[-29.5, 10.7]" << std::endl;
    
    std::cout << std::endl << "Your output shape: [" << features.size() << ", " << features[0].size() << "]" << std::endl;
    std::cout << "Your mean: " << feat_mean << std::endl;
    std::cout << "Your std: " << feat_std << std::endl;
    std::cout << "Your range: [" << feat_min << ", " << feat_max << "]" << std::endl;
    
    // Check if mean is close to zero (after normalization)
    if (std::abs(feat_mean) < 0.01) {
        std::cout << "\n✅ Mean normalization working correctly!" << std::endl;
    } else {
        std::cout << "\n⚠️  Mean not close to zero - check normalization" << std::endl;
    }
    
    return 0;
}

