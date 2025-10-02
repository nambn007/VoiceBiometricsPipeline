#include "audio_feature_extractor.h"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

int main() {
    std::cout << "=== Testing AudioFeatureExtractor ===" << std::endl << std::endl;
    
    // Create extractor with SpeechBrain-compatible parameters
    AudioFeatureExtractor extractor(
        16000,  // sample_rate
        400,    // n_fft
        160,    // hop_length
        80,     // n_mels
        0.0f,   // f_min
        8000.0f // f_max
    );
    
    std::cout << std::endl;
    
    // Test 1: Single waveform processing
    {
        std::cout << "Test 1: Processing single waveform" << std::endl;
        std::cout << "-----------------------------------" << std::endl;
        
        // Create test waveform (1 second of audio)
        std::vector<float> waveform(16000);
        std::mt19937 gen(42);  // Fixed seed for reproducibility
        std::normal_distribution<float> dist(0.0f, 0.1f);
        
        for (auto& sample : waveform) {
            sample = dist(gen);
        }
        
        std::cout << "Input: " << waveform.size() << " samples (" 
                  << (waveform.size() / 16000.0) << " seconds)" << std::endl;
        
        // Process
        auto features = extractor.process(waveform);
        
        if (!features.empty()) {
            std::cout << "Output: " << features.size() << " frames x " 
                      << features[0].size() << " features" << std::endl;
            
            // Compute statistics
            double sum = 0.0, sum_sq = 0.0;
            size_t count = 0;
            float min_val = features[0][0];
            float max_val = features[0][0];
            
            for (const auto& frame : features) {
                for (float val : frame) {
                    sum += val;
                    sum_sq += val * val;
                    min_val = std::min(min_val, val);
                    max_val = std::max(max_val, val);
                    ++count;
                }
            }
            
            double mean = sum / count;
            double variance = (sum_sq / count) - (mean * mean);
            double std_dev = std::sqrt(variance);
            
            std::cout << std::fixed << std::setprecision(6);
            std::cout << "Statistics:" << std::endl;
            std::cout << "  Mean: " << mean << std::endl;
            std::cout << "  Std:  " << std_dev << std::endl;
            std::cout << "  Min:  " << min_val << std::endl;
            std::cout << "  Max:  " << max_val << std::endl;
            
            // Show first frame
            std::cout << "First frame (first 5 values): [";
            for (size_t i = 0; i < std::min(size_t(5), features[0].size()); ++i) {
                std::cout << features[0][i];
                if (i < 4) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
            
            std::cout << "✅ Test 1 passed!" << std::endl;
        } else {
            std::cout << "❌ Test 1 failed: No features extracted" << std::endl;
        }
    }
    
    std::cout << std::endl;
    
    // Test 2: Multiple chunks processing (simulating VAD segments)
    {
        std::cout << "Test 2: Processing multiple chunks" << std::endl;
        std::cout << "-----------------------------------" << std::endl;
        
        std::mt19937 gen(123);
        std::normal_distribution<float> dist(0.0f, 0.1f);
        
        // Create 3 chunks of different lengths
        std::vector<std::vector<float>> chunks;
        
        chunks.push_back(std::vector<float>(8000));   // 0.5 seconds
        chunks.push_back(std::vector<float>(12000));  // 0.75 seconds
        chunks.push_back(std::vector<float>(16000));  // 1.0 second
        
        for (auto& chunk : chunks) {
            for (auto& sample : chunk) {
                sample = dist(gen);
            }
        }
        
        std::cout << "Input: " << chunks.size() << " chunks with sizes: [";
        for (size_t i = 0; i < chunks.size(); ++i) {
            std::cout << chunks[i].size();
            if (i < chunks.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        // Process all chunks
        auto features = extractor.process_chunks(chunks);
        
        if (!features.empty()) {
            std::cout << "✅ Test 2 passed!" << std::endl;
        } else {
            std::cout << "❌ Test 2 failed: No features extracted" << std::endl;
        }
    }
    
    std::cout << std::endl;
    std::cout << "=== All tests completed ===" << std::endl;
    
    return 0;
}

