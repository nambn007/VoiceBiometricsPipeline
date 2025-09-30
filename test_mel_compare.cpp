#include "mel_extractor.h"
#include <iostream>
#include <random>
#include <cmath>

int main() {
    // Create test waveform (deterministic, same seed as Python)
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    std::vector<float> wav(16000);
    for (int i = 0; i < 16000; ++i) {
        wav[i] = dist(gen);
    }
    
    // Kaldi Native Fbank
    MelExtractor extractor(16000, 400, 160, 80, 0.0f, 8000.0f);
    auto features = extractor.extract(wav);
    
    std::cout << "=== KALDI NATIVE FBANK ===" << std::endl;
    std::cout << "Output shape: [" << features.size() << ", " << features[0].size() << "]" << std::endl;
    
    // Compute statistics
    double sum = 0.0, sum_sq = 0.0;
    float min_val = features[0][0], max_val = features[0][0];
    
    for (const auto& frame : features) {
        for (float val : frame) {
            sum += val;
            sum_sq += val * val;
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
        }
    }
    
    int total = features.size() * features[0].size();
    double mean = sum / total;
    double variance = (sum_sq / total) - (mean * mean);
    double std_dev = std::sqrt(variance);
    
    std::cout << "Mean: " << mean << std::endl;
    std::cout << "Std: " << std_dev << std::endl;
    std::cout << "Min: " << min_val << std::endl;
    std::cout << "Max: " << max_val << std::endl;
    
    std::cout << "Sample values [0, 0:5]: [";
    for (int i = 0; i < 5 && i < features[0].size(); ++i) {
        std::cout << features[0][i];
        if (i < 4) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    return 0;
}
