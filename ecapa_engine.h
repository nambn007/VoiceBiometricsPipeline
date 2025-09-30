#pragma once

#include <string>
#include <vector>
#include <memory>
#include <onnxruntime_cxx_api.h>

/**
 * ECAPA-TDNN ONNX Inference Engine
 * 
 * Handles speaker embedding extraction using pre-trained ECAPA-TDNN model.
 * Input: mel-spectrogram features [batch, time_frames, 80]
 * Output: speaker embedding [192]
 */
class EcapaEngine {
public:
    explicit EcapaEngine(const std::string& model_path);
    
    // Compute speaker embedding from mel-spectrogram features
    // Input: mel_features [time_frames, 80]
    // Output: embedding [192]
    std::vector<float> compute_embedding(const std::vector<std::vector<float>>& mel_features) const;
    
    int embedding_dim() const { return embedding_dim_; }
    bool is_loaded() const { return session_ != nullptr; }

private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_;
    
    // Model metadata
    std::string input_name_;
    std::string output_name_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    int embedding_dim_ = 192;
};
