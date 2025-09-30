#include "ecapa_engine.h"
#include <iostream>
#include <stdexcept>

EcapaEngine::EcapaEngine(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "ecapa-engine"),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    
    if (model_path.empty()) {
        throw std::runtime_error("ECAPA model path is empty");
    }
    
    // Setup session options
    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    
    try {
        // Load model
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);
        
        // Get input/output names
        Ort::AllocatorWithDefaultOptions allocator;
        
        auto input_name_allocated = session_->GetInputNameAllocated(0, allocator);
        input_name_ = input_name_allocated.get();
        input_names_.push_back(input_name_.c_str());
        
        auto output_name_allocated = session_->GetOutputNameAllocated(0, allocator);
        output_name_ = output_name_allocated.get();
        output_names_.push_back(output_name_.c_str());
        
        // Get output shape to determine embedding dimension
        auto output_shape = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (!output_shape.empty()) {
            embedding_dim_ = static_cast<int>(output_shape.back());
        }
        
        std::cout << "✅ ECAPA Engine loaded: " << model_path << std::endl;
        std::cout << "   Input: " << input_name_ << std::endl;
        std::cout << "   Output: " << output_name_ << " [" << embedding_dim_ << "]" << std::endl;
        
    } catch (const Ort::Exception& e) {
        std::cerr << "❌ Failed to load ECAPA model: " << e.what() << std::endl;
        throw;
    }
}

std::vector<float> EcapaEngine::compute_embedding(const std::vector<std::vector<float>>& mel_features) const {
    if (!session_) {
        throw std::runtime_error("ECAPA session not initialized");
    }
    
    if (mel_features.empty() || mel_features[0].size() != 80) {
        throw std::runtime_error("Invalid mel features shape (expected [time, 80])");
    }
    
    try {
        // Prepare input tensor: [batch=1, time_frames, mel_features=80]
        int n_frames = static_cast<int>(mel_features.size());
        int n_mels = 80;
        
        std::vector<int64_t> input_shape = {1, n_frames, n_mels};
        size_t input_size = 1 * n_frames * n_mels;
        
        // Flatten mel_features to 1D array (row-major: batch, time, mel)
        std::vector<float> input_data(input_size);
        for (int t = 0; t < n_frames; ++t) {
            for (int f = 0; f < n_mels; ++f) {
                input_data[t * n_mels + f] = mel_features[t][f];
            }
        }
        
        // Create input tensor
        auto input_tensor = Ort::Value::CreateTensor<float>(
            memory_info_, 
            input_data.data(), 
            input_size,
            input_shape.data(), 
            input_shape.size()
        );
        
        // Run inference
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names_.data(),
            &input_tensor,
            1,
            output_names_.data(),
            1
        );
        
        // Extract output: [batch=1, 1, embedding_dim]
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        
        // Calculate total elements in output
        size_t total_elements = 1;
        for (auto dim : output_shape) {
            total_elements *= dim;
        }
        
        std::vector<float> embedding(output_data, output_data + total_elements);
        
        return embedding;
        
    } catch (const Ort::Exception& e) {
        std::cerr << "❌ ONNX Runtime error: " << e.what() << std::endl;
        throw;
    }
}
