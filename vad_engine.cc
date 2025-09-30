#include "vad_engine.h"
#include "voice_bio_pipeline.h"  // For Timestamp definition
#include <iostream>
#include <algorithm>
#include <cstring>

VadEngine::VadEngine(const std::string& model_path, int sample_rate)
    : sample_rate_(sample_rate),
      env_(ORT_LOGGING_LEVEL_WARNING, "vad-engine"),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      window_size_samples_(sample_rate == 16000 ? 512 : 256) {
    
    if (model_path.empty()) {
        throw std::runtime_error("VAD model path is empty");
    }
    
    // Initialize state (2, 1, 128)
    state_.resize(2 * 1 * 128, 0.0f);
    
    // Setup session options
    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    
    try {
        // Load model
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);
        
        // Get input names
        Ort::AllocatorWithDefaultOptions allocator;
        
        // Input: "input" - audio samples
        auto input_name_allocated = session_->GetInputNameAllocated(0, allocator);
        input_name_ = input_name_allocated.get();
        input_names_.push_back(input_name_.c_str());
        
        // Input: "sr" - sample rate
        auto sr_name_allocated = session_->GetInputNameAllocated(1, allocator);
        sr_name_ = sr_name_allocated.get();
        input_names_.push_back(sr_name_.c_str());
        
        // Input: "state" - RNN state
        auto state_name_allocated = session_->GetInputNameAllocated(2, allocator);
        state_name_ = state_name_allocated.get();
        input_names_.push_back(state_name_.c_str());
        
        // Output: "output" - speech probability
        auto output_name_allocated = session_->GetOutputNameAllocated(0, allocator);
        output_name_ = output_name_allocated.get();
        output_names_.push_back(output_name_.c_str());
        
        // Output: "stateN" - updated state
        auto output_state_allocated = session_->GetOutputNameAllocated(1, allocator);
        output_state_name_ = output_state_allocated.get();
        output_names_.push_back(output_state_name_.c_str());
        
        std::cout << "✅ VAD Engine loaded: " << model_path << std::endl;
        std::cout << "   Sample rate: " << sample_rate_ << " Hz" << std::endl;
        std::cout << "   Window size: " << window_size_samples_ << " samples" << std::endl;
        
    } catch (const Ort::Exception& e) {
        std::cerr << "❌ Failed to load VAD model: " << e.what() << std::endl;
        throw;
    }
}

void VadEngine::reset_states() {
    std::fill(state_.begin(), state_.end(), 0.0f);
}

float VadEngine::process_window(const float* audio_data, int size) {
    try {
        // Prepare input tensor: [1, window_size]
        std::vector<int64_t> input_shape = {1, size};
        std::vector<float> input_data(audio_data, audio_data + size);
        
        auto input_tensor = Ort::Value::CreateTensor<float>(
            memory_info_,
            input_data.data(),
            input_data.size(),
            input_shape.data(),
            input_shape.size()
        );
        
        // Prepare sample rate tensor: [1] - must be int64
        std::vector<int64_t> sr_shape = {1};
        int64_t sr_value = static_cast<int64_t>(sample_rate_);
        
        auto sr_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info_,
            &sr_value,
            1,
            sr_shape.data(),
            sr_shape.size()
        );
        
        // Prepare state tensor: [2, 1, 128]
        std::vector<int64_t> state_shape = {2, 1, 128};
        
        auto state_tensor = Ort::Value::CreateTensor<float>(
            memory_info_,
            state_.data(),
            state_.size(),
            state_shape.data(),
            state_shape.size()
        );
        
        // Run inference
        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(std::move(input_tensor));
        input_tensors.push_back(std::move(sr_tensor));
        input_tensors.push_back(std::move(state_tensor));
        
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names_.data(),
            input_tensors.data(),
            input_tensors.size(),
            output_names_.data(),
            output_names_.size()
        );
        
        // Extract speech probability
        float* prob_data = output_tensors[0].GetTensorMutableData<float>();
        float speech_prob = prob_data[0];
        
        // Update state
        float* new_state_data = output_tensors[1].GetTensorMutableData<float>();
        std::copy(new_state_data, new_state_data + state_.size(), state_.begin());
        
        return speech_prob;
        
    } catch (const Ort::Exception& e) {
        std::cerr << "❌ VAD inference error: " << e.what() << std::endl;
        throw;
    }
}

std::vector<Timestamp> VadEngine::detect_speech(
    const std::vector<float>& wav,
    float threshold,
    int min_speech_duration_ms,
    int min_silence_duration_ms
) {
    if (!session_) {
        throw std::runtime_error("VAD session not initialized");
    }
    
    reset_states();
    
    std::vector<Timestamp> timestamps;
    
    // Convert durations to samples
    int min_speech_samples = (min_speech_duration_ms * sample_rate_) / 1000;
    int min_silence_samples = (min_silence_duration_ms * sample_rate_) / 1000;
    
    bool triggered = false;
    int speech_start = 0;
    int temp_end = 0;
    int current_sample = 0;
    
    // Process audio in windows
    for (size_t i = 0; i + window_size_samples_ <= wav.size(); i += window_size_samples_) {
        float speech_prob = process_window(wav.data() + i, window_size_samples_);
        current_sample = static_cast<int>(i) + window_size_samples_;
        
        // Speech detected
        if (speech_prob >= threshold && !triggered) {
            triggered = true;
            speech_start = std::max(0, static_cast<int>(i));
            temp_end = 0;
        }
        
        // Silence detected during speech
        if (speech_prob < threshold - 0.15f && triggered) {
            if (temp_end == 0) {
                temp_end = current_sample;
            }
            
            // Check if silence duration is long enough
            if (current_sample - temp_end >= min_silence_samples) {
                int speech_end = temp_end;
                
                // Check if speech duration is long enough
                if (speech_end - speech_start >= min_speech_samples) {
                    timestamps.push_back({speech_start, speech_end});
                }
                
                triggered = false;
                temp_end = 0;
            }
        }
        
        // Reset temp_end if speech continues
        if (speech_prob >= threshold && temp_end != 0) {
            temp_end = 0;
        }
    }
    
    // Handle final speech segment
    if (triggered) {
        int speech_end = static_cast<int>(wav.size());
        if (speech_end - speech_start >= min_speech_samples) {
            timestamps.push_back({speech_start, speech_end});
        }
    }
    
    return timestamps;
}
