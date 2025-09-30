#pragma once

#include <string>
#include <vector>
#include <memory>
#include <onnxruntime_cxx_api.h>

// Forward declare Timestamp (defined in voice_bio_pipeline.h)
struct Timestamp;

/**
 * Silero VAD ONNX Inference Engine
 * 
 * Voice Activity Detection using Silero VAD model.
 * Detects speech segments in audio with high accuracy.
 * 
 * Model: silero_vad.onnx (16kHz, mono)
 */
class VadEngine {
public:
    explicit VadEngine(const std::string& model_path, int sample_rate = 16000);
    
    // Detect speech timestamps in audio
    // Returns list of [start, end] timestamps in samples
    std::vector<Timestamp> detect_speech(
        const std::vector<float>& wav,
        float threshold = 0.5f,
        int min_speech_duration_ms = 250,
        int min_silence_duration_ms = 100
    );
    
    // Reset internal state (for streaming)
    void reset_states();
    
    bool is_loaded() const { return session_ != nullptr; }

private:
    int sample_rate_;
    
    // ONNX Runtime
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_;
    
    // Model I/O names
    std::string input_name_;
    std::string sr_name_;
    std::string state_name_;
    std::string output_name_;
    std::string output_state_name_;
    
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    
    // State tensors (2, 1, 128)
    std::vector<float> state_;
    
    // Window size for processing
    int window_size_samples_;
    
    // Process single window
    float process_window(const float* audio_data, int size);
};
