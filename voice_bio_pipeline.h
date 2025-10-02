#pragma once

#include "audio_feature_extractor_v2.h"
#include "vad_engine_2.h"
#include <string>
#include <vector>
#include <utility>
#include <memory>

// Forward declarations
class EcapaEngine;
class MelExtractor;
class VadEngine;
class VadIterator;


/**
 * Voice Biometrics Pipeline
 * 
 */
class VoiceBiometricsPipeline {
public:
    explicit VoiceBiometricsPipeline(
        int sampling_rate = 16000,
        const std::string& ecapa_model_path = "",
        const std::string& vad_model_path = "");
    
    ~VoiceBiometricsPipeline();

    // Load a mono WAV (16kHz) file into float vector
    std::vector<float> read_audio(const std::string& path) const;

    // Run VAD to get timestamps (stub - implement your own VAD or bind Silero)
    std::vector<timestamp_t> get_speech_timestamps(const std::vector<float>& wav) const;

    // Slice chunks according to timestamps and filter by min duration
    std::vector<std::vector<float>> build_chunks(
        const std::vector<float>& wav,
        const std::vector<timestamp_t>& tss) const;

    // Aggregate embeddings (mean pool)
    std::vector<float> mean_pool(const std::vector<std::vector<float>>& embeddings) const;

    // Full pipeline from file
    std::vector<float> extract_embedding(const std::string& audio_path);

    // Cosine similarity
    static float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b);

private:

    int sr_;
    float min_speech_duration_sec_ = 0.5f;

    // Engines (separated for clean architecture)
    std::unique_ptr<EcapaEngine> ecapa_engine_;
    std::unique_ptr<MelExtractor> mel_extractor_;
    std::unique_ptr<VadEngine> vad_engine_;
    std::unique_ptr<VadIterator> vad_engine_2_;
    std::unique_ptr<AudioFeatureExtractor> extractor_;
};


