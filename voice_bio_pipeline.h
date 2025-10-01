#pragma once

#include <string>
#include <vector>
#include <utility>
#include <memory>

// Forward declarations
class EcapaEngine;
class MelExtractor;
class VadEngine;
class VadIterator;

struct Timestamp {
    int start; // in samples
    int end;   // in samples
};

/**
 * Voice Biometrics Pipeline
 * 
 * High-level pipeline for speaker verification:
 * 1. Load audio
 * 2. VAD to get speech segments
 * 3. Extract mel-spectrogram features
 * 4. Compute speaker embeddings
 * 5. Compare embeddings
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
    std::vector<Timestamp> get_speech_timestamps(const std::vector<float>& wav) const;

    // Slice chunks according to timestamps and filter by min duration
    std::vector<std::pair<std::vector<float>, Timestamp>> build_chunks(
        const std::vector<float>& wav,
        const std::vector<Timestamp>& tss) const;

    // Full encode pipeline: waveform -> mel -> embedding
    std::vector<float> encode_chunk(const std::vector<float>& wav_chunk) const;

    // Aggregate embeddings (mean pool)
    std::vector<float> mean_pool(const std::vector<std::vector<float>>& embeddings) const;

    // Full pipeline from file
    std::pair<std::vector<float>, std::vector<Timestamp>> extract_embedding(const std::string& audio_path) const;

    // Cosine similarity
    static float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b);

private:
    // Mean normalization for mel features (matches SpeechBrain InputNormalization)
    void normalize_features(std::vector<std::vector<float>>& features) const;
    int sr_;
    float min_speech_duration_sec_ = 0.5f;

    // Engines (separated for clean architecture)
    std::unique_ptr<EcapaEngine> ecapa_engine_;
    std::unique_ptr<MelExtractor> mel_extractor_;
    std::unique_ptr<VadEngine> vad_engine_;
    std::unique_ptr<VadIterator> vad_engine_2_;
    
};


