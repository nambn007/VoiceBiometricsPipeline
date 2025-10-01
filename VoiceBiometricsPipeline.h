#pragma once

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <memory>

struct SpeechTimestamp {
    int start;
    int end;
};

// Forward declarations của các engine đã có
class VadIterator;
class EcapaEngine;
class MelExtractor;

class VoiceBiometricsPipeline {
public:
    VoiceBiometricsPipeline(
        const std::string& vad_model_path,
        const std::string& ecapa_model_path,
        int sampling_rate = 16000
    );
    
    ~VoiceBiometricsPipeline();
    
    // Extract embedding from audio file
    std::vector<float> extractEmbedding(
        const std::string& audio_path,
        std::vector<SpeechTimestamp>& out_chunks
    );
    
    // Extract embedding from waveform
    std::vector<float> extractEmbeddingFromWaveform(
        const std::vector<float>& waveform
    );
    
    // Compute cosine similarity between two embeddings
    float computeSimilarity(
        const std::vector<float>& embedding1,
        const std::vector<float>& embedding2
    );

private:
    // Audio feature extraction (compute_features from SpeechBrain)
    std::vector<std::vector<float>> computeFeatures(
        const std::vector<float>& waveform
    );
    
    // Mean-variance normalization
    std::vector<std::vector<float>> meanVarNorm(
        const std::vector<std::vector<float>>& features
    );
    
    // Read audio file (supports WAV)
    std::vector<float> readAudio(
        const std::string& audio_path,
        int target_sr
    );
    
    // Collect speech chunks from timestamps
    std::vector<float> collectChunks(
        const std::vector<SpeechTimestamp>& timestamps,
        const std::vector<float>& waveform
    );
    
    // VAD and ECAPA engines
    std::unique_ptr<VadIterator> vad_engine_;
    std::unique_ptr<EcapaEngine> ecapa_engine_;
    std::unique_ptr<MelExtractor> mel_extractor_;
    
    int sampling_rate_;
    float min_speech_duration_;  // seconds
};

// Helper functions for audio processing
namespace AudioUtils {
    std::vector<float> loadWavFile(const std::string& filepath, int& sample_rate);
    
    std::vector<std::vector<float>> computeMelSpectrogram(
        const std::vector<float>& waveform,
        int sample_rate,
        int n_fft,
        int hop_length,
        int n_mels,
        int win_length
    );
    
    std::vector<float> applyHammingWindow(const std::vector<float>& frame);
    void computeFFT(std::vector<float>& real, std::vector<float>& imag);
    std::vector<float> powerSpectrum(const std::vector<float>& real, const std::vector<float>& imag);
    std::vector<std::vector<float>> createMelFilterbank(int n_mels, int n_fft, int sample_rate);
}