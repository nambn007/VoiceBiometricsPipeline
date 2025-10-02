#pragma once

#include <vector>
#include <memory>

// Forward declarations
class MelExtractor;

/**
 * AudioFeatureExtractor - Processes audio waveform chunks to extract features
 * 
 * This class replicates the SpeechBrain preprocessing pipeline:
 * 1. compute_features(wavs) -> Fbank mel-spectrogram
 * 2. mean_var_norm(feats, wav_lens) -> Per-utterance normalization
 * 
 * Input: std::vector<float> waveform (mono, 16kHz, normalized [-1, 1])
 * Output: std::vector<std::vector<float>> normalized features [n_frames, n_mels]
 */
class AudioFeatureExtractor {
public:
    /**
     * Constructor
     * 
     * @param sample_rate Sample rate in Hz (default: 16000)
     * @param n_fft FFT size (default: 400)
     * @param hop_length Hop length in samples (default: 160)
     * @param n_mels Number of mel bins (default: 80)
     * @param f_min Minimum frequency (default: 0.0)
     * @param f_max Maximum frequency (default: 8000.0)
     */
    AudioFeatureExtractor(
        int sample_rate = 16000,
        int n_fft = 400,
        int hop_length = 160,
        int n_mels = 80,
        float f_min = 0.0f,
        float f_max = 8000.0f
    );
    
    ~AudioFeatureExtractor();
    
    /**
     * Process audio waveform to extract normalized features
     * 
     * This method performs:
     * 1. Mel-spectrogram extraction (compute_features)
     * 2. Mean-variance normalization (mean_var_norm with norm_type="sentence", std_norm=False)
     * 
     * @param waveform Input audio samples (mono, normalized [-1, 1])
     * @return Normalized mel features [n_frames, n_mels]
     */
    std::vector<std::vector<float>> process(const std::vector<float>& waveform) const;
    
    /**
     * Process multiple audio chunks (used when you have multiple VAD segments)
     * 
     * Each chunk is processed independently, then features are concatenated.
     * This matches the behavior of processing collected VAD chunks.
     * 
     * @param audio_chunks Vector of audio chunks
     * @return Concatenated normalized features [total_frames, n_mels]
     */
    std::vector<std::vector<float>> process_chunks(
        const std::vector<std::vector<float>>& audio_chunks
    ) const;

private:
    /**
     * Step 1: Compute mel-spectrogram features (matches compute_features in SpeechBrain)
     * 
     * Uses kaldi-native-fbank to compute log-mel filterbank features
     * 
     * @param waveform Input audio samples
     * @return Raw mel features [n_frames, n_mels]
     */
    std::vector<std::vector<float>> compute_features(
        const std::vector<float>& waveform
    ) const;
    
    /**
     * Step 2: Mean-variance normalization (matches mean_var_norm in SpeechBrain)
     * 
     * Implements InputNormalization with:
     * - norm_type="sentence" (per-utterance normalization)
     * - std_norm=False (only normalize mean, not std)
     * 
     * Formula: normalized = (features - mean) / 1.0
     * 
     * @param features Input features [n_frames, n_mels]
     * @return Normalized features [n_frames, n_mels]
     */
    std::vector<std::vector<float>> mean_var_norm(
        const std::vector<std::vector<float>>& features
    ) const;

private:
    std::unique_ptr<MelExtractor> mel_extractor_;
    int sample_rate_;
};

