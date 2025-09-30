#pragma once

#include <vector>
#include <complex>

/**
 * Mel-Spectrogram Feature Extractor
 * 
 * Extracts mel-spectrogram features from raw audio waveform.
 * Compatible with SpeechBrain Fbank parameters:
 * - sample_rate: 16000 Hz
 * - n_fft: 400
 * - hop_length: 160
 * - n_mels: 80
 * - f_min: 0 Hz
 * - f_max: 8000 Hz
 */
class MelExtractor {
public:
    MelExtractor(
        int sample_rate = 16000,
        int n_fft = 400,
        int hop_length = 160,
        int n_mels = 80,
        float f_min = 0.0f,
        float f_max = 8000.0f
    );
    
    // Extract mel-spectrogram from waveform
    // Input: waveform (normalized float32, typically [-1, 1])
    // Output: mel_features [time_frames, n_mels]
    std::vector<std::vector<float>> extract(const std::vector<float>& waveform) const;

private:
    int sample_rate_;
    int n_fft_;
    int hop_length_;
    int n_mels_;
    float f_min_;
    float f_max_;
    
    // Mel filterbank matrix [n_mels x (n_fft/2 + 1)]
    std::vector<std::vector<float>> mel_filters_;
    
    // Hanning window
    std::vector<float> window_;
    
    void init_mel_filters();
    void init_window();
    
    // Helper functions
    static float hz_to_mel(float hz);
    static float mel_to_hz(float mel);
    
    // STFT computation
    std::vector<std::vector<std::complex<float>>> compute_stft(const std::vector<float>& signal) const;
    
    // Apply mel filterbank to power spectrogram
    std::vector<std::vector<float>> apply_mel_filterbank(
        const std::vector<std::vector<std::complex<float>>>& stft) const;
};
