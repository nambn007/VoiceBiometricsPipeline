#pragma once

#include <vector>
#include <complex>

inline
std::vector<float> HammingWindow(int size) {
    std::vector<float> window(size);
    for (int i = 0; i < size; ++i) {
        window[i] = 0.54 - 0.46 * std::cos(2.0 * M_PI * i / (size - 1));
    }
    return window;
}

/**
 * Mel-Spectrogram Feature Extractor
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
    std::vector<std::vector<float>> extract(const std::vector<float>& waveform);
    std::vector<std::vector<float>> extract(const std::vector<std::vector<float>>& waveform_chunks);

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
    
    // Helper functions
    static float hz_to_mel(float hz);
    static float mel_to_hz(float mel);
    
    std::vector<std::vector<float>> mean_var_norm(const std::vector<std::vector<float>>& features);

    // STFT computation
    std::vector<std::vector<std::complex<float>>> compute_stft(const std::vector<float>& signal,
                                                               int win_length_samples,
                                                               int hop_length_samples,
                                                               int n_fft,
                                                               bool center = true,
                                                               const std::string &pad_mode = "reflect") const {};
    
    std::vector<std::vector<float>> compute_spectral_magnitude(
        const std::vector<std::vector<std::complex<float>>>& stft,
        float power = 1.0f,
        bool log_scale = false,
        float eps = 1e-14f
    ) {};
};
