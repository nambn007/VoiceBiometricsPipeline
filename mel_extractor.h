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

class MelFilterbank {
    private:
        int n_mels_;
        int n_fft_;
        int sample_rate_;
        float f_min_;
        float f_max_;
        std::vector<std::vector<float>> filterbank_matrix_;
        
        // Convert Hz to Mel
        float hz_to_mel(float hz) {
            return 2595.0f * std::log10(1.0f + hz / 700.0f);
        }
        
        // Convert Mel to Hz
        float mel_to_hz(float mel) {
            return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
        }
        
        // Create triangular filters
        void create_filterbank() {
            int n_stft = n_fft_ / 2 + 1;
            filterbank_matrix_.resize(n_mels_, std::vector<float>(n_stft, 0.0f));
            
            // Mel scale points
            float mel_min = hz_to_mel(f_min_);
            float mel_max = hz_to_mel(f_max_);
            std::vector<float> mel_points(n_mels_ + 2);
            
            for (int i = 0; i < n_mels_ + 2; ++i) {
                mel_points[i] = mel_min + (mel_max - mel_min) * i / (n_mels_ + 1);
            }
            
            // Convert to Hz
            std::vector<float> hz_points(n_mels_ + 2);
            for (int i = 0; i < n_mels_ + 2; ++i) {
                hz_points[i] = mel_to_hz(mel_points[i]);
            }
            
            // Convert to FFT bin numbers (one-sided spectrum: n_stft = n_fft/2 + 1)
            std::vector<int> bin_points(n_mels_ + 2);
            for (int i = 0; i < n_mels_ + 2; ++i) {
                float bin = (static_cast<float>(n_fft_ / 2 + 1) * hz_points[i]) / static_cast<float>(sample_rate_);
                bin_points[i] = static_cast<int>(std::floor(bin));
            }
            
            // Create triangular filters
            for (int m = 0; m < n_mels_; ++m) {
                int left = bin_points[m];
                int center = bin_points[m + 1];
                int right = bin_points[m + 2];
                
                // Left slope
                for (int k = left; k < center; ++k) {
                    filterbank_matrix_[m][k] = 
                        (float)(k - left) / (center - left);
                }
                
                // Right slope
                for (int k = center; k < right; ++k) {
                    filterbank_matrix_[m][k] = 
                        (float)(right - k) / (right - center);
                }

                // Slaney-style normalization: make each filter sum to 1.0
                float sum_weights = 0.0f;
                for (int k = left; k < right && k < n_stft; ++k) {
                    if (k >= 0) sum_weights += filterbank_matrix_[m][k];
                }
                if (sum_weights > 0.0f) {
                    for (int k = left; k < right && k < n_stft; ++k) {
                        if (k >= 0) filterbank_matrix_[m][k] /= sum_weights;
                    }
                }
            }
        }
        
    public:
        MelFilterbank(int n_mels, int n_fft, int sample_rate, 
                      float f_min = 0.0f, float f_max = 8000.0f)
            : n_mels_(n_mels), n_fft_(n_fft), sample_rate_(sample_rate),
              f_min_(f_min), f_max_(f_max) {
            create_filterbank();
        }
        
        // Apply filterbank to magnitude spectrogram
        std::vector<std::vector<float>> apply(
            const std::vector<std::vector<float>>& magnitude,
            bool use_log = true,
            float amin = 1e-10f,
            float top_db = 80.0f
        ) {
            int num_frames = magnitude.size();
            std::vector<std::vector<float>> fbank(num_frames, 
                                                   std::vector<float>(n_mels_, 0.0f));
            
            // Apply filterbank matrix
            for (int t = 0; t < num_frames; ++t) {
                for (int m = 0; m < n_mels_; ++m) {
                    float sum = 0.0f;
                    for (size_t k = 0; k < magnitude[t].size(); ++k) {
                        sum += magnitude[t][k] * filterbank_matrix_[m][k];
                    }
                    fbank[t][m] = sum;
                }
            }
            
            // Convert to log scale if needed
            if (use_log) {
                // Note: input "magnitude" here is amplitude (not power).
                // Use 20*log10 for amplitude spectrograms to match SpeechBrain when
                // power_spectrogram=1. If you pass power spectrogram, set this to 10.
                float multiplier = 20.0f; // For amplitude spectrogram (power=1)
                float db_multiplier = std::log10(std::max(amin, 1.0f));
                
                for (int t = 0; t < num_frames; ++t) {
                    // Find max for this frame
                    float max_db = -std::numeric_limits<float>::infinity();
                    
                    for (int m = 0; m < n_mels_; ++m) {
                        // Clamp and convert to dB
                        float val = std::max(fbank[t][m], amin);
                        fbank[t][m] = multiplier * std::log10(val) - 
                                      multiplier * db_multiplier;
                        max_db = std::max(max_db, fbank[t][m]);
                    }
                    
                    // Apply top_db clipping
                    float min_db = max_db - top_db;
                    for (int m = 0; m < n_mels_; ++m) {
                        fbank[t][m] = std::max(fbank[t][m], min_db);
                    }
                }
            }
            
            return fbank;
        }
};

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
