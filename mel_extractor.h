#pragma once

#include <vector>
#include <complex>
#include <iostream>
#include <fftw3.h>
#include <memory>
#include "audio_features.h"


inline
std::vector<float> HammingWindow(int size) {
    std::vector<float> window(size);
    for (int i = 0; i < size; ++i) {
        window[i] = 0.54 - 0.46 * std::cos(2.0 * M_PI * i / (size - 1));
    }
    return window;
}


// Hàm tính magnitude từ STFT
inline
std::vector<std::vector<float>> ComputeMagnitude(
    const std::vector<std::vector<std::complex<float>>>& stft,
    float power = 1.0f,
    bool log_scale = false,
    float eps = 1e-14f
) {
    std::vector<std::vector<float>> magnitude;
    magnitude.reserve(stft.size());
    
    for (const auto& frame : stft) {
        std::vector<float> mag_frame;
        mag_frame.reserve(frame.size());
        
        for (const auto& val : frame) {
            // Tính real² + imag² (power spectrum)
            float power_spec = val.real() * val.real() + val.imag() * val.imag();
            
            // Add eps nếu power < 1 (để tránh NaN khi spectr = 0)
            if (power < 1.0f) {
                power_spec += eps;
            }
            
            // Áp dụng power
            float magnitude_val = std::pow(power_spec, power);
            
            // Áp dụng log nếu cần
            if (log_scale) {
                magnitude_val = std::log(magnitude_val + eps);
            }
            
            mag_frame.push_back(magnitude_val);
        }
        magnitude.push_back(mag_frame);
    }
    
    return magnitude;
}


// Hàm tính STFT (giống SpeechBrain)
inline 
std::vector<std::vector<std::complex<float>>> ComputeSTFT(
    const std::vector<float>& signal,
    int win_length_samples,
    int hop_length_samples,
    int n_fft,
    bool center = true,
    const std::string& pad_mode = "constant"
) {
    std::vector<std::vector<std::complex<float>>> stft_result;
    std::vector<float> window = HammingWindow(win_length_samples);
    
    // Apply center padding if needed
    std::vector<float> padded_signal = signal;
    if (center) {
        int pad_amount = n_fft / 2;
        std::vector<float> temp(signal.size() + 2 * pad_amount);
        
        // Pad left
        for (int i = 0; i < pad_amount; ++i) {
            if (pad_mode == "reflect") {
                temp[i] = signal[std::min(pad_amount - i, (int)signal.size() - 1)];
            } else { // constant
                temp[i] = 0.0f;
            }
        }
        
        // Copy signal
        std::copy(signal.begin(), signal.end(), temp.begin() + pad_amount);
        
        // Pad right
        for (int i = 0; i < pad_amount; ++i) {
            if (pad_mode == "reflect") {
                temp[signal.size() + pad_amount + i] = 
                    signal[std::max(0, (int)signal.size() - 2 - i)];
            } else { // constant
                temp[signal.size() + pad_amount + i] = 0.0f;
            }
        }
        
        padded_signal = temp;
    }
    
    // Calculate number of frames correctly
    int num_frames = 1 + (padded_signal.size() - n_fft) / hop_length_samples;
    
    // Setup FFTW
    fftwf_complex* fft_in = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * n_fft);
    fftwf_complex* fft_out = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * n_fft);
    fftwf_plan plan = fftwf_plan_dft_1d(n_fft, fft_in, fft_out, FFTW_FORWARD, FFTW_ESTIMATE);
    
    for (int frame_idx = 0; frame_idx < num_frames; ++frame_idx) {
        int start = frame_idx * hop_length_samples;
        
        // Zero padding and apply window
        for (int i = 0; i < n_fft; ++i) {
            if (i < win_length_samples && (start + i) < padded_signal.size()) {
                fft_in[i][0] = padded_signal[start + i] * window[i];
            } else {
                fft_in[i][0] = 0.0f;
            }
            fft_in[i][1] = 0.0f;
        }
        
        // Execute FFT
        fftwf_execute(plan);
        
        // Store result (only n_fft/2 + 1 bins for onesided=True)
        std::vector<std::complex<float>> frame_fft;
        frame_fft.reserve(n_fft / 2 + 1);
        for (int i = 0; i < n_fft / 2 + 1; ++i) {
            frame_fft.push_back(std::complex<float>(fft_out[i][0], fft_out[i][1]));
        }
        stft_result.push_back(frame_fft);
    }
    
    // Cleanup
    fftwf_destroy_plan(plan);
    fftwf_free(fft_in);
    fftwf_free(fft_out);
    
    return stft_result;
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
            filterbank_matrix_.assign(n_stft, std::vector<float>(n_mels_, 0.0f));
        
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
        
            // Compute bands & central freqs
            // Compute bands & central freqs
            std::vector<float> f_central(n_mels_);
            std::vector<float> band(n_mels_);
            for (int i = 0; i < n_mels_; ++i) {
                f_central[i] = hz_points[i+1];
                band[i] = hz_points[i+1] - hz_points[i];  // ĐÃ SỬA: lấy khoảng cách giữa 2 điểm liên tiếp
            }

            // Frequency axis (linear bins in Hz)
            std::vector<float> all_freqs(n_stft);
            for (int i = 0; i < n_stft; ++i) {
                all_freqs[i] = (sample_rate_ / 2.0f) * i / (n_stft - 1);
            }
        
            // Build triangular filters
            for (int m = 0; m < n_mels_; ++m) {
                for (int f = 0; f < n_stft; ++f) {
                    float slope = (all_freqs[f] - f_central[m]) / band[m];
                    float left_side = slope + 1.0f;
                    float right_side = -slope + 1.0f;
                    float val = std::min(left_side, right_side);
                    if (val < 0.0f) val = 0.0f;
                    filterbank_matrix_[f][m] = val;
                }
            }

            // for (int i = 0; i < filterbank_matrix_.size(); i++) {
            //     for (int j = 0; j < filterbank_matrix_[i].size(); j++) {
            //         std::cout << filterbank_matrix_[i][j] << " ";
            //     }
            //     std::cout << std::endl;
            // }
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
            std::cout << "num_frames: " << num_frames << std::endl;
            std::cout << "n_mels_: " << n_mels_ << std::endl;
            std::cout << "magnitude size: " << magnitude.size() << std::endl;
            std::cout << "magnitude[0] size: " << magnitude[0].size() << std::endl;
            std::cout << "filterbank_matrix_ size: " << filterbank_matrix_.size() << std::endl;
            std::cout << "filterbank_matrix_[0] size: " << filterbank_matrix_[0].size() << std::endl;
            
            for (int t = 0; t < num_frames; ++t) {
                for (int m = 0; m < n_mels_; ++m) {
                    float sum = 0.0f;
                    for (int f = 0; f < (int)magnitude[t].size(); ++f) {
                        sum += magnitude[t][f] * filterbank_matrix_[f][m];
                    }
                    fbank[t][m] = sum;
                }
            }
            
            // Convert to log scale if needed
            if (use_log) {
                // Note: input "magnitude" here is amplitude (not power).
                // Use 20*log10 for amplitude spectrograms to match SpeechBrain when
                // power_spectrogram=1. If you pass power spectrogram, set this to 10.
                // float multiplier = 20.0f; // For amplitude spectrogram (power=1)
                // float db_multiplier = std::log10(std::max(amin, 1.0f));
                
                // for (int t = 0; t < num_frames; ++t) {
                //     // Find max for this frame
                //     float max_db = -std::numeric_limits<float>::infinity();
                    
                //     for (int m = 0; m < n_mels_; ++m) {
                //         // Clamp and convert to dB
                //         float val = std::max(fbank[t][m], amin);
                //         fbank[t][m] = multiplier * std::log10(val) - 
                //                       multiplier * db_multiplier;
                //         max_db = std::max(max_db, fbank[t][m]);
                //     }
                    
                //     // Apply top_db clipping
                //     float min_db = max_db - top_db;
                //     for (int m = 0; m < n_mels_; ++m) {
                //         fbank[t][m] = std::max(fbank[t][m], min_db);
                //     }
                // }
                float multiplier = 20.0f;
                float db_multiplier = std::log10(std::max(amin, 1.0f));
    
                // chuyển sang dB
                float global_max_db = -std::numeric_limits<float>::infinity();
                for (int t = 0; t < num_frames; ++t) {
                    for (int m = 0; m < n_mels_; ++m) {
                        float val = std::max(fbank[t][m], amin);
                        fbank[t][m] = multiplier * std::log10(val) -
                                      multiplier * db_multiplier;
                        global_max_db = std::max(global_max_db, fbank[t][m]);
                    }
                }
    
                // clipping theo top_db (global)
                float min_db = global_max_db - top_db;
                for (int t = 0; t < num_frames; ++t) {
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
        float f_min = 0.0f,
        float f_max = 8000.0f,
        int n_fft = 400,
        int n_mels = 80,
        int win_length = 25, // ms 
        int hop_length = 10 // ms 
    );
    
    // Extract mel-spectrogram from waveform
    std::vector<std::vector<float>> extract(const std::vector<float>& waveform);
    std::vector<std::vector<float>> extract(const std::vector<std::vector<float>>& waveform_chunks);

private:
    int sample_rate_;
    int n_fft_;
    int hop_length_; // ms 
    int win_length_; // ms 
    int n_mels_;
    float f_min_;
    float f_max_;

    std::unique_ptr<STFTProcessor> stft_processor_;
    std::unique_ptr<FilterBank> filterbank_;
    
    // Mel filterbank matrix [n_mels x (n_fft/2 + 1)]
    std::vector<std::vector<float>> mel_filters_;
    
    // Hanning window
    std::vector<float> window_;
    
    // Helper functions
    static float hz_to_mel(float hz);
    static float mel_to_hz(float mel);
    
    std::vector<std::vector<float>> mean_var_norm(const std::vector<std::vector<float>>& features, bool std_norm = false);

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
