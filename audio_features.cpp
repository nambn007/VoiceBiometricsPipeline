#include "audio_features.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

std::vector<std::vector<float>>
compute_magnitude(
    std::vector<std::vector<std::complex<float>>> stft,
    float power,
    bool log_scale,
    float eps
){
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

std::vector<std::vector<float>> mean_var_norm(
    const std::vector<std::vector<float>>& features,
    bool std_norm  // Thêm tham số này
) {
    if (features.empty()) return features;
    
    const size_t num_frames = features.size();
    const size_t num_features = features[0].size();
    
    // Compute mean across time dimension (dim=0 in Python)
    std::vector<float> mean(num_features, 0.0f);
    
    // Calculate mean
    for (const auto& frame : features) {
        for (size_t i = 0; i < num_features; ++i) {
            mean[i] += frame[i];
        }
    }
    for (auto& m : mean) {
        m /= num_frames;
    }
    
    // Calculate standard deviation (nếu cần)
    std::vector<float> std_dev(num_features, 1.0f);  // Mặc định = 1.0
    
    if (std_norm) {
        std::fill(std_dev.begin(), std_dev.end(), 0.0f);
        
        // Calculate variance với Bessel's correction (N-1)
        for (const auto& frame : features) {
            for (size_t i = 0; i < num_features; ++i) {
                float diff = frame[i] - mean[i];
                std_dev[i] += diff * diff;
            }
        }
        
        // Chia cho (N-1) thay vì N, giống torch.std()
        float correction_factor = (num_frames > 1) ? (num_frames - 1) : 1;
        for (auto& s : std_dev) {
            s = std::sqrt(s / correction_factor);
            // Ensure numerical stability
            s = std::max(s, 1e-10f);
        }
    }
    
    // Normalize: (x - mean) / std
    std::vector<std::vector<float>> normalized = features;
    for (auto& frame : normalized) {
        for (size_t i = 0; i < num_features; ++i) {
            frame[i] = (frame[i] - mean[i]) / std_dev[i];
        }
    }
    
    return normalized;
}

STFTProcessor::STFTProcessor(
    int win_length_samples,
    int hop_length_samples,
    int n_fft,
    bool center,
    const std::string& pad_mode
) : win_length_samples_(win_length_samples),
    hop_length_samples_(hop_length_samples),
    n_fft_(n_fft),
    center_(center),
    pad_mode_(pad_mode)
{
    // Pre-compute window function
    window_ = createHammingWindow(win_length_samples_);
}

STFTProcessor::~STFTProcessor() {
    // Cleanup if needed
}

std::vector<float> STFTProcessor::createHammingWindow(int window_length) {
    std::vector<float> window(window_length);
    for (int i = 0; i < window_length; ++i) {
        window[i] = 0.54f - 0.46f * std::cos(2.0f * M_PI * i / (window_length - 1));
    }
    return window;
}

std::vector<float> STFTProcessor::applyPadding(const std::vector<float>& signal) {
    if (!center_) {
        return signal;
    }
    
    int pad_amount = n_fft_ / 2;
    std::vector<float> padded_signal(signal.size() + 2 * pad_amount);
    
    // Pad left
    for (int i = 0; i < pad_amount; ++i) {
        if (pad_mode_ == "reflect") {
            padded_signal[i] = signal[std::min(pad_amount - i, (int)signal.size() - 1)];
        } else { // constant
            padded_signal[i] = 0.0f;
        }
    }
    
    // Copy signal
    std::copy(signal.begin(), signal.end(), padded_signal.begin() + pad_amount);
    
    // Pad right
    for (int i = 0; i < pad_amount; ++i) {
        if (pad_mode_ == "reflect") {
            padded_signal[signal.size() + pad_amount + i] = 
                signal[std::max(0, (int)signal.size() - 2 - i)];
        } else { // constant
            padded_signal[signal.size() + pad_amount + i] = 0.0f;
        }
    }
    
    return padded_signal;
}

std::vector<std::vector<std::complex<float>>> STFTProcessor::compute(
    const std::vector<float>& signal
) {
    std::vector<std::vector<std::complex<float>>> stft_result;
    
    // Apply padding if needed
    std::vector<float> padded_signal = applyPadding(signal);
    
    // Calculate number of frames correctly
    int num_frames = 1 + (padded_signal.size() - n_fft_) / hop_length_samples_;
    
    // Setup FFTW
    fftwf_complex* fft_in = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * n_fft_);
    fftwf_complex* fft_out = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * n_fft_);
    fftwf_plan plan = fftwf_plan_dft_1d(n_fft_, fft_in, fft_out, FFTW_FORWARD, FFTW_ESTIMATE);
    
    for (int frame_idx = 0; frame_idx < num_frames; ++frame_idx) {
        int start = frame_idx * hop_length_samples_;
        
        // Zero padding and apply window
        for (int i = 0; i < n_fft_; ++i) {
            if (i < win_length_samples_ && (start + i) < padded_signal.size()) {
                fft_in[i][0] = padded_signal[start + i] * window_[i];
            } else {
                fft_in[i][0] = 0.0f;
            }
            fft_in[i][1] = 0.0f;
        }
        
        // Execute FFT
        fftwf_execute(plan);
        
        // Store result (only n_fft/2 + 1 bins for onesided=True)
        std::vector<std::complex<float>> frame_fft;
        frame_fft.reserve(n_fft_ / 2 + 1);
        for (int i = 0; i < n_fft_ / 2 + 1; ++i) {
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



FilterBank::FilterBank(int n_mels, int n_fft, int sample_rate, 
                       float f_min, float f_max)
    : n_mels_(n_mels), n_fft_(n_fft), sample_rate_(sample_rate),
      f_min_(f_min), f_max_(f_max) {
    create_filterbank();
}

float FilterBank::hz_to_mel(float hz) {
    return 2595.0f * std::log10(1.0f + hz / 700.0f);
}

float FilterBank::mel_to_hz(float mel) {
    return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

void FilterBank::create_filterbank() {
    int n_stft = n_fft_ / 2 + 1;
    filterbank_matrix_.assign(n_stft, std::vector<float>(n_mels_, 0.0f));

    // Create mel scale points
    float mel_min = hz_to_mel(f_min_);
    float mel_max = hz_to_mel(f_max_);
    std::vector<float> mel_points(n_mels_ + 2);
    for (int i = 0; i < n_mels_ + 2; ++i) {
        mel_points[i] = mel_min + (mel_max - mel_min) * i / (n_mels_ + 1);
    }

    // Convert mel points to Hz
    std::vector<float> hz_points(n_mels_ + 2);
    for (int i = 0; i < n_mels_ + 2; ++i) {
        hz_points[i] = mel_to_hz(mel_points[i]);
    }

    // Compute center frequencies and bandwidths for each filter
    std::vector<float> f_central(n_mels_);
    std::vector<float> band(n_mels_);
    for (int i = 0; i < n_mels_; ++i) {
        f_central[i] = hz_points[i + 1];
        band[i] = hz_points[i + 1] - hz_points[i];  // Distance between consecutive points
    }

    // Create frequency axis (linear bins in Hz)
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
}

std::vector<std::vector<float>> FilterBank::apply(
    const std::vector<std::vector<float>>& magnitude,
    bool use_log,
    float amin,
    float top_db
) {
    int num_frames = magnitude.size();
    std::vector<std::vector<float>> fbank(num_frames, 
                                           std::vector<float>(n_mels_, 0.0f));
    
    // Apply filterbank matrix to magnitude spectrogram
    for (int t = 0; t < num_frames; ++t) {
        for (int m = 0; m < n_mels_; ++m) {
            float sum = 0.0f;
            for (int f = 0; f < (int)magnitude[t].size(); ++f) {
                sum += magnitude[t][f] * filterbank_matrix_[f][m];
            }
            fbank[t][m] = sum;
        }
    }
    
    // Convert to log scale if requested
    if (use_log) {
        // Use 20*log10 for amplitude spectrograms (power_spectrogram=1 in SpeechBrain)
        // Use 10*log10 for power spectrograms (power_spectrogram=2)
        float multiplier = 20.0f;  // Assuming amplitude spectrogram
        float db_multiplier = std::log10(std::max(amin, 1.0f));

        // Convert to dB scale
        float global_max_db = -std::numeric_limits<float>::infinity();
        for (int t = 0; t < num_frames; ++t) {
            for (int m = 0; m < n_mels_; ++m) {
                float val = std::max(fbank[t][m], amin);
                fbank[t][m] = multiplier * std::log10(val) -
                              multiplier * db_multiplier;
                global_max_db = std::max(global_max_db, fbank[t][m]);
            }
        }

        // Apply top_db clipping (global maximum across all time-frequency bins)
        float min_db = global_max_db - top_db;
        for (int t = 0; t < num_frames; ++t) {
            for (int m = 0; m < n_mels_; ++m) {
                fbank[t][m] = std::max(fbank[t][m], min_db);
            }
        }
    }
    
    return fbank;
}