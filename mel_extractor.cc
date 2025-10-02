#include "mel_extractor.h"
#include <exception>
#include <stdexcept>
#include <iostream>

// Kaldi Native Fbank headers
#include "kaldi-native-fbank/csrc/feature-fbank.h"
#include "kaldi-native-fbank/csrc/feature-window.h"
#include "kaldi-native-fbank/csrc/online-feature.h"

#include <fftw3.h>

// Hàm tạo Hamming window


// Hàm tính STFT (giống SpeechBrain)
std::vector<std::vector<std::complex<float>>> ComputeSTFT(
    const std::vector<float>& signal,
    int win_length_samples,
    int hop_length_samples,
    int n_fft,
    bool center = true,
    const std::string& pad_mode = "reflect"
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

// Hàm tính magnitude từ STFT
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





MelExtractor::MelExtractor(
    int sample_rate,
    int n_fft,
    int hop_length,
    int n_mels,
    float f_min,
    float f_max
) : sample_rate_(sample_rate),
    n_fft_(n_fft),
    hop_length_(hop_length),
    n_mels_(n_mels),
    f_min_(f_min),
    f_max_(f_max) {
    
    // Note: kaldi-native-fbank doesn't need explicit initialization
    // It's configured per-call via FbankOptions
}


float MelExtractor::hz_to_mel(float hz) {
    return 2595.0f * std::log10(1.0f + hz / 700.0f);
}

float MelExtractor::mel_to_hz(float mel) {
    return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

std::vector<std::vector<float>> MelExtractor::extract(const std::vector<float>& waveform) {
    if (waveform.empty()) {
        throw std::runtime_error("Empty waveform");
    }

    int win_length = (n_fft_ * 1000.0f) / sample_rate_;
    int hop_length = (hop_length_ * 1000.0f) / sample_rate_;
    int win_length_sampels = (win_length * sample_rate_) / 1000;
    int hop_length_sampels = (hop_length * sample_rate_) / 1000;

    std::cout << win_length_sampels << " " << hop_length_sampels << " " << n_fft_ << std::endl;
    std::cout << win_length << " " << hop_length << std::endl;
    // int win_length = 25;
    // int hop_length = 10;
    
    auto stft = ComputeSTFT(waveform, win_length_sampels, hop_length_sampels, n_fft_);

    auto magnitude = ComputeMagnitude(stft);

    std::cout << n_mels_ << " " << n_fft_ << " " << sample_rate_ << " " << f_min_ << " " << f_max_ << std::endl;
    MelFilterbank mel_filterbank(n_mels_, n_fft_, sample_rate_, f_min_, f_max_);
    auto fbanks = mel_filterbank.apply(magnitude, true, 1e-10f, 80.0f);

    std::cout << "\n4. Fbanks shape: [1, " << fbanks.size() << ", " << fbanks[0].size() << "]" << std::endl;
    std::cout << "\nFirst 100 elements of Fbanks (frame 0):" << std::endl;
    for (int i = 0; i < 100; i++) {
        std::cout << fbanks[0][i] << " ";
    } std::cout << std::endl;
    std::cout << "========================================\n";


    auto feats = mean_var_norm(fbanks);
    return feats;
}

std::vector<std::vector<float>> MelExtractor::extract(const std::vector<std::vector<float>> &audio_chunks)
{
    using namespace knf;

    std::cout << "wave size: " << audio_chunks.size() << std::endl;
    std::cout << "wave 0 size: " << audio_chunks.at(0).size() << std::endl;
    
    FbankOptions opts;
    opts.frame_opts.samp_freq = sample_rate_;
    opts.frame_opts.dither = 0.0f;
    // opts.frame_opts.frame_length_ms = (n_fft_ * 1000.0f) / sample_rate_;  // 25ms for n_fft=400
    // opts.frame_opts.frame_shift_ms = (hop_length_ * 1000.0f) / sample_rate_;  // 10ms for hop_length=160
    opts.frame_opts.frame_length_ms = 25;  // 25ms for n_fft=400
    opts.frame_opts.frame_shift_ms = 10;  // 10ms for hop_length=160
    opts.frame_opts.remove_dc_offset = false;
    opts.frame_opts.preemph_coeff = 0.0f;  // No pre-emphasis (SpeechBrain doesn't use it)
    opts.frame_opts.window_type = "hamming";  // Hanning window
    opts.frame_opts.round_to_power_of_two = false;  // Use exact n_fft

    opts.mel_opts.num_bins = n_mels_;  // 80 mel bins
    opts.mel_opts.low_freq = f_min_;   // 0 Hz
    opts.mel_opts.high_freq = f_max_;  // 8000 Hz
    opts.mel_opts.vtln_low = 100.0f;
    opts.mel_opts.vtln_high = -500.0f;

    opts.use_energy = true;
    opts.use_power = true;
    opts.use_log_fbank = true;
    
    // FbankComputer fbank(opts);


    // std::vector<float> waveform;
    // for (const auto &chunk : audio_chunks) {
    //     waveform.insert(waveform.end(), chunk.begin(), chunk.end());
    // }

    // int64_t num_samples_total = waveform.size();
    // int32_t num_frames_new = NumFrames(num_samples_total, opts.frame_opts, true);

    // std::vector<std::vector<float>> features;
    
    // // OnlineFbank fbank(opts);
    // for (int32_t frame = 0; frame < num_frames_new; ++frame) {
    //     std::vector<float> window;
    //     std::fill(window.begin(), window.end(), 0);

    //     ExtractWindow(0, waveform, frame, opts.frame_opts, FeatureWindowFunction(fbank.GetFrameOptions()), &window, nullptr);
    
    //     std::vector<float> feature(fbank.Dim());
    //     fbank.Compute(0.0, 1.0 /* vtln */, &window, feature.data());
    
    //     features.push_back(feature);
    // }

    // std::cout << "A\n";
    // return features;
    
    OnlineFbank fbank(opts);
    std::vector<std::vector<float>> features;

    try {
        for (const auto &chunk : audio_chunks) {
            fbank.AcceptWaveform(static_cast<float>(sample_rate_), chunk.data(), static_cast<int32_t>(chunk.size()));
        }
        fbank.InputFinished();

        const int num_frames = fbank.NumFramesReady();
        const int feat_dim = fbank.Dim();  // Should be n_mels_

        if (feat_dim != n_mels_) {
            std::cerr << "⚠️  Warning: Expected " << n_mels_ << " mel bins, got " << feat_dim << std::endl;
        }

        // Get all frames;
        features.reserve(num_frames);

        for (int i = 0; i < num_frames; i++) {
            const float *frame = fbank.GetFrame(i);
            features.emplace_back(frame, frame + n_mels_);
        }

        return features;
    } catch (const std::exception &e) {
        std::cerr << "Kaldi fbank error: " << e.what() << std::endl;
        return {};
    }

}

std::vector<std::vector<float>> MelExtractor::mean_var_norm(const std::vector<std::vector<float>>& features) {
    if (features.empty()) return features;
    
    const size_t num_frames = features.size();
    const size_t num_features = features[0].size();
    
    // Compute mean and std across time dimension (axis=1 in Python)
    std::vector<float> mean(num_features, 0.0f);
    std::vector<float> std_dev(num_features, 0.0f);
    
    // Calculate mean
    for (const auto& frame : features) {
        for (size_t i = 0; i < num_features; ++i) {
            mean[i] += frame[i];
        }
    }
    for (auto& m : mean) {
        m /= num_frames;
    }
    
    // Calculate standard deviation
    for (const auto& frame : features) {
        for (size_t i = 0; i < num_features; ++i) {
            float diff = frame[i] - mean[i];
            std_dev[i] += diff * diff;
        }
    }
    for (auto& s : std_dev) {
        s = std::sqrt(s / num_frames + 1e-10f);
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
