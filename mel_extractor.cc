#include "mel_extractor.h"
#include <exception>
#include <stdexcept>
#include <iostream>


// Kaldi Native Fbank headers
#include "kaldi-native-fbank/csrc/feature-fbank.h"
#include "kaldi-native-fbank/csrc/feature-window.h"
#include "kaldi-native-fbank/csrc/online-feature.h"


// Hàm tạo Hamming window

MelExtractor::MelExtractor(
    int sample_rate,
    float f_min,
    float f_max,
    int n_fft,
    int n_mels,
    int win_length, // ms 
    int hop_length // ms 
) : sample_rate_(sample_rate),
    f_min_(f_min),
    f_max_(f_max),
    n_fft_(n_fft),
    n_mels_(n_mels),
    hop_length_(hop_length),
    win_length_(win_length) {

    int win_length_sampels = (win_length_ * sample_rate_) / 1000;
    int hop_length_sampels = (hop_length_ * sample_rate_) / 1000;
    stft_processor_ = std::make_unique<STFTProcessor>(win_length_sampels, hop_length_sampels, n_fft_);
    filterbank_ = std::make_unique<FilterBank>(n_mels_, n_fft_, sample_rate_, f_min_, f_max_);
}


float MelExtractor::hz_to_mel(float hz) {
    return 2595.0f * std::log10(1.0f + hz / 700.0f);
}

float MelExtractor::mel_to_hz(float mel) {
    return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

std::vector<std::vector<float>> MelExtractor::extract(const std::vector<float>& waveform) {
    if (waveform.empty()) {
        return {};
    }

    int win_length_sampels = (win_length_ * sample_rate_) / 1000;
    int hop_length_sampels = (hop_length_ * sample_rate_) / 1000;

    // auto stft = ComputeSTFT(waveform, 25 * 16000 / 1000, 10 * 16000 / 1000, 400);
    // auto stft = ComputeSTFT(waveform, win_length_sampels, hop_length_sampels, n_fft_);
    // auto magnitude = ComputeMagnitude(stft);
    // MelFilterbank mel_filterbank(n_mels_, n_fft_, sample_rate_, f_min_, f_max_);
    // auto fbanks = mel_filterbank.apply(magnitude, true, 1e-10f, 80.0f);
    // auto feats = mean_var_norm(fbanks);

    auto stft = stft_processor_->compute(waveform);
    auto magnitude = compute_magnitude(stft);
    auto fbanks = filterbank_->apply(magnitude, true, 1e-10f, 80.0f);
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

std::vector<std::vector<float>> MelExtractor::mean_var_norm(
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