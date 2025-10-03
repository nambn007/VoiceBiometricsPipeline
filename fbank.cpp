#include "fbank.h"
#include <exception>
#include <stdexcept>
#include <iostream>


// Kaldi Native Fbank headers
#include "kaldi-native-fbank/csrc/feature-fbank.h"
#include "kaldi-native-fbank/csrc/feature-window.h"
#include "kaldi-native-fbank/csrc/online-feature.h"


// Hàm tạo Hamming window

Fbank::Fbank(
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

    using namespace knf;
    
    FbankOptions opts;
    opts.frame_opts.samp_freq = sample_rate_;
    opts.frame_opts.dither = 0.0f;
    opts.frame_opts.frame_length_ms = win_length_;
    opts.frame_opts.frame_shift_ms = hop_length_;

    opts.frame_opts.remove_dc_offset = false;
    opts.frame_opts.preemph_coeff = 0.0f;  // No pre-emphasis (SpeechBrain doesn't use it)
    opts.frame_opts.window_type = "hamming";  // Hanning window
    opts.frame_opts.round_to_power_of_two = false;  // Use exact n_fft

    opts.mel_opts.num_bins = n_mels_;  // 80 mel bins
    opts.mel_opts.low_freq = f_min_;   // 0 Hz
    opts.mel_opts.high_freq = f_max_;  // 8000 Hz

    opts.use_energy = false;
    opts.use_power = true;
    opts.use_log_fbank = true;
    
    online_fbank_ = std::make_unique<knf::OnlineFbank>(opts);
}


float Fbank::hz_to_mel(float hz) {
    return 2595.0f * std::log10(1.0f + hz / 700.0f);
}

float Fbank::mel_to_hz(float mel) {
    return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

std::vector<std::vector<float>> Fbank::extract(const std::vector<float>& waveform) {
    if (waveform.empty()) {
        return {};
    }

    auto stft = stft_processor_->compute(waveform);
    auto magnitude = compute_magnitude(stft);
    auto fbanks = filterbank_->apply(magnitude, true, 1e-10f, 80.0f);
    auto feats = mean_var_norm(fbanks);

    return feats;
}

std::vector<std::vector<float>> Fbank::kaldi_extract(const std::vector<std::vector<float>>& waveforms)
{
    std::vector<std::vector<float>> features;

    try {

        for (const auto &waveform : waveforms) {
            online_fbank_->AcceptWaveform(static_cast<float>(sample_rate_), waveform.data(), static_cast<int32_t>(waveform.size()));
        }

        online_fbank_->InputFinished();

        const int num_frames = online_fbank_->NumFramesReady();
        const int feat_dim = online_fbank_->Dim();  // Should be n_mels_

        if (feat_dim != n_mels_) {
            std::cerr << "Warning: Expected " << n_mels_ << " mel bins, got " << feat_dim << std::endl;
        }

        // Get all frames;
        features.reserve(num_frames);

        for (int i = 0; i < num_frames; i++) {
            const float *frame = online_fbank_->GetFrame(i);
            features.emplace_back(frame, frame + n_mels_);
        }

        features = mean_var_norm(features);
        return features;
    } catch (const std::exception &e) {
        std::cerr << "Kaldi fbank error: " << e.what() << std::endl;
        return {};
    }

}


std::vector<std::vector<float>> Fbank::kaldi_extract(const std::vector<float>& waveform)
{
    std::vector<std::vector<float>> features;

    try {

        online_fbank_->AcceptWaveform(static_cast<float>(sample_rate_), waveform.data(), static_cast<int32_t>(waveform.size()));
        online_fbank_->InputFinished();

        const int num_frames = online_fbank_->NumFramesReady();
        const int feat_dim = online_fbank_->Dim();  // Should be n_mels_

        if (feat_dim != n_mels_) {
            std::cerr << "Warning: Expected " << n_mels_ << " mel bins, got " << feat_dim << std::endl;
        }

        // Get all frames;
        features.reserve(num_frames);

        for (int i = 0; i < num_frames; i++) {
            const float *frame = online_fbank_->GetFrame(i);
            features.emplace_back(frame, frame + n_mels_);
        }

        features = mean_var_norm(features);
        return features;
    } catch (const std::exception &e) {
        std::cerr << "Kaldi fbank error: " << e.what() << std::endl;
        return {};
    }

}
