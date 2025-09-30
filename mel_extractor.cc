#include "mel_extractor.h"
#include <stdexcept>
#include <iostream>

// Kaldi Native Fbank headers
#include "kaldi-native-fbank/csrc/feature-fbank.h"
#include "kaldi-native-fbank/csrc/online-feature.h"

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

void MelExtractor::init_window() {
    // Not needed with kaldi-native-fbank
}

float MelExtractor::hz_to_mel(float hz) {
    return 2595.0f * std::log10(1.0f + hz / 700.0f);
}

float MelExtractor::mel_to_hz(float mel) {
    return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

void MelExtractor::init_mel_filters() {
    // Not needed with kaldi-native-fbank
}

std::vector<std::vector<std::complex<float>>> MelExtractor::compute_stft(const std::vector<float>& signal) const {
    // Not used - kaldi-native-fbank handles this internally
    return {};
}

std::vector<std::vector<float>> MelExtractor::apply_mel_filterbank(
    const std::vector<std::vector<std::complex<float>>>& stft) const {
    // Not used - kaldi-native-fbank handles this internally
    return {};
}

std::vector<std::vector<float>> MelExtractor::extract(const std::vector<float>& waveform) const {
    if (waveform.empty()) {
        throw std::runtime_error("Empty waveform");
    }
    
    using namespace knf;  // kaldi-native-fbank namespace
    
    // Configure fbank options to match SpeechBrain parameters
    FbankOptions opts;
    opts.frame_opts.samp_freq = sample_rate_;
    opts.frame_opts.dither = 0.0f;  // No dithering (like SpeechBrain)
    opts.frame_opts.frame_length_ms = (n_fft_ * 1000.0f) / sample_rate_;  // 25ms for n_fft=400
    opts.frame_opts.frame_shift_ms = (hop_length_ * 1000.0f) / sample_rate_;  // 10ms for hop_length=160
    opts.frame_opts.remove_dc_offset = false;
    opts.frame_opts.preemph_coeff = 0.0f;  // No pre-emphasis (SpeechBrain doesn't use it)
    opts.frame_opts.window_type = "hanning";  // Hanning window
    opts.frame_opts.round_to_power_of_two = false;  // Use exact n_fft
    
    opts.mel_opts.num_bins = n_mels_;  // 80 mel bins
    opts.mel_opts.low_freq = f_min_;   // 0 Hz
    opts.mel_opts.high_freq = f_max_;  // 8000 Hz
    opts.mel_opts.vtln_low = 100.0f;
    opts.mel_opts.vtln_high = -500.0f;
    
    // Energy options
    opts.use_energy = false;  // Don't compute energy
    opts.use_log_fbank = true;  // Use log mel-spectrogram (like SpeechBrain)
    opts.use_power = true;  // Use power spectrum
    
    // Create online fbank computer
    OnlineFbank fbank(opts);
    
    // Extract features
    std::vector<float> features;
    try {
        // AcceptWaveform expects (float sampling_rate, const float* data, int32_t n)
        fbank.AcceptWaveform(static_cast<float>(sample_rate_), waveform.data(), static_cast<int32_t>(waveform.size()));
        fbank.InputFinished();
        
        const int num_frames = fbank.NumFramesReady();
        const int feat_dim = fbank.Dim();  // Should be n_mels
        
        if (feat_dim != n_mels_) {
            std::cerr << "⚠️  Warning: Expected " << n_mels_ << " mel bins, got " << feat_dim << std::endl;
        }
        
        // Get all frames
        features.resize(num_frames * feat_dim);
        for (int i = 0; i < num_frames; ++i) {
            const float* frame_data = fbank.GetFrame(i);
            std::copy(frame_data, frame_data + feat_dim, features.begin() + i * feat_dim);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Kaldi fbank error: " << e.what() << std::endl;
        throw;
    }
    
    // Convert flat array to 2D vector [n_frames, n_mels]
    const int num_frames = features.size() / n_mels_;
    std::vector<std::vector<float>> mel_spec(num_frames, std::vector<float>(n_mels_));
    
    for (int t = 0; t < num_frames; ++t) {
        for (int f = 0; f < n_mels_; ++f) {
            mel_spec[t][f] = features[t * n_mels_ + f];
        }
    }
    
    return mel_spec;
}