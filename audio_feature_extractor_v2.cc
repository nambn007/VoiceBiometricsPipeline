#include "audio_feature_extractor_v2.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

AudioFeatureExtractor::AudioFeatureExtractor(
    int sample_rate,
    int n_fft,
    float win_length_ms,
    float hop_length_ms,
    int n_mels,
    float f_min,
    float f_max
) : sample_rate_(sample_rate),
    n_fft_(n_fft),
    n_mels_(n_mels),
    f_min_(f_min),
    f_max_(f_max),
    center_(true),
    pad_mode_("constant")
{
    // Convert ms to samples
    win_length_ = static_cast<int>(std::round((sample_rate / 1000.0f) * win_length_ms));
    hop_length_ = static_cast<int>(std::round((sample_rate / 1000.0f) * hop_length_ms));
    
    std::cout << "✅ AudioFeatureExtractor initialized:" << std::endl;
    std::cout << "   Sample rate: " << sample_rate_ << " Hz" << std::endl;
    std::cout << "   n_fft: " << n_fft_ << ", win_length: " << win_length_ 
              << " samples (" << win_length_ms << "ms)" << std::endl;
    std::cout << "   hop_length: " << hop_length_ << " samples (" << hop_length_ms << "ms)" << std::endl;
    std::cout << "   n_mels: " << n_mels_ << ", f_range: [" << f_min_ << ", " << f_max_ << "] Hz" << std::endl;
    
    // Create Hamming window
    window_ = create_hamming_window(win_length_);
    
    // Create mel filterbank matrix
    create_mel_filterbank();
}

AudioFeatureExtractor::~AudioFeatureExtractor() = default;

std::vector<float> AudioFeatureExtractor::create_hamming_window(int size) {
    // Hamming window: 0.54 - 0.46 * cos(2*pi*n / (N-1))
    std::vector<float> window(size);
    for (int i = 0; i < size; ++i) {
        window[i] = 0.54f - 0.46f * std::cos(2.0f * M_PI * i / (size - 1));
    }
    return window;
}

float AudioFeatureExtractor::hz_to_mel(float hz) {
    // Convert Hz to Mel scale
    // mel = 2595 * log10(1 + hz / 700)
    return 2595.0f * std::log10(1.0f + hz / 700.0f);
}

float AudioFeatureExtractor::mel_to_hz(float mel) {
    // Convert Mel to Hz scale
    // hz = 700 * (10^(mel / 2595) - 1)
    return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

void AudioFeatureExtractor::create_mel_filterbank() {
    // Create triangular mel filterbank matrix [n_mels, n_freqs]
    int n_freqs = n_fft_ / 2 + 1;
    mel_filterbank_.resize(n_mels_, std::vector<float>(n_freqs, 0.0f));
    
    // Convert f_min and f_max to mel scale
    float mel_min = hz_to_mel(f_min_);
    float mel_max = hz_to_mel(f_max_);
    
    // Create n_mels+2 equally spaced points in mel scale
    std::vector<float> mel_points(n_mels_ + 2);
    for (int i = 0; i < n_mels_ + 2; ++i) {
        mel_points[i] = mel_min + (mel_max - mel_min) * i / (n_mels_ + 1);
    }
    
    // Convert mel points back to Hz
    std::vector<float> hz_points(n_mels_ + 2);
    for (int i = 0; i < n_mels_ + 2; ++i) {
        hz_points[i] = mel_to_hz(mel_points[i]);
    }
    
    // Convert Hz to FFT bin indices
    std::vector<int> bin_points(n_mels_ + 2);
    for (int i = 0; i < n_mels_ + 2; ++i) {
        bin_points[i] = static_cast<int>(std::floor((n_fft_ + 1) * hz_points[i] / sample_rate_));
    }
    
    // Create triangular filters
    for (int m = 0; m < n_mels_; ++m) {
        int left_bin = bin_points[m];
        int center_bin = bin_points[m + 1];
        int right_bin = bin_points[m + 2];
        
        // Rising slope (left to center)
        for (int k = left_bin; k < center_bin; ++k) {
            if (k >= 0 && k < n_freqs && center_bin != left_bin) {
                mel_filterbank_[m][k] = static_cast<float>(k - left_bin) / (center_bin - left_bin);
            }
        }
        
        // Falling slope (center to right)
        for (int k = center_bin; k < right_bin; ++k) {
            if (k >= 0 && k < n_freqs && right_bin != center_bin) {
                mel_filterbank_[m][k] = static_cast<float>(right_bin - k) / (right_bin - center_bin);
            }
        }
    }
}

std::vector<std::complex<float>> AudioFeatureExtractor::fft(const std::vector<float>& frame) {
    // Simple Cooley-Tukey FFT implementation
    // Note: This is a basic implementation. For production, use FFTW or similar library.
    
    int n = frame.size();
    if (n <= 1) {
        std::vector<std::complex<float>> result(n);
        for (int i = 0; i < n; ++i) {
            result[i] = std::complex<float>(frame[i], 0.0f);
        }
        return result;
    }
    
    // If not power of 2, pad with zeros to next power of 2
    int n_padded = 1;
    while (n_padded < n) {
        n_padded *= 2;
    }
    
    std::vector<float> padded_frame = frame;
    if (n_padded > n) {
        padded_frame.resize(n_padded, 0.0f);
        n = n_padded;
    }
    
    // Convert to complex
    std::vector<std::complex<float>> x(n);
    for (int i = 0; i < n; ++i) {
        x[i] = std::complex<float>(padded_frame[i], 0.0f);
    }
    
    // Bit-reversal permutation
    int j = 0;
    for (int i = 0; i < n - 1; ++i) {
        if (i < j) {
            std::swap(x[i], x[j]);
        }
        int m = n / 2;
        while (m >= 1 && j >= m) {
            j -= m;
            m /= 2;
        }
        j += m;
    }
    
    // Cooley-Tukey FFT
    for (int s = 1; s <= std::log2(n); ++s) {
        int m = 1 << s;  // 2^s
        std::complex<float> wm = std::exp(std::complex<float>(0, -2.0f * M_PI / m));
        
        for (int k = 0; k < n; k += m) {
            std::complex<float> w(1.0f, 0.0f);
            for (int j = 0; j < m / 2; ++j) {
                std::complex<float> t = w * x[k + j + m / 2];
                std::complex<float> u = x[k + j];
                x[k + j] = u + t;
                x[k + j + m / 2] = u - t;
                w *= wm;
            }
        }
    }
    
    return x;
}

std::vector<std::vector<std::vector<std::complex<float>>>> AudioFeatureExtractor::compute_stft(
    const std::vector<float>& wav
) {
    // Implement STFT with center padding (matches SpeechBrain)
    
    std::vector<float> padded_wav = wav;
    
    // Apply center padding if enabled
    if (center_) {
        int pad_amount = n_fft_ / 2;
        std::vector<float> temp(pad_amount + wav.size() + pad_amount, 0.0f);
        std::copy(wav.begin(), wav.end(), temp.begin() + pad_amount);
        padded_wav = temp;
    }
    
    // Calculate number of frames
    int n_frames = (padded_wav.size() - win_length_) / hop_length_ + 1;
    int n_freqs = n_fft_ / 2 + 1;  // One-sided FFT
    
    std::vector<std::vector<std::vector<std::complex<float>>>> stft_result;
    stft_result.reserve(n_frames);
    
    for (int frame_idx = 0; frame_idx < n_frames; ++frame_idx) {
        int start = frame_idx * hop_length_;
        
        // Extract frame and apply window
        std::vector<float> frame(n_fft_, 0.0f);
        for (int i = 0; i < win_length_ && (start + i) < padded_wav.size(); ++i) {
            frame[i] = padded_wav[start + i] * window_[i];
        }
        
        // Compute FFT
        auto fft_result = fft(frame);
        
        // Take only positive frequencies (one-sided)
        std::vector<std::vector<std::complex<float>>> frame_fft(1);
        frame_fft[0].resize(n_freqs);
        for (int i = 0; i < n_freqs; ++i) {
            frame_fft[0][i] = fft_result[i];
        }
        
        stft_result.push_back(frame_fft);
    }
    
    return stft_result;
}

std::vector<std::vector<float>> AudioFeatureExtractor::spectral_magnitude(
    const std::vector<std::vector<std::vector<std::complex<float>>>>& stft,
    float power
) {
    // Compute power spectrum: magnitude^(2*power)
    // SpeechBrain uses power=1 which gives magnitude^2 (power spectrum)
    
    int n_frames = stft.size();
    if (n_frames == 0) return {};
    
    int n_freqs = stft[0][0].size();
    
    std::vector<std::vector<float>> magnitude(n_frames, std::vector<float>(n_freqs));
    
    for (int t = 0; t < n_frames; ++t) {
        for (int f = 0; f < n_freqs; ++f) {
            // magnitude = (real^2 + imag^2)^power
            float real = stft[t][0][f].real();
            float imag = stft[t][0][f].imag();
            float mag_sq = real * real + imag * imag;  // Power spectrum
            
            if (power == 1.0f) {
                magnitude[t][f] = mag_sq;
            } else if (power == 0.5f) {
                magnitude[t][f] = std::sqrt(mag_sq);  // Magnitude spectrum
            } else {
                magnitude[t][f] = std::pow(mag_sq, power);
            }
        }
    }
    
    return magnitude;
}

std::vector<std::vector<float>> AudioFeatureExtractor::compute_fbanks(
    const std::vector<std::vector<float>>& magnitude
) {
    // Apply mel filterbank and take log
    // Matches SpeechBrain Filterbank with log_mel=True
    
    int n_frames = magnitude.size();
    if (n_frames == 0) return {};
    
    std::vector<std::vector<float>> fbanks(n_frames, std::vector<float>(n_mels_));
    
    const float amin = 1e-10f;  // Minimum amplitude for numerical stability
    
    for (int t = 0; t < n_frames; ++t) {
        for (int m = 0; m < n_mels_; ++m) {
            float mel_energy = 0.0f;
            
            // Apply mel filter (dot product)
            for (size_t f = 0; f < magnitude[t].size(); ++f) {
                mel_energy += mel_filterbank_[m][f] * magnitude[t][f];
            }
            
            // Apply log (with small epsilon for stability)
            fbanks[t][m] = std::log(mel_energy + amin);
        }
    }
    
    return fbanks;
}

std::vector<std::vector<float>> AudioFeatureExtractor::mean_var_norm(
    const std::vector<std::vector<float>>& features
) {
    // Sentence-level mean normalization (std_norm=False)
    // Matches SpeechBrain InputNormalization(norm_type="sentence", std_norm=False)
    
    if (features.empty() || features[0].empty()) {
        return features;
    }
    
    int n_frames = features.size();
    int n_features = features[0].size();
    
    // Compute mean across time dimension (dim=0 in PyTorch)
    std::vector<float> mean(n_features, 0.0f);
    
    for (const auto& frame : features) {
        for (int i = 0; i < n_features; ++i) {
            mean[i] += frame[i];
        }
    }
    
    for (int i = 0; i < n_features; ++i) {
        mean[i] /= n_frames;
    }
    
    // Subtract mean (std=1.0, so no division)
    std::vector<std::vector<float>> normalized = features;
    for (auto& frame : normalized) {
        for (int i = 0; i < n_features; ++i) {
            frame[i] -= mean[i];
        }
    }
    
    return normalized;
}

std::vector<std::vector<float>> AudioFeatureExtractor::process(
    const std::vector<float>& waveform
) {
    if (waveform.empty()) {
        std::cerr << "⚠️  Empty waveform" << std::endl;
        return {};
    }
    
    // Step 1: STFT
    auto stft = compute_stft(waveform);
    
    // Step 2: Spectral magnitude (power spectrum)
    auto magnitude = spectral_magnitude(stft, 1.0f);  // power=1
    
    // Step 3: Mel filterbank + log
    auto fbanks = compute_fbanks(magnitude);
    
    // Step 4: Mean-variance normalization
    auto normalized = mean_var_norm(fbanks);
    
    return normalized;
}

std::vector<std::vector<float>> AudioFeatureExtractor::process_chunks(
    const std::vector<std::vector<float>>& audio_chunks
) {
    if (audio_chunks.empty()) {
        std::cerr << "⚠️  No audio chunks to process" << std::endl;
        return {};
    }
    
    std::vector<std::vector<float>> all_features;
    
    for (size_t i = 0; i < audio_chunks.size(); ++i) {
        const auto& chunk = audio_chunks[i];
        
        if (chunk.empty()) {
            std::cerr << "⚠️  Skipping empty chunk " << i << std::endl;
            continue;
        }
        
        // Process each chunk independently
        auto chunk_features = process(chunk);
        
        if (chunk_features.empty()) {
            std::cerr << "⚠️  Failed to extract features for chunk " << i << std::endl;
            continue;
        }
        
        // Concatenate all features
        all_features.insert(
            all_features.end(),
            chunk_features.begin(),
            chunk_features.end()
        );
    }
    
    if (all_features.empty()) {
        std::cerr << "❌ No features extracted from any chunks" << std::endl;
        return {};
    }
    
    std::cout << "✅ Processed " << audio_chunks.size() << " chunks -> "
              << all_features.size() << " frames x "
              << all_features[0].size() << " features" << std::endl;
    
    return all_features;
}

