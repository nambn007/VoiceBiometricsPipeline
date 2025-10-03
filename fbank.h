#pragma once

#include <vector>
#include <complex>
#include <iostream>
#include <fftw3.h>
#include <memory>
#include "audio_features.h"
#include "kaldi-native-fbank/csrc/online-feature.h"

/**
 * Generate features for input to the speech pipeline
 * Returns a set of features generated from the input waveforms
 */
class Fbank {
public:
    Fbank(
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
    std::vector<std::vector<float>> kaldi_extract(const std::vector<float>& waveform);
    std::vector<std::vector<float>> kaldi_extract(const std::vector<std::vector<float>>& waveforms);

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
    std::unique_ptr<knf::OnlineFbank> online_fbank_;
    
    // Mel filterbank matrix [n_mels x (n_fft/2 + 1)]
    std::vector<std::vector<float>> mel_filters_;
    
    // Hanning window
    std::vector<float> window_;
    
    // Helper functions
    static float hz_to_mel(float hz);
    static float mel_to_hz(float mel);

};
