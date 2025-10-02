#pragma once

#include <vector>
#include <string>
#include <complex>

/**
 * AudioFeatureExtractor - Extract speech features matching SpeechBrain exactly
 * 
 * This implements the SpeechBrain Fbank pipeline:
 * 1. STFT (Short-Time Fourier Transform) with Hamming window
 * 2. spectral_magnitude (power spectrum)
 * 3. Filterbank (mel filterbank + log)
 * 4. mean_var_norm (sentence-level normalization, std_norm=False)
 * 
 * Matches: speechbrain/lobes/features.py Fbank class
 */
class AudioFeatureExtractor {
public:
    /**
     * Constructor
     * 
     * @param sample_rate Sample rate in Hz (default: 16000)
     * @param n_fft FFT size (default: 400)
     * @param win_length Window length in ms (default: 25)
     * @param hop_length Hop length in ms (default: 10)
     * @param n_mels Number of mel bins (default: 80)
     * @param f_min Minimum frequency in Hz (default: 0)
     * @param f_max Maximum frequency in Hz (default: 8000)
     */
    AudioFeatureExtractor(
        int sample_rate = 16000,
        int n_fft = 400,
        float win_length_ms = 25.0f,
        float hop_length_ms = 10.0f,
        int n_mels = 80,
        float f_min = 0.0f,
        float f_max = 8000.0f
    );
    
    ~AudioFeatureExtractor();
    
    /**
     * Process waveform to extract normalized mel features
     * 
     * Implements:
     * 1. compute_STFT(wav)
     * 2. spectral_magnitude(STFT)
     * 3. compute_fbanks(magnitude)
     * 4. mean_var_norm(fbanks)
     * 
     * @param waveform Input samples [n_samples], mono, normalized [-1, 1]
     * @return Normalized mel features [n_frames, n_mels]
     */
    std::vector<std::vector<float>> process(const std::vector<float>& waveform);
    
    /**
     * Process multiple audio chunks independently
     * 
     * @param audio_chunks Vector of waveforms
     * @return Concatenated normalized features
     */
    std::vector<std::vector<float>> process_chunks(
        const std::vector<std::vector<float>>& audio_chunks
    );

private:
    /**
     * Step 1: Compute STFT
     * 
     * @param wav Input waveform [n_samples]
     * @return Complex STFT [n_frames, n_fft/2+1, 2] (real, imag)
     */
    std::vector<std::vector<std::vector<std::complex<float>>>> compute_stft(
        const std::vector<float>& wav
    );
    
    /**
     * Step 2: Compute power spectrum magnitude
     * 
     * @param stft Complex STFT [n_frames, n_freqs, 2]
     * @return Power spectrum [n_frames, n_freqs]
     */
    std::vector<std::vector<float>> spectral_magnitude(
        const std::vector<std::vector<std::vector<std::complex<float>>>>& stft,
        float power = 1.0f
    );
    
    /**
     * Step 3: Apply mel filterbank
     * 
     * @param magnitude Power spectrum [n_frames, n_freqs]
     * @return Log-mel features [n_frames, n_mels]
     */
    std::vector<std::vector<float>> compute_fbanks(
        const std::vector<std::vector<float>>& magnitude
    );
    
    /**
     * Step 4: Mean-variance normalization (sentence-level, std_norm=False)
     * 
     * @param features Input features [n_frames, n_mels]
     * @return Normalized features [n_frames, n_mels]
     */
    std::vector<std::vector<float>> mean_var_norm(
        const std::vector<std::vector<float>>& features
    );
    
    /**
     * Helper: Create Hamming window
     */
    std::vector<float> create_hamming_window(int size);
    
    /**
     * Helper: Create mel filterbank matrix
     */
    void create_mel_filterbank();
    
    /**
     * Helper: Convert Hz to Mel scale
     */
    float hz_to_mel(float hz);
    
    /**
     * Helper: Convert Mel to Hz scale
     */
    float mel_to_hz(float mel);
    
    /**
     * Helper: Perform FFT on a frame
     */
    std::vector<std::complex<float>> fft(const std::vector<float>& frame);

private:
    int sample_rate_;
    int n_fft_;
    int win_length_;      // in samples
    int hop_length_;      // in samples  
    int n_mels_;
    float f_min_;
    float f_max_;
    
    bool center_;         // STFT center padding
    std::string pad_mode_; // Padding mode
    
    std::vector<float> window_;           // Hamming window
    std::vector<std::vector<float>> mel_filterbank_;  // [n_mels, n_freqs]
};

