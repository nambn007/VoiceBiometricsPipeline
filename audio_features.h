#include <vector>
#include <complex> 
#include <fftw3.h>

std::vector<float>
hamming_window(int size);

// Compute magnitude from STFT
std::vector<std::vector<float>>
compute_magnitude(
    std::vector<std::vector<std::complex<float>>> stft,
    float power = 1.0f,
    bool log_scale = false,
    float eps = 1e-14f
);

// Mean-variance normalization
std::vector<std::vector<float>> 
mean_var_norm(
    const std::vector<std::vector<float>>& features, 
    bool std_norm = false);


class STFTProcessor {
public:
    STFTProcessor(
        int win_length_samples,
        int hop_length_samples,
        int n_fft,
        bool center = true,
        const std::string& pad_mode = "constant"
    );
    
    // Destructor
    ~STFTProcessor();
    
    // Compute STFT for a signal
    std::vector<std::vector<std::complex<float>>> compute(
        const std::vector<float>& signal
    );

private:
    int win_length_samples_;
    int hop_length_samples_;
    int n_fft_;
    bool center_;
    std::string pad_mode_;
    std::vector<float> window_;
    
    // Generate Hamming window
    std::vector<float> createHammingWindow(int window_length);
    
    // Apply padding to signal
    std::vector<float> applyPadding(const std::vector<float>& signal);
};


class FilterBank {
public:
    FilterBank(int n_mels, int n_fft, int sample_rate, 
                float f_min = 0.0f, float f_max = 8000.0f);
    
    // Apply filterbank to magnitude spectrogram
    // Returns mel-scale features (log or linear)
    std::vector<std::vector<float>> apply(
        const std::vector<std::vector<float>>& magnitude,
        bool use_log = true,
        float amin = 1e-10f,
        float top_db = 80.0f
    );
    
private:
    int n_mels_;
    int n_fft_;
    int sample_rate_;
    float f_min_;
    float f_max_;
    std::vector<std::vector<float>> filterbank_matrix_;
    
    // Convert Hz to Mel scale
    float hz_to_mel(float hz);
    
    // Convert Mel to Hz scale
    float mel_to_hz(float mel);
    
    // Create triangular mel filterbank
    void create_filterbank();
};
    