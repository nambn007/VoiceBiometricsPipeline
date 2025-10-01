#include "VoiceBiometricsPipeline.h"
#include "vad_engine_2.h"
#include "ecapa_engine.h"
#include "mel_extractor.h"
#include <cmath>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <complex>

VoiceBiometricsPipeline::VoiceBiometricsPipeline(
    const std::string& vad_model_path,
    const std::string& ecapa_model_path,
    int sampling_rate
) : sampling_rate_(sampling_rate),
    min_speech_duration_(0.5f)
{
    std::cout << "Initializing Voice Biometrics Pipeline..." << std::endl;
    
    // Initialize VAD engine
    vad_engine_ = std::make_unique<VadIterator>(
        sampling_rate,      // sample_rate
        32,                 // windows_frame_size (ms)
        0.5f,               // threshold
        100,                // min_silence_duration_ms
        30,                 // speech_pad_ms
        250,                // min_speech_duration_ms
        std::numeric_limits<float>::infinity()  // max_speech_duration_s
    );
    vad_engine_->loadModel(vad_model_path);
    
    // Initialize Mel Extractor (using kaldi-native-fbank)
    mel_extractor_ = std::make_unique<MelExtractor>(
        16000,  // sample_rate
        400,    // n_fft
        160,    // hop_length
        80,     // n_mels
        0.0f,   // f_min
        8000.0f // f_max
    );
    
    // Initialize ECAPA engine
    ecapa_engine_ = std::make_unique<EcapaEngine>(ecapa_model_path);
    
    std::cout << "✅ Pipeline initialized successfully!" << std::endl;
}

VoiceBiometricsPipeline::~VoiceBiometricsPipeline() = default;

std::vector<float> VoiceBiometricsPipeline::readAudio(
    const std::string& audio_path,
    int target_sr
) {
    int original_sr = 0;
    auto waveform = AudioUtils::loadWavFile(audio_path, original_sr);
    
    if (original_sr != target_sr) {
        throw std::runtime_error(
            "Audio sample rate mismatch. Expected " + std::to_string(target_sr) + 
            "Hz, got " + std::to_string(original_sr) + "Hz. Please resample audio to 16kHz."
        );
    }
    
    return waveform;
}

std::vector<float> VoiceBiometricsPipeline::collectChunks(
    const std::vector<SpeechTimestamp>& timestamps,
    const std::vector<float>& waveform
) {
    std::vector<float> collected;
    
    for (const auto& ts : timestamps) {
        if (ts.start >= 0 && ts.end <= static_cast<int>(waveform.size())) {
            collected.insert(
                collected.end(),
                waveform.begin() + ts.start,
                waveform.begin() + ts.end
            );
        }
    }
    
    return collected;
}

std::vector<std::vector<float>> VoiceBiometricsPipeline::computeFeatures(
    const std::vector<float>& waveform
) {
    // Use MelExtractor (kaldi-native-fbank) for robust feature extraction
    return mel_extractor_->extract(waveform);
}

std::vector<std::vector<float>> VoiceBiometricsPipeline::meanVarNorm(
    const std::vector<std::vector<float>>& features
) {
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

std::vector<float> VoiceBiometricsPipeline::extractEmbedding(
    const std::string& audio_path,
    std::vector<SpeechTimestamp>& out_chunks
) {
    std::cout << "Loading audio: " << audio_path << std::endl;
    
    // Load audio
    auto waveform = readAudio(audio_path, sampling_rate_);
    
    std::cout << "Running VAD..." << std::endl;
    
    // Run VAD to get speech timestamps
    vad_engine_->reset();
    vad_engine_->process(waveform);
    auto vad_timestamps = vad_engine_->get_speech_timestamps();
    
    if (vad_timestamps.empty()) {
        std::cout << "No speech detected" << std::endl;
        return {};
    }
    
    std::cout << "Found " << vad_timestamps.size() << " speech segments" << std::endl;
    
    // Convert timestamp_t to SpeechTimestamp and filter by duration
    std::vector<SpeechTimestamp> speech_chunks;
    for (const auto& ts : vad_timestamps) {
        float duration = static_cast<float>(ts.end - ts.start) / sampling_rate_;
        if (duration >= min_speech_duration_) {
            speech_chunks.push_back({ts.start, ts.end});
        }
    }
    
    if (speech_chunks.empty()) {
        std::cout << "No speech chunks meet minimum duration requirement" << std::endl;
        return {};
    }
    
    std::cout << "Processing " << speech_chunks.size() << " valid chunks" << std::endl;
    
    // Extract embeddings from each chunk
    std::vector<std::vector<float>> embeddings;
    
    for (const auto& chunk_ts : speech_chunks) {
        // Extract audio chunk
        std::vector<float> chunk(
            waveform.begin() + chunk_ts.start,
            waveform.begin() + chunk_ts.end
        );
        
        // Compute mel-spectrogram features
        auto features = computeFeatures(chunk);
        
        if (features.empty()) {
            continue;
        }
        
        // Mean-variance normalization
        auto normalized_features = meanVarNorm(features);
        
        // Extract embedding using ECAPA-TDNN
        auto embedding = ecapa_engine_->compute_embedding(normalized_features);
        
        if (!embedding.empty()) {
            embeddings.push_back(embedding);
            out_chunks.push_back(chunk_ts);
        }
    }
    
    if (embeddings.empty()) {
        std::cout << "Failed to extract embeddings" << std::endl;
        return {};
    }
    
    std::cout << "Extracted " << embeddings.size() << " embeddings" << std::endl;
    
    // Average embeddings across all chunks
    size_t embedding_dim = embeddings[0].size();
    std::vector<float> final_embedding(embedding_dim, 0.0f);
    
    for (const auto& emb : embeddings) {
        for (size_t i = 0; i < embedding_dim; ++i) {
            final_embedding[i] += emb[i];
        }
    }
    
    for (auto& val : final_embedding) {
        val /= embeddings.size();
    }
    
    return final_embedding;
}

std::vector<float> VoiceBiometricsPipeline::extractEmbeddingFromWaveform(
    const std::vector<float>& waveform
) {
    // Run VAD
    vad_engine_->reset();
    vad_engine_->process(waveform);
    auto vad_timestamps = vad_engine_->get_speech_timestamps();
    
    if (vad_timestamps.empty()) {
        return {};
    }
    
    // Convert to SpeechTimestamp
    std::vector<SpeechTimestamp> speech_chunks;
    for (const auto& ts : vad_timestamps) {
        speech_chunks.push_back({ts.start, ts.end});
    }
    
    // Collect all speech chunks
    auto full_speech = collectChunks(speech_chunks, waveform);
    
    if (full_speech.empty()) {
        return {};
    }
    
    // Compute features
    auto features = computeFeatures(full_speech);
    
    // Normalize
    auto normalized_features = meanVarNorm(features);
    
    // Extract embedding
    return ecapa_engine_->compute_embedding(normalized_features);
}

float VoiceBiometricsPipeline::computeSimilarity(
    const std::vector<float>& embedding1,
    const std::vector<float>& embedding2
) {
    if (embedding1.size() != embedding2.size()) {
        throw std::invalid_argument("Embeddings must have the same size");
    }
    
    if (embedding1.empty()) {
        throw std::invalid_argument("Embeddings cannot be empty");
    }
    
    // Cosine similarity: dot(a, b) / (norm(a) * norm(b))
    float dot_product = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;
    
    for (size_t i = 0; i < embedding1.size(); ++i) {
        dot_product += embedding1[i] * embedding2[i];
        norm1 += embedding1[i] * embedding1[i];
        norm2 += embedding2[i] * embedding2[i];
    }
    
    return dot_product / (std::sqrt(norm1) * std::sqrt(norm2));
}

// ==================== Audio Utility Implementations ====================

namespace AudioUtils {

std::vector<float> loadWavFile(const std::string& filepath, int& sample_rate) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open WAV file: " + filepath);
    }
    
    // Read RIFF header
    char riff[4], wave[4], fmt[4], data[4];
    uint32_t chunk_size, subchunk1_size, subchunk2_size;
    uint16_t audio_format, num_channels, bits_per_sample, block_align;
    uint32_t byte_rate;
    
    file.read(riff, 4);
    if (std::string(riff, 4) != "RIFF") {
        throw std::runtime_error("Invalid WAV file: missing RIFF header");
    }
    
    file.read(reinterpret_cast<char*>(&chunk_size), 4);
    file.read(wave, 4);
    
    if (std::string(wave, 4) != "WAVE") {
        throw std::runtime_error("Invalid WAV file: missing WAVE header");
    }
    
    // Read fmt subchunk
    file.read(fmt, 4);
    file.read(reinterpret_cast<char*>(&subchunk1_size), 4);
    file.read(reinterpret_cast<char*>(&audio_format), 2);
    file.read(reinterpret_cast<char*>(&num_channels), 2);
    file.read(reinterpret_cast<char*>(&sample_rate), 4);
    file.read(reinterpret_cast<char*>(&byte_rate), 4);
    file.read(reinterpret_cast<char*>(&block_align), 2);
    file.read(reinterpret_cast<char*>(&bits_per_sample), 2);
    
    // Skip extra format bytes if present
    if (subchunk1_size > 16) {
        file.seekg(subchunk1_size - 16, std::ios::cur);
    }
    
    // Find data chunk (handle potential metadata chunks)
    bool found_data = false;
    while (!found_data && file.read(data, 4)) {
        if (std::string(data, 4) == "data") {
            found_data = true;
            file.read(reinterpret_cast<char*>(&subchunk2_size), 4);
        } else {
            // Skip this chunk
            uint32_t skip_size;
            file.read(reinterpret_cast<char*>(&skip_size), 4);
            file.seekg(skip_size, std::ios::cur);
        }
    }
    
    if (!found_data) {
        throw std::runtime_error("Invalid WAV file: data chunk not found");
    }
    
    // Read audio samples
    std::vector<float> waveform;
    size_t num_samples = subchunk2_size / (bits_per_sample / 8) / num_channels;
    waveform.reserve(num_samples);
    
    if (bits_per_sample == 16) {
        for (size_t i = 0; i < num_samples; ++i) {
            int16_t sample;
            file.read(reinterpret_cast<char*>(&sample), 2);
            waveform.push_back(static_cast<float>(sample) / 32768.0f);
            
            // Skip other channels if stereo
            if (num_channels > 1) {
                file.seekg((num_channels - 1) * 2, std::ios::cur);
            }
        }
    } else if (bits_per_sample == 32) {
        for (size_t i = 0; i < num_samples; ++i) {
            float sample;
            file.read(reinterpret_cast<char*>(&sample), 4);
            waveform.push_back(sample);
            
            // Skip other channels if stereo
            if (num_channels > 1) {
                file.seekg((num_channels - 1) * 4, std::ios::cur);
            }
        }
    } else {
        throw std::runtime_error("Unsupported bit depth: " + std::to_string(bits_per_sample));
    }
    
    return waveform;
}

std::vector<float> applyHammingWindow(const std::vector<float>& frame) {
    std::vector<float> windowed(frame.size());
    const float pi = 3.14159265358979323846f;
    
    for (size_t i = 0; i < frame.size(); ++i) {
        float window_val = 0.54f - 0.46f * std::cos(2.0f * pi * i / (frame.size() - 1));
        windowed[i] = frame[i] * window_val;
    }
    
    return windowed;
}

// Simple FFT implementation (for production, use FFTW or similar)
void computeFFT(std::vector<float>& real, std::vector<float>& imag) {
    int n = real.size();
    
    // Bit reversal
    int j = 0;
    for (int i = 0; i < n - 1; ++i) {
        if (i < j) {
            std::swap(real[i], real[j]);
            std::swap(imag[i], imag[j]);
        }
        int m = n / 2;
        while (m >= 1 && j >= m) {
            j -= m;
            m /= 2;
        }
        j += m;
    }
    
    // FFT computation
    const float pi = 3.14159265358979323846f;
    for (int s = 1; s <= std::log2(n); ++s) {
        int m = 1 << s;
        float theta = -2.0f * pi / m;
        std::complex<float> wm(std::cos(theta), std::sin(theta));
        
        for (int k = 0; k < n; k += m) {
            std::complex<float> w(1.0f, 0.0f);
            for (int j = 0; j < m / 2; ++j) {
                std::complex<float> t = w * std::complex<float>(real[k + j + m/2], imag[k + j + m/2]);
                std::complex<float> u(real[k + j], imag[k + j]);
                
                real[k + j] = (u + t).real();
                imag[k + j] = (u + t).imag();
                real[k + j + m/2] = (u - t).real();
                imag[k + j + m/2] = (u - t).imag();
                
                w *= wm;
            }
        }
    }
}

std::vector<float> powerSpectrum(const std::vector<float>& real, const std::vector<float>& imag) {
    size_t n = real.size() / 2 + 1;  // Only need positive frequencies
    std::vector<float> power(n);
    
    for (size_t i = 0; i < n; ++i) {
        power[i] = real[i] * real[i] + imag[i] * imag[i];
    }
    
    return power;
}

std::vector<std::vector<float>> createMelFilterbank(int n_mels, int n_fft, int sample_rate) {
    // Create mel filterbank matrix
    int n_freqs = n_fft / 2 + 1;
    std::vector<std::vector<float>> filterbank(n_mels, std::vector<float>(n_freqs, 0.0f));
    
    // Helper functions
    auto hz_to_mel = [](float hz) {
        return 2595.0f * std::log10(1.0f + hz / 700.0f);
    };
    
    auto mel_to_hz = [](float mel) {
        return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
    };
    
    // Create mel scale
    float mel_min = hz_to_mel(0.0f);
    float mel_max = hz_to_mel(sample_rate / 2.0f);
    
    std::vector<float> mel_points(n_mels + 2);
    for (int i = 0; i < n_mels + 2; ++i) {
        mel_points[i] = mel_min + (mel_max - mel_min) * i / (n_mels + 1);
    }
    
    // Convert back to Hz
    std::vector<float> hz_points(n_mels + 2);
    for (int i = 0; i < n_mels + 2; ++i) {
        hz_points[i] = mel_to_hz(mel_points[i]);
    }
    
    // Convert Hz to FFT bin
    std::vector<int> bin_points(n_mels + 2);
    for (int i = 0; i < n_mels + 2; ++i) {
        bin_points[i] = static_cast<int>(std::floor((n_fft + 1) * hz_points[i] / sample_rate));
    }
    
    // Create triangular filters
    for (int i = 0; i < n_mels; ++i) {
        int start = bin_points[i];
        int center = bin_points[i + 1];
        int end = bin_points[i + 2];
        
        // Rising slope
        for (int j = start; j < center; ++j) {
            if (center != start) {
                filterbank[i][j] = static_cast<float>(j - start) / (center - start);
            }
        }
        
        // Falling slope
        for (int j = center; j < end; ++j) {
            if (end != center) {
                filterbank[i][j] = static_cast<float>(end - j) / (end - center);
            }
        }
    }
    
    return filterbank;
}

std::vector<std::vector<float>> computeMelSpectrogram(
    const std::vector<float>& waveform,
    int sample_rate,
    int n_fft,
    int hop_length,
    int n_mels,
    int win_length
) {
    // Create mel filterbank
    auto mel_filters = createMelFilterbank(n_mels, n_fft, sample_rate);
    int n_freqs = n_fft / 2 + 1;
    
    // Calculate number of frames
    int num_frames = (waveform.size() - win_length) / hop_length + 1;
    if (num_frames <= 0) {
        return {};
    }
    
    std::vector<std::vector<float>> mel_spec;
    mel_spec.reserve(num_frames);
    
    // Process each frame
    for (int frame_idx = 0; frame_idx < num_frames; ++frame_idx) {
        int start = frame_idx * hop_length;
        
        // Extract frame
        std::vector<float> frame(n_fft, 0.0f);
        for (int i = 0; i < win_length && (start + i) < waveform.size(); ++i) {
            frame[i] = waveform[start + i];
        }
        
        // Apply window
        frame = applyHammingWindow(frame);
        
        // Compute FFT
        std::vector<float> real = frame;
        std::vector<float> imag(n_fft, 0.0f);
        computeFFT(real, imag);
        
        // Compute power spectrum
        auto power = powerSpectrum(real, imag);
        
        // Apply mel filterbank
        std::vector<float> mel_frame(n_mels, 0.0f);
        for (int m = 0; m < n_mels; ++m) {
            for (int f = 0; f < n_freqs; ++f) {
                mel_frame[m] += mel_filters[m][f] * power[f];
            }
        }
        
        mel_spec.push_back(mel_frame);
    }
    
    return mel_spec;
}

} // namespace AudioUtils