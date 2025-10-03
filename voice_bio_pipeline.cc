#include "voice_bio_pipeline.h"
#include "ecapa_engine.h"
#include "mel_extractor.h"
#include "vad_engine.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <numeric>

namespace {

// Minimal WAV reader for 16-bit PCM mono
std::vector<float> read_wav_mono_16k(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Failed to open WAV: " + path);

    char riff[4]; f.read(riff, 4);
    if (std::string(riff, 4) != "RIFF") throw std::runtime_error("Invalid WAV (RIFF)");
    f.ignore(4); // chunk size
    char wave[4]; f.read(wave, 4);
    if (std::string(wave, 4) != "WAVE") throw std::runtime_error("Invalid WAV (WAVE)");

    uint16_t audio_format = 0, num_channels = 0; uint32_t sample_rate = 0; uint16_t bits_per_sample = 0;
    uint32_t data_size = 0;

    while (f && (!data_size)) {
        char chunk_id[4]; uint32_t chunk_size = 0; f.read(chunk_id, 4); f.read(reinterpret_cast<char*>(&chunk_size), 4);
        if (!f) break;
        std::string id(chunk_id, 4);
        if (id == "fmt ") {
            uint16_t block_align, byte_rate_low16; uint32_t byte_rate;
            f.read(reinterpret_cast<char*>(&audio_format), 2);
            f.read(reinterpret_cast<char*>(&num_channels), 2);
            f.read(reinterpret_cast<char*>(&sample_rate), 4);
            f.read(reinterpret_cast<char*>(&byte_rate), 4);
            f.read(reinterpret_cast<char*>(&block_align), 2);
            f.read(reinterpret_cast<char*>(&bits_per_sample), 2);
            f.ignore(chunk_size - 16);
        } else if (id == "data") {
            data_size = chunk_size;
            break;
        } else {
            f.ignore(chunk_size);
        }
    }

    if (audio_format != 1 || num_channels != 1 || sample_rate != 16000 || bits_per_sample != 16)
        throw std::runtime_error("Expected 16kHz mono 16-bit PCM WAV");

    std::vector<int16_t> pcm(data_size / 2);
    f.read(reinterpret_cast<char*>(pcm.data()), data_size);
    std::vector<float> wav(pcm.size());
    std::transform(pcm.begin(), pcm.end(), wav.begin(), [](int16_t v){ return static_cast<float>(v) / 32768.0f; });
    return wav;
}

} // namespace

VoiceBiometricsPipeline::VoiceBiometricsPipeline(
    int sampling_rate,
    const std::string& ecapa_model_path,
    const std::string& vad_model_path)
    : sr_(sampling_rate) {
    
    // Initialize mel extractor with SpeechBrain-compatible parameters
    mel_extractor_ = std::make_unique<MelExtractor>(
        16000,   // sample_rate
        0.0f,    // f_min
        8000.0f, // f_max
        400,     // n_fft
        80,      // n_mels
        25,      // win_length
        10       // hop_length
    );
    
    std::cout << "Mel Extractor initialized" << std::endl;
    
    // Initialize VAD engine if model path provided
    if (!vad_model_path.empty()) {
        vad_engine_2_ = std::make_unique<VadIterator>();
        vad_engine_2_->loadModel(vad_model_path);
    } else {
        std::cout << "No VAD model path provided, processing full audio" << std::endl;
    }
    
    
    // Initialize ECAPA engine if model path provided
    if (!ecapa_model_path.empty()) {
        ecapa_engine_ = std::make_unique<EcapaEngine>(ecapa_model_path);
    } else {
        std::cout << "No ECAPA model path provided, using dummy embeddings" << std::endl;
    }
}

VoiceBiometricsPipeline::~VoiceBiometricsPipeline() = default;

std::vector<float> VoiceBiometricsPipeline::read_audio(const std::string& path) const {
    return read_wav_mono_16k(path);
}

std::vector<timestamp_t> VoiceBiometricsPipeline::get_speech_timestamps(const std::vector<float>& wav) const {
    if (vad_engine_2_) {
        // Use Silero VAD ONNX engine
        std::cout << "Start using VAD Engine\n";
        vad_engine_2_->process(wav);
        auto res = vad_engine_2_->get_speech_timestamps();
        return res;
    } else {
        // Fallback: return full audio as one segment
        std::vector<timestamp_t> tss;
        if (wav.size() > static_cast<size_t>(0.5f * sr_)) {
            tss.push_back({0, static_cast<int>(wav.size())});
        }
        return tss;
    }
}

std::vector<std::vector<float>> VoiceBiometricsPipeline::build_chunks(
    const std::vector<float>& wav,
    const std::vector<timestamp_t>& tss) const {
    std::vector<std::vector<float>> out;
    for (const auto& ts : tss) {
        int start = std::max(0, ts.start);
        int end = std::min(static_cast<int>(wav.size()), ts.end);
        if (end <= start) continue;
        float dur_sec = static_cast<float>(end - start) / static_cast<float>(sr_);
        if (dur_sec < min_speech_duration_sec_) continue;
        std::vector<float> chunk(wav.begin() + start, wav.begin() + end);
        out.emplace_back(std::move(chunk));
    }
    return out;
}

std::vector<float> VoiceBiometricsPipeline::mean_pool(const std::vector<std::vector<float>>& embeddings) const {
    if (embeddings.empty()) return {};
    const int d = static_cast<int>(embeddings.front().size());
    std::vector<float> mean(d, 0.0f);
    for (const auto& e : embeddings) {
        for (int i = 0; i < d; ++i) mean[i] += e[i];
    }
    const float inv = 1.0f / static_cast<float>(embeddings.size());
    for (int i = 0; i < d; ++i) mean[i] *= inv;
    return mean;
}

std::vector<float> VoiceBiometricsPipeline::extract_embedding(const std::string& audio_path) {
    auto wav = read_audio(audio_path);
    auto tss = get_speech_timestamps(wav);
    
    // TODO Remove
    std::cout << "Found " << tss.size() << " speech segments:" << std::endl;
    for (const auto &ts : tss) {
        std::cout << "  " << ts.start << " - " << ts.end << std::endl;
    }

    auto chunks = build_chunks(wav, tss);
    // Gộp các chunks lại thành một waveform duy nhất và save ra file "speech.wav"
    // if (!chunks.empty()) {
    //     // Concatenate all chunks into a single waveform
    //     std::vector<float> speech_wav;
    //     for (const auto& chunk : chunks) {
    //         speech_wav.insert(speech_wav.end(), chunk.begin(), chunk.end());
    //     }

    //     // Write to "speech.wav" as 16-bit PCM mono, 16kHz
    //     static int count = 0;
    //     std::ofstream out("speech" + std::to_string(count) + ".wav", std::ios::binary);
    //     count++;
    //     if (out) {
    //         // WAV header
    //         int sample_rate = sr_;
    //         int num_channels = 1;
    //         int bits_per_sample = 16;
    //         int byte_rate = sample_rate * num_channels * bits_per_sample / 8;
    //         int block_align = num_channels * bits_per_sample / 8;
    //         int data_size = static_cast<int>(speech_wav.size()) * num_channels * bits_per_sample / 8;
    //         int chunk_size = 36 + data_size;

    //         // Write RIFF header
    //         out.write("RIFF", 4);
    //         out.write(reinterpret_cast<const char*>(&chunk_size), 4);
    //         out.write("WAVE", 4);

    //         // fmt subchunk
    //         out.write("fmt ", 4);
    //         int subchunk1_size = 16;
    //         short audio_format = 1;
    //         out.write(reinterpret_cast<const char*>(&subchunk1_size), 4);
    //         out.write(reinterpret_cast<const char*>(&audio_format), 2);
    //         out.write(reinterpret_cast<const char*>(&num_channels), 2);
    //         out.write(reinterpret_cast<const char*>(&sample_rate), 4);
    //         out.write(reinterpret_cast<const char*>(&byte_rate), 4);
    //         out.write(reinterpret_cast<const char*>(&block_align), 2);
    //         out.write(reinterpret_cast<const char*>(&bits_per_sample), 2);

    //         // data subchunk
    //         out.write("data", 4);
    //         out.write(reinterpret_cast<const char*>(&data_size), 4);

    //         // Write PCM data
    //         for (float sample : speech_wav) {
    //             // Clamp to [-1, 1]
    //             if (sample > 1.0f) sample = 1.0f;
    //             if (sample < -1.0f) sample = -1.0f;
    //             int16_t pcm = static_cast<int16_t>(sample * 32767.0f);
    //             out.write(reinterpret_cast<const char*>(&pcm), 2);
    //         }
    //         out.close();
    //         std::cout << "Saved concatenated speech to speech.wav (" << speech_wav.size() << " samples)" << std::endl;
    //     } else {
    //         std::cerr << "Failed to open speech.wav for writing" << std::endl;
    //     }
    // }

    if (chunks.empty()) {
        std::cerr << "No valid speech chunks found" << std::endl;
        return {};
    }
    
    std::vector<std::vector<float>> embeddings;
    for (const auto& chunk : chunks) {
        auto mel_features = mel_extractor_->extract(chunk);
        auto embedding = ecapa_engine_->compute_embedding(mel_features);
        embeddings.push_back(embedding);
    }
    
    if (embeddings.empty()) {
        std::cerr << "Failed to extract embeddings" << std::endl;
        return {};
    }
    
    // Average embeddings across all chunks (mean pooling)
    std::vector<float> final_embedding = mean_pool(embeddings);
    return final_embedding;
}

float VoiceBiometricsPipeline::cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return 0.0f;

    double dot = std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
    double na  = std::inner_product(a.begin(), a.end(), a.begin(), 0.0);
    double nb  = std::inner_product(b.begin(), b.end(), b.begin(), 0.0);

    if (na == 0.0 || nb == 0.0) return 0.0f;

    return static_cast<float>(dot / (std::sqrt(na) * std::sqrt(nb)));

}
