#include "voice_bio_pipeline.h"
#include "ecapa_engine.h"
#include "mel_extractor.h"
#include "vad_engine.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

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
        16000,  // sample_rate
        400,    // n_fft
        160,    // hop_length
        80,     // n_mels
        0.0f,   // f_min
        8000.0f // f_max
    );
    std::cout << "✅ Mel Extractor initialized" << std::endl;
    
    // Initialize VAD engine if model path provided
    if (!vad_model_path.empty()) {
        vad_engine_ = std::make_unique<VadEngine>(vad_model_path, sampling_rate);
    } else {
        std::cout << "⚠️  No VAD model path provided, processing full audio" << std::endl;
    }
    
    // Initialize ECAPA engine if model path provided
    if (!ecapa_model_path.empty()) {
        ecapa_engine_ = std::make_unique<EcapaEngine>(ecapa_model_path);
    } else {
        std::cout << "⚠️  No ECAPA model path provided, using dummy embeddings" << std::endl;
    }
}

VoiceBiometricsPipeline::~VoiceBiometricsPipeline() = default;

std::vector<float> VoiceBiometricsPipeline::read_audio(const std::string& path) const {
    return read_wav_mono_16k(path);
}

std::vector<Timestamp> VoiceBiometricsPipeline::get_speech_timestamps(const std::vector<float>& wav) const {
    if (vad_engine_ && vad_engine_->is_loaded()) {
        // Use Silero VAD ONNX engine
        return vad_engine_->detect_speech(
            wav,
            0.5f,  // threshold
            static_cast<int>(min_speech_duration_sec_ * 1000),  // min_speech_duration_ms
            100    // min_silence_duration_ms
        );
    } else {
        // Fallback: return full audio as one segment
        std::vector<Timestamp> tss;
        if (wav.size() > static_cast<size_t>(0.5f * sr_)) {
            tss.push_back({0, static_cast<int>(wav.size())});
        }
        return tss;
    }
}

std::vector<std::pair<std::vector<float>, Timestamp>> VoiceBiometricsPipeline::build_chunks(
    const std::vector<float>& wav,
    const std::vector<Timestamp>& tss) const {
    std::vector<std::pair<std::vector<float>, Timestamp>> out;
    for (const auto& ts : tss) {
        int start = std::max(0, ts.start);
        int end = std::min(static_cast<int>(wav.size()), ts.end);
        if (end <= start) continue;
        float dur_sec = static_cast<float>(end - start) / static_cast<float>(sr_);
        if (dur_sec < min_speech_duration_sec_) continue;
        std::vector<float> chunk(wav.begin() + start, wav.begin() + end);
        out.emplace_back(std::move(chunk), Timestamp{start, end});
    }
    return out;
}

std::vector<float> VoiceBiometricsPipeline::encode_chunk(const std::vector<float>& wav_chunk) const {
    // Step 1: Extract mel-spectrogram features using real implementation
    auto mel_features = mel_extractor_->extract(wav_chunk);
    
    // Step 2: Run ECAPA-TDNN inference
    if (ecapa_engine_ && ecapa_engine_->is_loaded()) {
        return ecapa_engine_->compute_embedding(mel_features);
    } else {
        std::cerr << "⚠️  ECAPA engine not loaded, returning dummy embedding" << std::endl;
        return std::vector<float>(192, 0.0f);
    }
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

std::pair<std::vector<float>, std::vector<Timestamp>> VoiceBiometricsPipeline::extract_embedding(const std::string& audio_path) const {
    auto wav = read_audio(audio_path);
    auto tss = get_speech_timestamps(wav);
    auto chunks = build_chunks(wav, tss);
    std::vector<std::vector<float>> embs;
    std::vector<Timestamp> kept;
    for (const auto& ct : chunks) {
        const auto& chunk = ct.first;
        const auto& ts = ct.second;
        auto e = encode_chunk(chunk);
        if (!e.empty()) { embs.push_back(std::move(e)); kept.push_back(ts); }
    }
    if (embs.empty()) return {{}, {}};
    auto pooled = mean_pool(embs);
    return {pooled, kept};
}

float VoiceBiometricsPipeline::cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return 0.0f;
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    if (na == 0.0 || nb == 0.0) return 0.0f;
    return static_cast<float>(dot / (std::sqrt(na) * std::sqrt(nb)));
}


