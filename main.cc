#include "voice_bio_pipeline.h"

#include <iostream>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: voice_bio <wav1.wav> <wav2.wav> [ecapa_model.onnx] [vad_model.onnx]" << std::endl;
        std::cerr << "Example: ./voice_bio pair1/1.wav pair1/2.wav ecapa-tdnn-small.onnx silero_vad.onnx" << std::endl;
        return 1;
    }

    std::string ecapa_model_path = "";
    std::string vad_model_path = "";
    
    if (argc >= 4) {
        ecapa_model_path = argv[3];
    }
    if (argc >= 5) {
        vad_model_path = argv[4];
    }

    std::cout << "Initializing Voice Biometrics Pipeline..." << std::endl;
    std::cout << ecapa_model_path << " " << vad_model_path << std::endl;
    VoiceBiometricsPipeline pipeline(16000, ecapa_model_path, vad_model_path);
    
    std::cout << "\nEnrolling speaker from: " << argv[1] << std::endl;
    auto [emb1, chunks1] = pipeline.extract_embedding(argv[1]);
    
    std::cout << "\nVerifying speaker from: " << argv[2] << std::endl;
    auto [emb2, chunks2] = pipeline.extract_embedding(argv[2]);

    if (emb1.empty() || emb2.empty()) {
        std::cerr << "❌ Failed to extract embeddings" << std::endl;
        return 2;
    }

    std::cout << "\nEmbedding dimensions: " << emb1.size() << std::endl;
    float sim = VoiceBiometricsPipeline::cosine_similarity(emb1, emb2);
    std::cout << "Similarity score: " << sim << std::endl;
    std::cout << (sim >= 0.55f ? "✅ The speakers are the same" : "❌ The speakers are different") << std::endl;
    return 0;
}


