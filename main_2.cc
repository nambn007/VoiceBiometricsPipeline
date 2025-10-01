#include "VoiceBiometricsPipeline.h"
#include <iostream>
#include <iomanip>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <wav1> <wav2>" << std::endl;
        return 1;
    }

    std::string wav1 = argv[1];
    std::string wav2 = argv[2];

    try {
        // Initialize pipeline with model paths
        VoiceBiometricsPipeline pipeline(
            "/u02/libs/silero-vad/cc/models/silero_vad.onnx",
            "/u02/libs/silero-vad/cc/models/ecapa-tdnn-small.onnx",
            16000
        );

        std::cout << wav1 << "\n";
        std::cout << wav2 << "\n";

        std::cout << "Enrolling speaker..." << std::endl;
        std::vector<SpeechTimestamp> enrollment_chunks;
        auto enrollment_embedding = pipeline.extractEmbedding(wav1, enrollment_chunks);
        std::cout << "Enroll Ok\n";
        if (enrollment_embedding.empty()) {
            std::cout << "No speech detected in the enrollment audio" << std::endl;
            return 1;
        }

        std::cout << "\nVerifying speaker..." << std::endl;
        std::vector<SpeechTimestamp> test_chunks;
        auto test_embedding = pipeline.extractEmbedding(wav2, test_chunks);
        std::cout << "Enroll Ok2\n";
        if (test_embedding.empty()) {
            std::cout << "No speech detected in the test audio" << std::endl;
            return 1;
        }

        float similarity = pipeline.computeSimilarity(enrollment_embedding, test_embedding);
        
        std::cout << "\nSimilarity score: " << std::fixed << std::setprecision(4) 
                  << similarity << std::endl;
    std::cout << "Enroll Ok3\n";
        if (similarity > 0.5f) {
            std::cout << "The speakers are the same" << std::endl;
        } else {
            std::cout << "The speakers are different" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}