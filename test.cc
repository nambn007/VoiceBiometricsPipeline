#include <iostream>
#include "mel_extractor.h"

int main() {
    std::vector<std::vector<float>> audio_chunks;
    
    std::vector<float> temp;

    for (int i = 0; i < 4096; i++) {
        temp.push_back(0.5);
    }

    audio_chunks.push_back(temp);

    MelExtractor mel_extractor(16000, 400, 160, 80, 0.0f, 8000.0f);

    auto outputs = mel_extractor.extract(audio_chunks);

    std::cout << outputs.size() << std::endl;
    std::cout << outputs[0].size() << std::endl;

    for (int i = 0; i < outputs[0].size(); i++) {
        std::cout << outputs[0][i] << " ";
    }
    std::cout << std::endl;

    return 0;
}