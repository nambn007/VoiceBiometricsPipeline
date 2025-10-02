#include "mel_extractor.h"
#include <iostream>

int main() {
    MelFilterbank mel_filterbank(80, 400, 16000, 0.0f, 8000.0f);

    std::vector<std::vector<float>> magnitude;

    for (int i = 0; i < 101; i++) {
        std::vector<float> magnitude_frame;
        for (int j = 0; j < 201; j++) {
            magnitude_frame.push_back(1);
        }
        magnitude.push_back(magnitude_frame);
    }

    // Match SpeechBrain Filterbank default behavior: amplitude -> log mel with 20*log10
    std::vector<std::vector<float>> fbanks = mel_filterbank.apply(magnitude, true, 1e-10f, 80.0f);

    for (int i = 0; i < fbanks.size(); i++) {
        for (int j = 0; j < fbanks[i].size(); j++) {
            std::cout << fbanks[i][j] << " ";
        }
        std::cout << std::endl;
    }
}