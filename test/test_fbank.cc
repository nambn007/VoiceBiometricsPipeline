#include "mel_extractor.h"
#include <iostream>

int main() {
    // MelFilterbank mel_filterbank(80, 400, 16000, 0.0f, 8000.0f);

    // std::vector<std::vector<float>> magnitude;

    // for (int i = 0; i < 101; i++) {
    //     std::vector<float> magnitude_frame;
    //     for (int j = 0; j < 201; j++) {
    //         magnitude_frame.push_back(2);
    //     }
    //     magnitude.push_back(magnitude_frame);
    // }

    // std::cout << "magnitude size: " << magnitude.size() << std::endl;
    // std::cout << "magnitude[0] size: " << magnitude[0].size() << std::endl;

    // // Match SpeechBrain Filterbank default behavior: amplitude -> log mel with 20*log10
    // std::vector<std::vector<float>> fbanks = mel_filterbank.apply(magnitude, true, 1e-10f, 80.0f);

    // for (int i = 0; i < fbanks.size(); i++) {
    //     for (int j = 0; j < fbanks[i].size(); j++) {
    //         std::cout << fbanks[i][j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    std::vector<float> waveform;
    for (int i = 0; i < 10; i++) {
        waveform.push_back(2);
    }


    auto STFT = ComputeSTFT(waveform, 25 * 16000 / 1000, 10 * 16000 / 1000, 400);
    std::cout << "STFT size: " << STFT.size() << std::endl;
    std::cout << "STFT[0] size: " << STFT[0].size() << std::endl;
    for (int i = 0; i < STFT.size(); i++) {
        for (int j = 0; j < STFT[i].size(); j++) {
            std::cout << STFT[i][j] << " ";
        }
    }


    auto magnitude = ComputeMagnitude(STFT);
    std::cout << "magnitude size: " << magnitude.size() << std::endl;
    std::cout << "magnitude[0] size: " << magnitude[0].size() << std::endl;
    for (int i = 0; i < magnitude.size(); i++) {
        for (int j = 0; j < magnitude[i].size(); j++) {
            std::cout << magnitude[i][j] << " ";
        }
    }

}