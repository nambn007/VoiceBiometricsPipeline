# Voice Biometrics Pipeline - C++ Implementation

Clean architecture C++ implementation of voice biometrics pipeline với ONNX Runtime.

## 🏗️ Architecture

```
voice_bio_pipeline.cc    # High-level pipeline orchestration
├── ecapa_engine.cc      # ECAPA-TDNN ONNX inference engine
├── mel_extractor.cc     # Mel-spectrogram feature extraction
└── (TODO) vad_engine.cc # Silero VAD ONNX engine
```

### Design Principles

✅ **Separation of Concerns**: Mỗi engine độc lập, dễ test và thay thế  
✅ **No Circular Dependencies**: Forward declarations, clean includes  
✅ **RAII**: Smart pointers, automatic resource management  
✅ **Type Safety**: Strong typing, const correctness  

## 📦 Components

### 1. **EcapaEngine** (`ecapa_engine.h/cc`)
- Load ECAPA-TDNN ONNX model
- Input: mel-spectrogram `[time_frames, 80]`
- Output: speaker embedding `[192]`
- Thread-safe inference

### 2. **MelExtractor** (`mel_extractor.h/cc`)
- STFT computation (DFT implementation)
- Mel filterbank (80 filters, 0-8000Hz)
- Log mel-spectrogram
- **Parameters match SpeechBrain Fbank**:
  - `n_fft`: 400
  - `hop_length`: 160
  - `n_mels`: 80
  - `f_min`: 0 Hz
  - `f_max`: 8000 Hz

### 3. **VoiceBiometricsPipeline** (`voice_bio_pipeline.h/cc`)
- WAV file loading (16kHz mono PCM)
- VAD integration (stub)
- Chunk processing
- Embedding aggregation (mean pooling)
- Cosine similarity

## 🔧 Build

```bash
cd cc
cmake -S . -B build
cmake --build build -j
```

## 🚀 Usage

```bash
./build/voice_bio <enroll.wav> <verify.wav> <ecapa_model.onnx>

# Example
./build/voice_bio \
    ../examples/pipeline/pair1/1.wav \
    ../examples/pipeline/pair1/2.wav \
    ../examples/pipeline/ecapa-tdnn-small.onnx
```

### Output
```
✅ Mel Extractor initialized
✅ ECAPA Engine loaded: ecapa-tdnn-small.onnx
   Input: mel_features
   Output: embedding [192]

Enrolling speaker from: pair1/1.wav
Verifying speaker from: pair1/2.wav

Embedding dimensions: 192
Similarity score: 0.9427
✅ The speakers are the same
```

## 📊 Performance Notes

### Current Results
- Same speaker: ~0.94 similarity ✅
- Different speakers: ~0.85 similarity ⚠️

### Known Issues & TODOs

#### 🐛 Mel Extraction Accuracy
- **Issue**: DFT implementation chậm và có thể chưa chính xác 100%
- **Solution**: Integrate FFT library (FFTW, KissFFT, hoặc Intel MKL)
  ```cpp
  // TODO: Replace DFT with FFT
  #include <fftw3.h>
  fftwf_plan plan = fftwf_plan_dft_r2c_1d(n_fft, in, out, FFTW_ESTIMATE);
  ```

#### 🎯 Threshold Tuning
- Default threshold: `0.55`
- Có thể cần điều chỉnh dựa trên dataset cụ thể
- Consider using:
  - EER (Equal Error Rate) analysis
  - ROC curve để chọn threshold tối ưu

#### 🚀 VAD Integration
- Hiện tại: Stub (toàn bộ audio)
- TODO: Integrate Silero VAD ONNX
  ```cpp
  // Add vad_engine.h/cc
  class VadEngine {
      std::vector<Timestamp> detect(const std::vector<float>& wav);
  };
  ```

## 🔬 Accuracy Improvement Roadmap

### Phase 1: Mel Extraction (Priority: HIGH)
- [ ] Integrate FFTW or KissFFT
- [ ] Validate against Python SpeechBrain output
- [ ] Add unit tests with reference data

### Phase 2: Model Optimization
- [ ] Enable ONNX Runtime optimization
- [ ] Quantization (INT8) nếu cần speed
- [ ] GPU inference (CUDA provider)

### Phase 3: VAD Integration
- [ ] Load Silero VAD ONNX
- [ ] Implement speech/silence detection
- [ ] Smart chunk aggregation

### Phase 4: Production Features
- [ ] Thread pool for batch processing
- [ ] Streaming audio support
- [ ] Metric logging (latency, throughput)
- [ ] Error handling & retry logic

## 📝 Dependencies

- **ONNX Runtime**: v1.12.1+ (bundled)
- **C++17**: `std::unique_ptr`, structured bindings
- **Optional** (recommended):
  - FFTW3: Fast FFT computation
  - spdlog: Structured logging
  - Catch2: Unit testing

## 🧪 Testing

```bash
# Test same speaker
./build/voice_bio pair1/1.wav pair1/2.wav model.onnx

# Test different speakers
./build/voice_bio pair1/1.wav pair2/1.wav model.onnx
```

**Expected**:
- Same speaker: > 0.9
- Different speakers: < 0.6

**Current** (need improvement):
- Same speaker: ~0.94 ✅
- Different speakers: ~0.85 (too high)

## 📚 References

- [SpeechBrain ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
- [ONNX Runtime C++ API](https://onnxruntime.ai/docs/api/c/)
- [Mel-Spectrogram Theory](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)

## 🤝 Contributing

Improvements welcome! Priority areas:
1. FFT integration (replace DFT)
2. Mel filterbank validation
3. VAD engine implementation
4. Benchmark suite
