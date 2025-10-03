#!/bin/bash
# Setup kaldi-native-fbank for mel-spectrogram extraction

set -e

echo "📦 Installing kaldi-native-fbank..."

# Clone repository
if [ ! -d "third_party/kaldi-native-fbank" ]; then
    mkdir -p third_party
    cd third_party
    git clone https://github.com/csukuangfj/kaldi-native-fbank.git
    cd kaldi-native-fbank
    
    # Build
    mkdir -p build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make -j$(nproc)
    
    echo "✅ kaldi-native-fbank built successfully"
else
    echo "⚠️  kaldi-native-fbank already exists"
fi

echo ""
echo "📝 Next steps:"
echo "1. Update CMakeLists.txt to link kaldi-native-fbank"
echo "2. Replace mel_extractor.cc with kaldi-native-fbank API"
echo "3. Rebuild project"
