cc_audio_processing (FBANK + CMVN via kaldi-native-fbank)
=========================================================

This module computes Kaldi-compatible log-mel filterbank (FBANK) features and applies utterance-level mean-variance normalization (CMVN).

- Backend: kaldi-native-fbank (https://github.com/csukuangfj/kaldi-native-fbank)
- Defaults: sr=16k, win=25 ms, hop=10 ms, num_bins=40, dither=0, snip_edges=false

Build
-----

Fetch dependency automatically and build example:

```
cmake -B build -S . -DUSE_FETCHCONTENT=ON -DBUILD_EXAMPLE=ON
cmake --build build -j
```

Or use a local checkout of kaldi-native-fbank:

```
cmake -B build -S . -DKALDI_NATIVE_FBANK_DIR=/abs/path/to/kaldi-native-fbank -DBUILD_EXAMPLE=ON
cmake --build build -j
```

Usage (library)
---------------

```
#include "audio_feature_processor.h"
using namespace cc_audio_processing;

FbankConfig cfg;  // adjust if needed
AudioFeatureProcessor proc(cfg);
std::vector<float> samples = /* 16k mono float32 [-1,1] */;
auto feats = proc.compute_fbank_cmvn(samples.data(), samples.size());
// feats: [num_frames][num_bins]
```

Example
-------

If built with BUILD_EXAMPLE=ON and libsndfile is available:

```
./build/example_fbank_cmvn /path/to/16k_mono.wav
```

Notes
-----

- kaldi-native-fbank outputs log-mel fbank frames.
- CMVN here is per-utterance; adapt if your model used global/sliding CMVN.


