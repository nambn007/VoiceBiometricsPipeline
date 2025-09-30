#include "audio_feature_processor.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

#ifdef USE_LIBSNDFILE
#include <sndfile.hh>
#endif

using cc_audio_processing::AudioFeatureProcessor;
using cc_audio_processing::FbankConfig;

static std::vector<float> load_wav_mono_16k(const char* path) {
#ifdef USE_LIBSNDFILE
  SndfileHandle h(path);
  if (!h || h.error()) {
    std::fprintf(stderr, "Failed to open %s\n", path);
    return {};
  }
  if (h.samplerate() != 16000) {
    std::fprintf(stderr, "Expected 16kHz, got %d\n", h.samplerate());
    return {};
  }
  const int channels = h.channels();
  const sf_count_t n = h.frames();
  std::vector<float> buf(static_cast<std::size_t>(n) * channels);
  h.readf(buf.data(), n);
  // Convert to mono if needed
  if (channels == 1) return buf;
  std::vector<float> mono(static_cast<std::size_t>(n), 0.0f);
  for (sf_count_t i = 0; i < n; ++i) {
    double sum = 0.0;
    for (int c = 0; c < channels; ++c) sum += buf[static_cast<std::size_t>(i) * channels + c];
    mono[static_cast<std::size_t>(i)] = static_cast<float>(sum / channels);
  }
  return mono;
#else
  (void)path;
  std::fprintf(stderr, "Rebuild with libsndfile to load wav files in example.\n");
  return {};
#endif
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "Usage: %s <wav-16k-mono>\n", argv[0]);
    return 1;
  }
  auto samples = load_wav_mono_16k(argv[1]);
  if (samples.empty()) return 1;

  FbankConfig cfg;
  cfg.sample_rate = 16000;
  cfg.num_bins = 40;
  cfg.frame_length_ms = 25.0f;
  cfg.frame_shift_ms = 10.0f;
  cfg.snip_edges = false;
  cfg.dither = 0.0f;

  AudioFeatureProcessor proc(cfg);
  auto feats = proc.compute_fbank_cmvn(samples.data(), samples.size());

  std::printf("frames=%zu, dim=%zu\n", feats.size(), feats.empty() ? 0 : feats[0].size());
  // Print first frame as a sanity check
  if (!feats.empty()) {
    for (std::size_t d = 0; d < feats[0].size(); ++d) {
      std::printf("%s%.6f", (d==0?"":" "), feats[0][d]);
    }
    std::printf("\n");
  }
  return 0;
}


