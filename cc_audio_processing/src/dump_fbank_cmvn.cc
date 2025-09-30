#include "audio_feature_processor.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
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
  if (argc < 3) {
    std::fprintf(stderr, "Usage: %s <wav-16k-mono> <out.csv>\n", argv[0]);
    return 1;
  }
  auto samples = load_wav_mono_16k(argv[1]);
  if (samples.empty()) return 1;

  FbankConfig cfg;
  cfg.sample_rate = 16000;
  cfg.num_bins = 40;  // change to 80 if needed
  cfg.frame_length_ms = 25.0f;
  cfg.frame_shift_ms = 10.0f;
  cfg.snip_edges = false;
  cfg.dither = 0.0f;

  AudioFeatureProcessor proc(cfg);
  auto feats = proc.compute_fbank_cmvn(samples.data(), samples.size());

  std::ofstream ofs(argv[2]);
  if (!ofs) {
    std::fprintf(stderr, "Failed to open %s for writing\n", argv[2]);
    return 1;
  }
  for (const auto& f : feats) {
    for (std::size_t i = 0; i < f.size(); ++i) {
      if (i) ofs << ",";
      ofs << f[i];
    }
    ofs << "\n";
  }
  ofs.close();
  std::fprintf(stdout, "Wrote %zu frames x %zu dims to %s\n", feats.size(), feats.empty()?0:feats[0].size(), argv[2]);
  return 0;
}


