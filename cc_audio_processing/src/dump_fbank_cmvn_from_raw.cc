#include "audio_feature_processor.h"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using cc_audio_processing::AudioFeatureProcessor;
using cc_audio_processing::FbankConfig;

static std::vector<float> load_txt_floats(const char* path) {
  std::ifstream ifs(path);
  if (!ifs) {
    std::fprintf(stderr, "Failed to open %s\n", path);
    return {};
  }
  std::vector<float> v;
  std::string s;
  while (ifs >> s) {
    try {
      v.push_back(std::stof(s));
    } catch (...) {
      // ignore non-numeric tokens
    }
  }
  return v;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "Usage: %s <raw_f32_txt> <out.csv>\n", argv[0]);
    return 1;
  }
  auto samples = load_txt_floats(argv[1]);
  if (samples.empty()) return 1;

  FbankConfig cfg;
  cfg.sample_rate = 16000;
  cfg.num_bins = 40;  // or 80
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


