#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cc_audio_processing {

struct FbankConfig {
  int sample_rate = 16000;
  float frame_length_ms = 25.0f;
  float frame_shift_ms = 10.0f;
  int num_bins = 40;
  bool snip_edges = false;
  float dither = 0.0f;
};

class AudioFeatureProcessor {
 public:
  explicit AudioFeatureProcessor(const FbankConfig& config);

  // Compute log-mel fbank features (Kaldi-compatible) and apply utterance CMVN.
  // Input samples must be mono PCM float32 in range [-1, 1] at config.sample_rate.
  // Returns features as [num_frames][num_bins].
  std::vector<std::vector<float>> compute_fbank_cmvn(
      const float* samples, std::size_t num_samples) const;

  // Compute raw fbank without CMVN.
  std::vector<std::vector<float>> compute_fbank(
      const float* samples, std::size_t num_samples) const;

  // Apply per-utterance mean-variance normalization in-place.
  static void apply_cmvn(std::vector<std::vector<float>>& feats);

 private:
  FbankConfig cfg_;
};

}  // namespace cc_audio_processing


