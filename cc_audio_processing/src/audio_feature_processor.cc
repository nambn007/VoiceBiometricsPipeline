#include "audio_feature_processor.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "kaldi-native-fbank/csrc/online-feature.h"

namespace cc_audio_processing {

namespace {
inline knf::FbankOptions ToKnf(const FbankConfig& c) {
  knf::FbankOptions o;
  o.frame_opts.samp_freq = static_cast<float>(c.sample_rate);
  o.frame_opts.frame_length_ms = c.frame_length_ms;
  o.frame_opts.frame_shift_ms = c.frame_shift_ms;
  o.frame_opts.snip_edges = c.snip_edges;
  o.frame_opts.dither = c.dither;
  // Align with SpeechBrain STFT defaults
  o.frame_opts.preemph_coeff = 0.0f;        // no pre-emphasis
  o.frame_opts.window_type = "hann";        // SpeechBrain uses Hann window
  o.frame_opts.remove_dc_offset = false;    // do not remove DC by default
  o.mel_opts.num_bins = c.num_bins;
  // Use Slaney mel, power spectrogram, and log-mel after mel filtering
  o.use_power = true;
  o.use_log_fbank = true;
  return o;
}
}  // namespace

AudioFeatureProcessor::AudioFeatureProcessor(const FbankConfig& config)
    : cfg_(config) {}

std::vector<std::vector<float>> AudioFeatureProcessor::compute_fbank(
    const float* samples, std::size_t num_samples) const {
  if (samples == nullptr || num_samples == 0) {
    return {};
  }

  knf::OnlineFbank fbank(ToKnf(cfg_));
  std::vector<float> buf(samples, samples + num_samples);
  // Emulate center=True in SpeechBrain STFT by zero-padding left/right
  const int32_t win_samples = static_cast<int32_t>(
      std::lround(static_cast<double>(cfg_.sample_rate) * (cfg_.frame_length_ms / 1000.0f)));
  const int32_t pad = win_samples / 2;
  if (pad > 0) {
    std::vector<float> padded;
    padded.resize(static_cast<std::size_t>(pad) + buf.size() + static_cast<std::size_t>(pad), 0.0f);
    // copy original in the middle
    std::copy(buf.begin(), buf.end(), padded.begin() + pad);
    buf.swap(padded);
  }
  fbank.AcceptWaveform(static_cast<float>(cfg_.sample_rate), buf.data(),
                       static_cast<int32_t>(buf.size()));
  fbank.InputFinished();

  const int32_t T = fbank.NumFramesReady();
  const int32_t D = fbank.Dim();
  std::vector<std::vector<float>> feats;
  feats.reserve(static_cast<std::size_t>(T));
  for (int32_t t = 0; t < T; ++t) {
    const float* frame_ptr = fbank.GetFrame(t);
    std::vector<float> frame(static_cast<std::size_t>(D));
    std::copy(frame_ptr, frame_ptr + D, frame.begin());
    feats.emplace_back(std::move(frame));
  }
  return feats;
}

void AudioFeatureProcessor::apply_cmvn(
    std::vector<std::vector<float>>& feats) {
  if (feats.empty()) return;
  const std::size_t T = feats.size();
  const std::size_t D = feats[0].size();
  std::vector<double> mean(D, 0.0), var(D, 0.0);
  for (const auto& f : feats) {
    for (std::size_t d = 0; d < D; ++d) mean[d] += f[d];
  }
  for (std::size_t d = 0; d < D; ++d) mean[d] /= static_cast<double>(T);
  for (const auto& f : feats) {
    for (std::size_t d = 0; d < D; ++d) {
      const double diff = static_cast<double>(f[d]) - mean[d];
      var[d] += diff * diff;
    }
  }
  for (std::size_t d = 0; d < D; ++d) {
    var[d] = std::sqrt(var[d] / std::max<std::size_t>(T, 1));
    if (var[d] < 1e-10) var[d] = 1e-10;  // epsilon
  }
  for (auto& f : feats) {
    for (std::size_t d = 0; d < D; ++d) {
      f[d] = static_cast<float>((static_cast<double>(f[d]) - mean[d]) / var[d]);
    }
  }
}

std::vector<std::vector<float>> AudioFeatureProcessor::compute_fbank_cmvn(
    const float* samples, std::size_t num_samples) const {
  auto feats = compute_fbank(samples, num_samples);
  apply_cmvn(feats);
  return feats;
}

}  // namespace cc_audio_processing


