import torch 
import math 

from speechbrain.processing.features import Filterbank

torch.set_printoptions(threshold=torch.inf)

# Nếu muốn in đẹp hơn, có thể thêm các tuỳ chọn:
torch.set_printoptions(
    precision=6,       # số chữ số thập phân
    sci_mode=False,    # tắt chế độ scientific notation
    linewidth=200      # độ rộng mỗi dòng khi in
)


# compute_fbanks = Filterbank(n_mels=80)

# # inputs = torch.randn([10, 101, 201])
# # inputs = torch.zeros(1, 101, 201)
# inputs = torch.full((1, 101, 201), 2, dtype=torch.float32)
# features = compute_fbanks(inputs)

# print(features.shape)
# print(features)


from speechbrain.processing.features import STFT
from speechbrain.lobes.features import spectral_magnitude

compute_STFT = STFT(
    sample_rate=16000,              # opts.frame_opts.samp_freq
    win_length=25,                  # opts.frame_opts.frame_length_ms (25ms)
    hop_length=10,                  # opts.frame_opts.frame_shift_ms (10ms)
    n_fft=400,                      # FFT size
    window_fn=torch.hamming_window, # opts.frame_opts.window_type = "hamming"
    normalized_stft=False,
    center=True,                    # opts.frame_opts.snip_edges = false
    pad_mode="constant",
    onesided=True
)

audio_samples = torch.full((1, 10), 2, dtype=torch.float32)
stft_output = compute_STFT(audio_samples)
magnitude = spectral_magnitude(stft_output)

print(stft_output.shape)
print(stft_output)

print(magnitude.shape)
print(magnitude)