import argparse
import csv
import os
import subprocess

import torch
import torchaudio

from speechbrain.lobes.features import Fbank
from speechbrain.processing.features import InputNormalization


def save_samples_txt(waveform: torch.Tensor, path: str):
    # waveform: [1, T]
    data = waveform.squeeze(0).cpu().numpy().tolist()
    with open(path, "w") as f:
        for v in data:
            f.write(f"{v}\n")


def save_feats_csv(feats: torch.Tensor, path: str):
    # feats: [B, Frames, Dim], B=1
    arr = feats.squeeze(0).cpu().numpy()
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in arr:
            writer.writerow([float(x) for x in row])


def load_csv(path: str):
    rows = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        for r in reader:
            if not r:
                continue
            rows.append([float(x) for x in r])
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wav", type=str)
    parser.add_argument("cc_build_dir", type=str, help="Path to cc_audio_processing/build")
    parser.add_argument("num_bins", type=int, default=40, nargs="?")
    args = parser.parse_args()

    device = torch.device("cpu")
    wav, fs = torchaudio.load(args.wav)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    target_fs = 16000
    if fs != target_fs:
        wav = torchaudio.functional.resample(wav, fs, target_fs)

    # Save samples for C++ raw dumper
    out_dir = os.path.join(args.cc_build_dir)
    os.makedirs(out_dir, exist_ok=True)
    samples_txt = os.path.join(out_dir, "samples.txt")
    save_samples_txt(wav, samples_txt)

    # SpeechBrain Fbank + CMVN
    fbank = Fbank(sample_rate=target_fs, n_mels=args.num_bins, win_length=25, hop_length=10)
    feats = fbank(wav)
    cmvn = InputNormalization()
    wav_lens = torch.tensor([1.0])
    feats_cmvn = cmvn(feats, wav_lens)

    sb_csv = os.path.join(out_dir, "sb_feats.csv")
    save_feats_csv(feats_cmvn, sb_csv)

    # Run C++ raw dumper
    cpp_csv = os.path.join(out_dir, "cpp_feats.csv")
    bin_path = os.path.join(args.cc_build_dir, "dump_fbank_cmvn_from_raw")
    subprocess.check_call([bin_path, samples_txt, cpp_csv])

    # Compare
    sb = load_csv(sb_csv)
    cpp = load_csv(cpp_csv)

    if len(sb) != len(cpp):
        print(f"Frame mismatch: sb={len(sb)} cpp={len(cpp)}")
    n = min(len(sb), len(cpp))
    max_abs = 0.0
    mean_abs = 0.0
    count = 0
    for i in range(n):
        if len(sb[i]) != len(cpp[i]):
            print(f"Dim mismatch at frame {i}: sb={len(sb[i])} cpp={len(cpp[i])}")
            m = min(len(sb[i]), len(cpp[i]))
        else:
            m = len(sb[i])
        for j in range(m):
            d = abs(sb[i][j] - cpp[i][j])
            max_abs = max(max_abs, d)
            mean_abs += d
            count += 1
    mean_abs = mean_abs / max(1, count)
    print(f"Compared {n} frames, dims up to {len(sb[0]) if sb else 0}: max_abs={max_abs:.6g}, mean_abs={mean_abs:.6g}")


if __name__ == "__main__":
    main()


