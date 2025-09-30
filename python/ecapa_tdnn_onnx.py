#!/usr/bin/env python3
"""
Export SpeechBrain ECAPA-TDNN (small) speaker embedding model to ONNX.

This script loads the pretrained SpeechBrain encoder and exports a pure
embedding graph with dynamic batch and time dimensions.

Requirements:
  - torch
  - speechbrain

Example:
  python python/ecapa_tdnn_onnx.py --out ecapa_tdnn_small.onnx --opset 13
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from speechbrain.pretrained import EncoderClassifier
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "speechbrain must be installed: pip install speechbrain"
    ) from exc


DEFAULT_REPO = "speechbrain/spkrec-ecapa-voxceleb"


class ECAPAEmbeddingWrapper(nn.Module):
    """Wrapper that consumes precomputed features and exposes `mods.embedding_model`.

    Input:  features tensor [batch, frames, feat_dim] (e.g., fbank)
    Output: L2-normalized embeddings [batch, embedding_dim]
    """

    def __init__(self, classifier: EncoderClassifier):
        super().__init__()
        self.mods = classifier.mods

    def forward(self, features: torch.Tensor) -> torch.Tensor:  # [B, T, F]
        batch_size = features.shape[0]
        lengths = torch.ones(batch_size, device=features.device)
        embeddings = self.mods.embedding_model(features, lengths)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


def export_onnx(
    out_path: Path,
    hf_repo: str = DEFAULT_REPO,
    opset: int = 13,
    frames: int = 200,
    feat_dim: int = 80,
) -> None:
    # Load pretrained classifier
    classifier = EncoderClassifier.from_hparams(source=hf_repo, run_opts={"device": "cpu"})
    classifier.eval()

    model = ECAPAEmbeddingWrapper(classifier)
    model.eval()

    # Prepare dummy features [B, T, F], dynamic in batch and time
    frames = max(1, int(frames))
    feat_dim = max(1, int(feat_dim))
    dummy = torch.zeros(1, frames, feat_dim, dtype=torch.float32)

    input_names = ["features"]
    output_names = ["embedding"]
    dynamic_axes = {
        "features": {0: "batch", 1: "time"},
        "embedding": {0: "batch"},
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    print(f"Exported ONNX model to: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export ECAPA-TDNN embedding model (features-in) to ONNX")
    parser.add_argument("--out", type=Path, required=True, help="Output ONNX file path")
    parser.add_argument("--hf-repo", type=str, default=DEFAULT_REPO, help="HuggingFace repo or local path")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version")
    parser.add_argument("--frames", type=int, default=200, help="Dummy frames (time) for export")
    parser.add_argument("--feat-dim", type=int, default=80, help="Feature dimension (e.g., fbank bins)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_onnx(
        out_path=args.out,
        hf_repo=args.hf_repo,
        opset=args.opset,
        frames=args.frames,
        feat_dim=args.feat_dim,
    )


