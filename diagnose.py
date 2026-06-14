import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ppgcn.diagnosis import build_model_from_checkpoint, diagnose_window, load_window, save_diagnosis


def load_cached_sample(cache_path: Path, split: str, sample_index: int) -> tuple[np.ndarray, int]:
    payload = np.load(cache_path)
    x_key = f"x_{split}"
    y_key = f"y_{split}"
    if x_key not in payload or y_key not in payload:
        raise KeyError(f"Cache file must contain {x_key} and {y_key}.")
    x = payload[x_key]
    y = payload[y_key]
    if sample_index < 0 or sample_index >= len(x):
        raise IndexError(f"sample-index {sample_index} is out of range for {split} size {len(x)}.")
    return x[sample_index], int(y[sample_index])


def main():
    parser = argparse.ArgumentParser(description="Diagnose one process window with a trained PP-GCN or Path-GAT model.")
    parser.add_argument("--checkpoint", required=True, help="Path to best_ppgcn.pt or best_path_gat.pt.")
    parser.add_argument("--input", help="Single window file, .npy or .csv, shaped [time, nodes].")
    parser.add_argument("--cache", help="Optional .npz dataset cache for sampling a normalized train/val/test window.")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--evidence-top-k", type=int, default=5)
    parser.add_argument("--abnormal-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", help="Optional JSON output path.")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    payload = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    model = build_model_from_checkpoint(payload, args.device)

    true_label = None
    if args.input:
        window = load_window(Path(args.input), window=int(payload["window"]), nodes=int(payload["nodes"]))
    elif args.cache:
        window, true_label = load_cached_sample(Path(args.cache), args.split, args.sample_index)
    else:
        raise ValueError("Provide either --input or --cache.")

    result = diagnose_window(
        model,
        window,
        model_type=payload["model_type"],
        class_names=list(payload["class_names"]),
        faults=tuple(payload["faults"]),
        device=args.device,
        top_n=args.top_n,
        evidence_top_k=args.evidence_top_k,
        abnormal_threshold=args.abnormal_threshold,
    )
    result["checkpoint"] = str(checkpoint_path)
    if true_label is not None:
        result["true_label_index"] = true_label
        result["true_label"] = payload["class_names"][true_label]

    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        save_diagnosis(Path(args.output), result)


if __name__ == "__main__":
    main()
