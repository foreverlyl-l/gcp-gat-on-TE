from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def plot_history(histories, output_dir: Path) -> None:
    plt.figure(figsize=(8, 4.8))
    all_acc = []
    for name, history in histories.items():
        accs = [h["accuracy"] for h in history]
        all_acc.extend(accs)
        plt.plot([h["epoch"] for h in history], accs, label=f"{name} acc", marker="o", markersize=4)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    if all_acc:
        lo = min(all_acc)
        hi = max(all_acc)
        margin = max((hi - lo) * 0.15, 0.005)
        plt.ylim(lo - margin, min(hi + margin, 1.02))
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_accuracy.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4.8))
    for name, history in histories.items():
        plt.plot([h["epoch"] for h in history], [h["loss"] for h in history], label=f"{name} loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss.png", dpi=180)
    plt.close()
