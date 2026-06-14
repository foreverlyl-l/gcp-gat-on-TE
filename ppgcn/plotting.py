from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(labels, preds, names: list[str], output_dir: Path, filename: str) -> None:
    cm = confusion_matrix(labels, preds, labels=list(range(len(names))))
    plt.figure(figsize=(8.4, 7.2))
    plt.imshow(cm, cmap="Blues")
    plt.title(filename.replace("_", " ").replace(".png", ""))
    plt.colorbar()
    plt.xticks(range(len(names)), names, rotation=35, ha="right")
    plt.yticks(range(len(names)), names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm.max() * 0.55 else "black"
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=7)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=180)
    plt.close()

