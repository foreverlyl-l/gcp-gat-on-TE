from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from ppgcn.tep_metadata import edge_physical_meaning, edge_physical_meaning_zh, variable_info, variable_name


def write_markdown_table(path: Path, headers: list[str], rows: list[dict]) -> None:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def representative_samples(x: np.ndarray, y: np.ndarray, faults: tuple[int, ...]) -> dict[int, torch.Tensor]:
    samples = {}
    for label, _fault in enumerate(faults, start=1):
        idx = np.flatnonzero(y == label)
        if len(idx) > 0:
            samples[label] = torch.from_numpy(x[idx[:1]])
    return samples


def export_ppgcn_node_contributions(
    model,
    x_ref: np.ndarray,
    y_ref: np.ndarray,
    faults: tuple[int, ...],
    output_dir: Path,
    *,
    device: str,
    top_k: int = 5,
) -> None:
    model.eval()
    rows = []
    for label, sample in representative_samples(x_ref, y_ref, faults).items():
        fault = faults[label - 1]
        sample = sample.to(device)
        scores = model.path_contributions(sample, label)
        path = model.path_indices[fault]
        order = np.argsort(scores)[::-1][:top_k]
        for rank, pos in enumerate(order, start=1):
            node = path[int(pos)]
            info = variable_info(node)
            rows.append(
                {
                    "fault": f"fault_{fault}",
                    "rank": rank,
                    "variable": info["name"],
                    "node_index": node,
                    "physical_description": info["description"],
                    "physical_description_zh": info["description_zh"],
                    "area": info["area"],
                    "area_zh": info["area_zh"],
                    "contribution": round(float(scores[int(pos)]), 6),
                    "meaning": "higher activation on this fault propagation path",
                }
            )
    write_csv(output_dir / "ppgcn_node_contributions.csv", rows)
    write_markdown_table(
        output_dir / "ppgcn_node_contributions.md",
        ["fault", "rank", "variable", "physical_description_zh", "area_zh", "contribution", "meaning"],
        rows,
    )


def export_path_gat_edge_attention(
    model,
    x_ref: np.ndarray,
    y_ref: np.ndarray,
    faults: tuple[int, ...],
    output_dir: Path,
    *,
    device: str,
    top_k: int = 8,
) -> None:
    model.eval()
    rows = []
    for label, sample in representative_samples(x_ref, y_ref, faults).items():
        fault = faults[label - 1]
        sample = sample.to(device)
        edge_rows = model.edge_attention_summary(sample, label)
        edge_rows = sorted(edge_rows, key=lambda item: item["attention_mean"], reverse=True)[:top_k]
        for rank, item in enumerate(edge_rows, start=1):
            source = int(item["source_index"])
            target = int(item["target_index"])
            source_info = variable_info(source)
            target_info = variable_info(target)
            rows.append(
                {
                    "fault": f"fault_{fault}",
                    "rank": rank,
                    "layer": int(item["layer"]),
                    "edge": f"{source_info['name']} -> {target_info['name']}",
                    "source": source_info["name"],
                    "source_physical": source_info["description"],
                    "source_physical_zh": source_info["description_zh"],
                    "source_area": source_info["area"],
                    "source_area_zh": source_info["area_zh"],
                    "target": target_info["name"],
                    "target_physical": target_info["description"],
                    "target_physical_zh": target_info["description_zh"],
                    "target_area": target_info["area"],
                    "target_area_zh": target_info["area_zh"],
                    "attention_mean": round(float(item["attention_mean"]), 6),
                    "attention_forward": round(float(item["attention_forward"]), 6),
                    "attention_reverse": round(float(item["attention_reverse"]), 6),
                    "physical_meaning": edge_physical_meaning(source, target),
                    "physical_meaning_zh": edge_physical_meaning_zh(source, target),
                    "meaning": "model evidence on a candidate propagation edge, not causal proof",
                }
            )
    write_csv(output_dir / "path_gat_edge_attention.csv", rows)
    write_markdown_table(
        output_dir / "path_gat_edge_attention.md",
        [
            "fault",
            "rank",
            "layer",
            "edge",
            "source_physical_zh",
            "target_physical_zh",
            "attention_mean",
            "physical_meaning_zh",
            "attention_forward",
            "attention_reverse",
            "meaning",
        ],
        rows,
    )
