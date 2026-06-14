from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ppgcn.models import PropagationPathGAT, PropagationPathGCN
from ppgcn.tep_metadata import edge_physical_meaning, edge_physical_meaning_zh, variable_info


def load_window(path: Path, *, window: int | None = None, nodes: int | None = None) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    elif path.suffix.lower() == ".csv":
        arr = np.loadtxt(path, delimiter=",", dtype=np.float32)
    else:
        raise ValueError(f"Unsupported input format: {path.suffix}. Use .npy or .csv.")

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D window [time, nodes], got shape {arr.shape}.")
    if window is not None and nodes is not None and arr.shape == (nodes, window):
        arr = arr.T
    if window is not None and arr.shape[0] != window:
        raise ValueError(f"Expected window length {window}, got shape {arr.shape}.")
    if nodes is not None and arr.shape[1] != nodes:
        raise ValueError(f"Expected {nodes} nodes, got shape {arr.shape}.")
    return arr


def build_model_from_checkpoint(payload: dict, device: str):
    model_type = payload["model_type"]
    common = {
        "nodes": int(payload["nodes"]),
        "faults": tuple(payload["faults"]),
        "paths": {int(k): v for k, v in payload["paths"].items()},
        "filters": int(payload["filters"]),
        "graph_layers": int(payload["graph_layers"]),
    }
    if model_type == "ppgcn":
        model = PropagationPathGCN(**common)
    elif model_type == "path_gat":
        model = PropagationPathGAT(**common, heads=int(payload.get("gat_heads", 2)))
    else:
        raise ValueError(f"Unsupported diagnostic model type: {model_type}")
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model


def top_node_evidence(model, x_tensor: torch.Tensor, label: int, fault: int, top_k: int) -> list[dict]:
    scores = model.path_contributions(x_tensor, label)
    path = model.path_indices[fault]
    order = np.argsort(scores)[::-1][:top_k]
    return [
        {
            "rank": rank,
            "variable": variable_info(path[int(pos)])["name"],
            "node_index": int(path[int(pos)]),
            "physical_description": variable_info(path[int(pos)])["description"],
            "physical_description_zh": variable_info(path[int(pos)])["description_zh"],
            "area": variable_info(path[int(pos)])["area"],
            "area_zh": variable_info(path[int(pos)])["area_zh"],
            "kind": variable_info(path[int(pos)])["kind"],
            "kind_zh": variable_info(path[int(pos)])["kind_zh"],
            "score": round(float(scores[int(pos)]), 6),
        }
        for rank, pos in enumerate(order, start=1)
    ]


def top_edge_evidence(model, x_tensor: torch.Tensor, label: int, top_k: int) -> list[dict]:
    rows = sorted(model.edge_attention_summary(x_tensor, label), key=lambda item: item["attention_mean"], reverse=True)
    out = []
    for rank, item in enumerate(rows[:top_k], start=1):
        source = int(item["source_index"])
        target = int(item["target_index"])
        source_info = variable_info(source)
        target_info = variable_info(target)
        out.append(
            {
                "rank": rank,
                "layer": int(item["layer"]),
                "edge": f"{source_info['name']} -> {target_info['name']}",
                "source": source_info["name"],
                "source_physical": source_info["description"],
                "source_physical_zh": source_info["description_zh"],
                "source_area": source_info["area"],
                "source_area_zh": source_info["area_zh"],
                "source_kind": source_info["kind"],
                "source_kind_zh": source_info["kind_zh"],
                "target": target_info["name"],
                "target_physical": target_info["description"],
                "target_physical_zh": target_info["description_zh"],
                "target_area": target_info["area"],
                "target_area_zh": target_info["area_zh"],
                "target_kind": target_info["kind"],
                "target_kind_zh": target_info["kind_zh"],
                "physical_meaning": edge_physical_meaning(source, target),
                "physical_meaning_zh": edge_physical_meaning_zh(source, target),
                "attention_mean": round(float(item["attention_mean"]), 6),
                "attention_forward": round(float(item["attention_forward"]), 6),
                "attention_reverse": round(float(item["attention_reverse"]), 6),
            }
        )
    return out


@torch.no_grad()
def diagnose_window(
    model,
    window: np.ndarray,
    *,
    model_type: str,
    class_names: list[str],
    faults: tuple[int, ...],
    device: str,
    top_n: int = 3,
    evidence_top_k: int = 5,
    abnormal_threshold: float = 0.5,
) -> dict:
    x_tensor = torch.from_numpy(window.astype(np.float32)).unsqueeze(0).to(device)
    logits = model(x_tensor)
    probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
    normal_probability = float(probs[0])
    abnormal_probability = float(1.0 - normal_probability)
    status = "abnormal" if abnormal_probability >= abnormal_threshold and int(probs.argmax()) != 0 else "normal"
    status_zh = "异常" if status == "abnormal" else "正常"

    fault_order = np.argsort(probs[1:])[::-1][:top_n] + 1
    candidates = []
    for label in fault_order:
        fault = faults[int(label) - 1]
        item = {
            "rank": len(candidates) + 1,
            "label_index": int(label),
            "fault": class_names[int(label)],
            "confidence": round(float(probs[int(label)]), 6),
        }
        if model_type == "ppgcn":
            item["key_variables"] = top_node_evidence(model, x_tensor, int(label), fault, evidence_top_k)
        elif model_type == "path_gat":
            item["key_edges"] = top_edge_evidence(model, x_tensor, int(label), evidence_top_k)
        candidates.append(item)

    return {
        "status": status,
        "status_zh": status_zh,
        "predicted_class": class_names[int(probs.argmax())],
        "normal_probability": round(normal_probability, 6),
        "abnormal_probability": round(abnormal_probability, 6),
        "top_faults": candidates,
        "note": "Evidence scores are model diagnostics, not strict causal proof.",
        "note_zh": "证据分数表示模型诊断依据，不等同于严格的物理因果证明。",
    }


def save_diagnosis(path: Path, diagnosis: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(diagnosis, indent=2, ensure_ascii=False), encoding="utf-8")
