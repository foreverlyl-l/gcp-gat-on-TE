import argparse
import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyreadr
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ppgcn.baselines import run_pca_svm_baseline, run_random_forest_baseline, run_svm_baseline
from ppgcn.evaluation import classification_metrics, counts
from ppgcn.explain import export_path_gat_edge_attention, export_ppgcn_node_contributions
from ppgcn.models import CNNBaseline, PropagationPathGAT, PropagationPathGCN
from ppgcn.plotting import plot_confusion_matrix
from ppgcn.training import plot_history, set_seed


@dataclass
class RiethConfig:
    seed: int = 42
    window: int = 20
    faults: tuple[int, ...] = (1, 4, 5, 6, 7, 8, 12, 13)
    train_runs: tuple[int, ...] = tuple(range(1, 19))
    val_runs: tuple[int, ...] = (19, 20)
    test_runs: tuple[int, ...] = (21, 22)
    segment_length: int = 170
    batch_size: int = 256
    epochs: int = 12
    lr: float = 1e-3
    filters: int = 20
    graph_layers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def token_to_index(token: str) -> int:
    token = token.upper()
    if token.startswith("XMV"):
        return 41 + int(token[3:]) - 1
    if token.startswith("X"):
        return int(token[1:]) - 1
    raise ValueError(f"Unsupported variable token: {token}")


def load_token_paths(path_file: Path) -> dict[int, list[int]]:
    with open(path_file, "r", encoding="utf-8") as f:
        payload = json.load(f)
    raw = payload["paths"]
    return {int(key): [token_to_index(token) for token in values] for key, values in raw.items()}


def feature_columns(df):
    return [col for col in df.columns if str(col).startswith("xmeas_") or str(col).startswith("xmv_")]


def windows_from_run(values: np.ndarray, label: int, window: int, fault: bool) -> tuple[list[np.ndarray], list[int]]:
    xs, ys = [], []
    if fault:
        # Training data faults are introduced after sample 20. Each online fault
        # sample is represented by a 20-sample window ending at fault sample t.
        for end in range(window, min(170, len(values))):
            xs.append(values[end - window + 1 : end + 1])
            ys.append(label)
    else:
        for end in range(window, min(170, len(values))):
            xs.append(values[end - window + 1 : end + 1])
            ys.append(label)
    return xs, ys


def collect_runs(df, runs, label: int, window: int, fault_number: int | None):
    xs, ys = [], []
    if fault_number is not None:
        df = df[df["faultNumber"].astype(int) == fault_number]
    cols = feature_columns(df)
    for run in runs:
        run_df = df[df["simulationRun"].astype(int) == run].sort_values("sample")
        values = run_df[cols].to_numpy(np.float32)
        wx, wy = windows_from_run(values, label, window, fault=fault_number is not None)
        xs.extend(wx)
        ys.extend(wy)
    return xs, ys


def load_rieth_dataset(cfg: RiethConfig, data_dir: Path):
    cache_name = (
        f"rieth_windows_w{cfg.window}_"
        f"faults{'-'.join(str(item) for item in cfg.faults)}_"
        f"train{'-'.join(str(item) for item in cfg.train_runs)}_"
        f"val{'-'.join(str(item) for item in cfg.val_runs)}_"
        f"test{'-'.join(str(item) for item in cfg.test_runs)}.npz"
    )
    cache_path = data_dir / cache_name
    if cache_path.exists():
        cached = np.load(cache_path)
        return (
            cached["x_train"],
            cached["y_train"],
            cached["x_val"],
            cached["y_val"],
            cached["x_test"],
            cached["y_test"],
        )

    free_path = data_dir / "TEP_FaultFree_Training.RData"
    faulty_path = data_dir / "TEP_Faulty_Training.RData"
    if not free_path.exists() or not faulty_path.exists():
        raise FileNotFoundError("Rieth RData files are missing. Download them with the Dataverse API first.")

    splits = {
        "train": (cfg.train_runs, [], []),
        "val": (cfg.val_runs, [], []),
        "test": (cfg.test_runs, [], []),
    }

    free = pyreadr.read_r(str(free_path), use_objects=["fault_free_training"])["fault_free_training"]
    for runs, xs, ys in splits.values():
        wx, wy = collect_runs(free, runs, 0, cfg.window, None)
        xs.extend(wx)
        ys.extend(wy)
    del free
    gc.collect()

    faulty = pyreadr.read_r(str(faulty_path), use_objects=["faulty_training"])["faulty_training"]
    for runs, xs, ys in splits.values():
        for label, fault in enumerate(cfg.faults, start=1):
            wx, wy = collect_runs(faulty, runs, label, cfg.window, fault)
            xs.extend(wx)
            ys.extend(wy)
    del faulty
    gc.collect()

    x_train, y_train = np.stack(splits["train"][1]).astype(np.float32), np.asarray(splits["train"][2], np.int64)
    x_val, y_val = np.stack(splits["val"][1]).astype(np.float32), np.asarray(splits["val"][2], np.int64)
    x_test, y_test = np.stack(splits["test"][1]).astype(np.float32), np.asarray(splits["test"][2], np.int64)

    mean = x_train.mean(axis=(0, 1), keepdims=True)
    std = x_train.std(axis=(0, 1), keepdims=True) + 1e-6
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std
    np.savez_compressed(
        cache_path,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
    )
    return x_train, y_train, x_val, y_val, x_test, y_test


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    for xb, yb in loader:
        logits = model(xb.to(device))
        preds.append(logits.argmax(dim=1).cpu().numpy())
        labels.append(yb.numpy())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    return {**classification_metrics(labels, preds), "preds": preds, "labels": labels}


def train_with_validation(model, train_loader, val_loader, test_loader, cfg: RiethConfig):
    model.to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    best_state, best_val, best_epoch = None, -1.0, 0
    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(cfg.device), yb.to(cfg.device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(xb)
        val = evaluate(model, val_loader, cfg.device)
        item = {"epoch": epoch, "loss": total / len(train_loader.dataset), **{k: val[k] for k in ("accuracy", "macro_f1")}}
        history.append(item)
        if val["macro_f1"] > best_val:
            best_val = val["macro_f1"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch:02d} | loss {item['loss']:.4f} | val_acc {item['accuracy']:.4f} | val_f1 {item['macro_f1']:.4f}")
    model.load_state_dict(best_state)
    test = evaluate(model, test_loader, cfg.device)
    return history, test, {"best_val_macro_f1": best_val, "best_epoch": best_epoch}


def save_diagnostic_checkpoint(
    path: Path,
    model,
    *,
    model_type: str,
    cfg: RiethConfig,
    nodes: int,
    class_names: list[str],
    paths: dict[int, list[int]],
    gat_heads: int | None = None,
) -> None:
    payload = {
        "model_type": model_type,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "nodes": int(nodes),
        "window": int(cfg.window),
        "faults": tuple(int(item) for item in cfg.faults),
        "paths": {int(k): [int(v) for v in values] for k, values in paths.items()},
        "filters": int(cfg.filters),
        "graph_layers": int(cfg.graph_layers),
        "class_names": class_names,
    }
    if gat_heads is not None:
        payload["gat_heads"] = int(gat_heads)
    torch.save(payload, path)


def main():
    parser = argparse.ArgumentParser(description="Propagation-path GCN on Rieth TEP RData")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--data-dir", type=str, default="data/rieth")
    parser.add_argument("--path-file", type=str, default="propagation_paths_rieth_tokens.json")
    parser.add_argument("--output-dir", type=str, default="outputs_ppgcn_rieth")
    parser.add_argument("--filters", type=int, default=20)
    parser.add_argument("--skip-svm", action="store_true", help="Skip the SVM baseline.")
    parser.add_argument("--skip-pca", action="store_true", help="Skip the PCA-SVM baseline.")
    parser.add_argument("--skip-rf", action="store_true", help="Skip the RandomForest baseline.")
    parser.add_argument("--skip-gat", action="store_true", help="Skip the Path-GAT model.")
    parser.add_argument("--svm-kernel", type=str, default="linear", choices=("linear", "rbf", "poly", "sigmoid"))
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--svm-max-iter", type=int, default=10000)
    parser.add_argument("--pca-components", type=int, default=128)
    parser.add_argument("--rf-estimators", type=int, default=200)
    parser.add_argument("--rf-max-depth", type=int, default=0, help="0 means no max-depth limit.")
    parser.add_argument("--gat-heads", type=int, default=2)
    parser.add_argument("--explain-top-k", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    cfg = RiethConfig(epochs=args.epochs, filters=args.filters, lr=args.lr, batch_size=args.batch_size)
    set_seed(cfg.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = load_token_paths(Path(args.path_file))

    x_train, y_train, x_val, y_val, x_test, y_test = load_rieth_dataset(cfg, Path(args.data_dir))
    class_names = ["normal"] + [f"fault_{fault}" for fault in cfg.faults]
    print(f"Using device: {cfg.device}")
    print(f"Train {x_train.shape} {counts(y_train)}")
    print(f"Val   {x_val.shape} {counts(y_val)}")
    print(f"Test  {x_test.shape} {counts(y_test)}")

    train_loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)), batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test)), batch_size=cfg.batch_size, shuffle=False)

    histories, metrics, best = {}, {}, {}

    if not args.skip_svm:
        print("\nTraining SVM baseline")
        svm_metrics, svm_preds = run_svm_baseline(
            x_train,
            y_train,
            x_test,
            y_test,
            seed=cfg.seed,
            kernel=args.svm_kernel,
            c=args.svm_c,
            max_iter=args.svm_max_iter,
        )
        metrics["SVM"] = svm_metrics
        plot_confusion_matrix(y_test, svm_preds, class_names, output_dir, "svm_confusion.png")

    if not args.skip_pca:
        print("\nTraining PCA-SVM baseline")
        pca_metrics, pca_preds = run_pca_svm_baseline(
            x_train,
            y_train,
            x_test,
            y_test,
            seed=cfg.seed,
            n_components=min(args.pca_components, x_train.reshape(x_train.shape[0], -1).shape[1]),
            c=args.svm_c,
            max_iter=args.svm_max_iter,
        )
        metrics["PCA-SVM"] = pca_metrics
        plot_confusion_matrix(y_test, pca_preds, class_names, output_dir, "pca_svm_confusion.png")

    if not args.skip_rf:
        print("\nTraining RandomForest baseline")
        rf_metrics, rf_preds = run_random_forest_baseline(
            x_train,
            y_train,
            x_test,
            y_test,
            seed=cfg.seed,
            n_estimators=args.rf_estimators,
            max_depth=args.rf_max_depth or None,
        )
        metrics["RandomForest"] = rf_metrics
        plot_confusion_matrix(y_test, rf_preds, class_names, output_dir, "random_forest_confusion.png")

    print("\nTraining CNN baseline")
    cnn = CNNBaseline(x_train.shape[-1], len(class_names), hidden=48)
    histories["CNN"], cnn_test, best["CNN"] = train_with_validation(cnn, train_loader, val_loader, test_loader, cfg)
    metrics["CNN"] = {k: v for k, v in cnn_test.items() if k not in {"preds", "labels"}}
    plot_confusion_matrix(cnn_test["labels"], cnn_test["preds"], class_names, output_dir, "cnn_confusion.png")

    print("\nTraining PP-GCN")
    ppgcn = PropagationPathGCN(x_train.shape[-1], cfg.faults, paths, filters=cfg.filters, graph_layers=cfg.graph_layers)
    histories["PP-GCN"], pp_test, best["PP-GCN"] = train_with_validation(ppgcn, train_loader, val_loader, test_loader, cfg)
    metrics["PP-GCN"] = {k: v for k, v in pp_test.items() if k not in {"preds", "labels"}}
    plot_confusion_matrix(pp_test["labels"], pp_test["preds"], class_names, output_dir, "ppgcn_confusion.png")
    save_diagnostic_checkpoint(
        output_dir / "best_ppgcn.pt",
        ppgcn,
        model_type="ppgcn",
        cfg=cfg,
        nodes=x_train.shape[-1],
        class_names=class_names,
        paths=paths,
    )
    export_ppgcn_node_contributions(
        ppgcn,
        x_test,
        y_test,
        cfg.faults,
        output_dir,
        device=cfg.device,
        top_k=args.explain_top_k,
    )

    if not args.skip_gat:
        print("\nTraining Path-GAT")
        gat = PropagationPathGAT(
            x_train.shape[-1],
            cfg.faults,
            paths,
            filters=cfg.filters,
            graph_layers=cfg.graph_layers,
            heads=args.gat_heads,
        )
        histories["Path-GAT"], gat_test, best["Path-GAT"] = train_with_validation(
            gat, train_loader, val_loader, test_loader, cfg
        )
        metrics["Path-GAT"] = {k: v for k, v in gat_test.items() if k not in {"preds", "labels"}}
        plot_confusion_matrix(gat_test["labels"], gat_test["preds"], class_names, output_dir, "path_gat_confusion.png")
        save_diagnostic_checkpoint(
            output_dir / "best_path_gat.pt",
            gat,
            model_type="path_gat",
            cfg=cfg,
            nodes=x_train.shape[-1],
            class_names=class_names,
            paths=paths,
            gat_heads=args.gat_heads,
        )
        export_path_gat_edge_attention(
            gat,
            x_test,
            y_test,
            cfg.faults,
            output_dir,
            device=cfg.device,
            top_k=args.explain_top_k,
        )

    plot_history(histories, output_dir)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": asdict(cfg),
                "classes": class_names,
                "train_shape": list(x_train.shape),
                "val_shape": list(x_val.shape),
                "test_shape": list(x_test.shape),
                "metrics": metrics,
                "best_validation": best,
                "path_file": args.path_file,
            },
            f,
            indent=2,
        )
    print("\nFinal test metrics at best validation epoch")
    print(json.dumps(metrics, indent=2))
    print(f"Artifacts saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
