"""Train and checkpoint the frozen Mutagenicity GIN used by the Stage-1 pilot."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from examples.ba2motifs_experiment import load_graphs, save_split, seed_everything, stratified_split
from examples.mutagenicity_experiment import MutagenicityGIN


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = F.cross_entropy(logits, batch.y.view(-1))
            prediction = logits.argmax(dim=-1)
            correct += int((prediction == batch.y.view(-1)).sum())
            total += int(batch.y.numel())
            total_loss += float(loss) * int(batch.y.numel())
    return correct / max(total, 1), total_loss / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Mutagenicity GIN")
    parser.add_argument("--data-root", type=Path, default=Path("data/Mutagenicity"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/mutagenicity_seed0"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = TUDataset(root=str(args.data_root), name="Mutagenicity", use_node_attr=False)
    if dataset.num_features <= 0:
        raise ValueError("Mutagenicity has no usable node features in this processed dataset")
    split = stratified_split(dataset.y, args.seed, 0.8, 0.1)
    train_loader = DataLoader(load_graphs(dataset, split["train"]), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(load_graphs(dataset, split["val"]), batch_size=args.batch_size)
    test_loader = DataLoader(load_graphs(dataset, split["test"]), batch_size=args.batch_size)

    model_kwargs = {
        "in_channels": int(dataset.num_features),
        "hidden_channels": args.hidden,
        "num_classes": int(dataset.num_classes),
        "dropout": args.dropout,
    }
    model = MutagenicityGIN(**model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val = -1.0
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_correct = 0
        train_total = 0
        train_loss_sum = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = F.cross_entropy(logits, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach()) * int(batch.y.numel())
            train_correct += int((logits.argmax(dim=-1) == batch.y.view(-1)).sum())
            train_total += int(batch.y.numel())

        val_accuracy, val_loss = evaluate(model, val_loader, device)
        if val_accuracy > best_val:
            best_val = val_accuracy
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch={epoch} train_accuracy={train_correct / max(train_total, 1):.4f} "
                f"train_loss={train_loss_sum / max(train_total, 1):.4f} "
                f"val_accuracy={val_accuracy:.4f} val_loss={val_loss:.4f} best={best_val:.4f}"
            )
        if stale >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("no checkpoint state was selected")
    model.load_state_dict(best_state)
    test_accuracy, test_loss = evaluate(model, test_loader, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "model.pt"
    torch.save(
        {
            "model_state": best_state,
            "model_kwargs": model_kwargs,
            "seed": args.seed,
            "best_epoch": best_epoch,
            "val_accuracy": best_val,
            "test_accuracy": test_accuracy,
            "test_loss": test_loss,
        },
        checkpoint_path,
    )
    split_path = args.output_dir / "split.json"
    label_counts = dataset.y.detach().cpu().view(-1).bincount().tolist()
    save_split(
        split_path,
        split,
        {
            "dataset": "Mutagenicity",
            "dataset_class": "torch_geometric.datasets.TUDataset",
            "seed": args.seed,
            "train_fraction": 0.8,
            "val_fraction": 0.1,
            "num_graphs": len(dataset),
            "num_features": int(dataset.num_features),
            "num_classes": int(dataset.num_classes),
            "label_counts": label_counts,
            "test_accuracy": test_accuracy,
            "test_loss": test_loss,
        },
    )
    manifest = {
        "checkpoint": str(checkpoint_path),
        "split": str(split_path),
        "best_epoch": best_epoch,
        "val_accuracy": best_val,
        "test_accuracy": test_accuracy,
        "test_loss": test_loss,
        "num_graphs": len(dataset),
        "num_features": int(dataset.num_features),
        "num_classes": int(dataset.num_classes),
        "label_counts": label_counts,
        "runtime_seconds": time.perf_counter() - started,
        "device": str(device),
    }
    (args.output_dir / "train_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
