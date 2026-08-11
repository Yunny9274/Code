"""Train and checkpoint the frozen BA2Motifs model used by the pilot."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.datasets import BA2MotifDataset
from torch_geometric.loader import DataLoader

from examples.ba2motifs_experiment import BA2MotifGCN, load_graphs, save_split, seed_everything, stratified_split


def evaluate(model, loader, device: torch.device) -> tuple[float, float]:
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
    parser = argparse.ArgumentParser(description="Train the BA2Motifs GCN")
    parser.add_argument("--data-root", type=Path, default=Path("data/BA2Motif"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/ba2motifs_seed0"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--hidden", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = BA2MotifDataset(root=str(args.data_root))
    split = stratified_split(dataset.y, args.seed, 0.8, 0.1)
    train_graphs = load_graphs(dataset, split["train"])
    val_graphs = load_graphs(dataset, split["val"])
    test_graphs = load_graphs(dataset, split["test"])
    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size)

    model_kwargs = {
        "in_channels": int(dataset.num_features),
        "hidden_channels": args.hidden,
        "num_classes": int(dataset.num_classes),
        "dropout": args.dropout,
    }
    model = BA2MotifGCN(**model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val = -1.0
    best_epoch = -1
    best_state = None
    stale = 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0
        total_train_graphs = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = F.cross_entropy(logits, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            total_train_loss += float(loss.detach()) * int(batch.y.numel())
            total_train_graphs += int(batch.y.numel())
        train_accuracy, train_eval_loss = evaluate(model, train_loader, device)
        val_accuracy, val_loss = evaluate(model, val_loader, device)
        if val_accuracy > best_val:
            best_val = val_accuracy
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            mean_train_loss = total_train_loss / max(total_train_graphs, 1)
            print(
                f"epoch={epoch} train_accuracy={train_accuracy:.4f} "
                f"train_loss={mean_train_loss:.4f} eval_loss={train_eval_loss:.4f} "
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
    save_split(
        split_path,
        split,
        {
            "dataset": "BA2Motifs",
            "seed": args.seed,
            "train_fraction": 0.8,
            "val_fraction": 0.1,
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
        "runtime_seconds": time.perf_counter() - started,
        "device": str(device),
    }
    (args.output_dir / "train_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
