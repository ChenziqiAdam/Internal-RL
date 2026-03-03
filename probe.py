"""
Phase 3 Validation + Phase 4: Linear Probe & Supervised Controller
- Linear probe: decode subgoal from residual stream at each layer
- Supervised controller: low-rank (rank-16) U^(g) inserted at mid-depth
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from model import AutoregressiveTransformer, TRANSFORMER_DIM, NUM_LAYERS
from train_pretrain import TrajectoryDataset, collate_fn
from data_gen import load_dataset
from env import OBS_DIM, NUM_ACTIONS, PRETRAIN_TASKS, POSTRAIN_TASK
from utils import get_best_device


# ── Linear Probe ──────────────────────────────────────────────────────────────

def extract_residuals_and_labels(
    model: AutoregressiveTransformer,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 50,
):
    """Extract residual stream activations and subgoal labels for all layers."""
    model.eval()
    all_residuals = [[] for _ in range(NUM_LAYERS)]
    all_labels = []

    with torch.no_grad():
        for i, (obs, acts, subgoals, mask) in enumerate(loader):
            if i >= max_batches:
                break
            obs, acts = obs.to(device), acts.to(device)
            residuals = model.get_residual_stream(obs, acts)
            # residuals: list of (B, T, dim)

            subgoal_flat = subgoals[mask]  # (N,)
            for layer_idx, res in enumerate(residuals):
                res_flat = res[mask.to(device)]  # (N, dim)
                all_residuals[layer_idx].append(res_flat.cpu())
            all_labels.append(subgoal_flat)

    return [torch.cat(r) for r in all_residuals], torch.cat(all_labels)


def train_linear_probe(features: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    """Train a simple linear classifier and return accuracy."""
    N, D = features.shape
    split = int(0.8 * N)
    X_tr, X_te = features[:split], features[split:]
    y_tr, y_te = labels[:split], labels[split:]

    clf = nn.Linear(D, num_classes)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)

    for _ in range(200):
        logits = clf(X_tr)
        loss = F.cross_entropy(logits, y_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        preds = clf(X_te).argmax(dim=-1)
        acc = (preds == y_te).float().mean().item()

    return acc


def run_linear_probe(model_path: str, data_path: str):
    device = get_best_device()

    # Load model
    ckpt = torch.load(model_path, map_location=device)
    model = AutoregressiveTransformer(predict_obs=True).to(device)
    model.load_state_dict(ckpt["model_state"])

    # Load data
    dataset = load_dataset(data_path)
    ds = TrajectoryDataset(dataset)
    loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    residuals_by_layer, labels = extract_residuals_and_labels(model, loader, device)

    # Determine number of subgoal classes
    num_classes = labels.max().item() + 1
    print(f"Subgoal classes: {num_classes}")
    print(f"Samples: {labels.shape[0]}")

    print("\n=== Linear Probe Accuracy per Layer ===")
    for layer_idx, res in enumerate(residuals_by_layer):
        acc = train_linear_probe(res.float(), labels.long(), num_classes)
        print(f"  Layer {layer_idx + 1}: {acc:.3f}")


# ── Supervised Controller ─────────────────────────────────────────────────────

class LowRankController(nn.Module):
    """
    Rank-r linear controller: U = A @ B^T, A ∈ R^{dim×r}, B ∈ R^{dim×r}
    Adds perturbation: e_t <- e_t + U @ e_t
    """
    def __init__(self, dim: int = TRANSFORMER_DIM, rank: int = 16):
        super().__init__()
        self.A = nn.Parameter(torch.randn(dim, rank) * 0.01)
        self.B = nn.Parameter(torch.randn(dim, rank) * 0.01)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        U = self.A @ self.B.T  # (dim, dim)
        return e + e @ U.T     # (B, T, dim)


class ControlledTransformer(nn.Module):
    """
    Frozen pretrained transformer with a per-subgoal low-rank controller
    inserted at mid-depth (layer 3).
    """
    def __init__(self, base_model: AutoregressiveTransformer, num_subgoals: int, insert_layer: int = 3):
        super().__init__()
        self.base = base_model
        self.insert_layer = insert_layer  # 0-indexed

        # Freeze base model
        for p in self.base.parameters():
            p.requires_grad = False

        dim = base_model.dim
        self.controllers = nn.ModuleList([
            LowRankController(dim) for _ in range(num_subgoals)
        ])

    def forward(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
        subgoal_idx: int,
    ):
        """Forward with controller for given subgoal active."""
        tokens, is_action = self.base.embed_sequence(obs_seq, action_seq)
        x = tokens

        for layer_idx, layer in enumerate(self.base.layers):
            x = layer(x)
            if layer_idx == self.insert_layer:
                x = self.controllers[subgoal_idx](x)

        x = self.base.final_norm(x)
        T = action_seq.shape[1]
        obs_positions = x[:, 0::2, :][:, :T, :]
        action_logits = self.base.action_head(obs_positions)
        return action_logits


def train_supervised_controller(
    model_path: str,
    data_path: str,
    save_path: str,
    num_subgoals: int = 8,
    insert_layer: int = 2,  # 0-indexed, layer 3 = index 2
    rank: int = 16,
    steps: int = 10000,
    lr: float = 1e-3,
    batch_size: int = 256,
):
    device = get_best_device()

    # Load frozen base model
    ckpt = torch.load(model_path, map_location=device)
    base_model = AutoregressiveTransformer(predict_obs=True).to(device)
    base_model.load_state_dict(ckpt["model_state"])

    controlled = ControlledTransformer(base_model, num_subgoals, insert_layer).to(device)

    dataset = load_dataset(data_path)
    from train_pretrain import TrajectoryDataset, collate_fn
    ds = TrajectoryDataset(dataset)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)

    optimizer = torch.optim.Adam(
        [p for p in controlled.parameters() if p.requires_grad], lr=lr
    )

    step = 0
    print("Training supervised controllers...")
    while step < steps:
        for obs, acts, subgoals, mask in loader:
            obs, acts, subgoals, mask = obs.to(device), acts.to(device), subgoals.to(device), mask.to(device)

            # For each batch, we use the subgoal at each timestep
            # Simplification: use a single dominant subgoal per trajectory
            # (proper approach would inject controller per timestep)
            dominant_subgoal = subgoals[:, 0].mode().values.item()  # most common first subgoal

            logits = controlled(obs, acts, int(dominant_subgoal))
            loss = F.cross_entropy(logits[mask], acts[mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            if step % 500 == 0:
                print(f"  Step {step}/{steps} | loss={loss.item():.4f}")
            if step >= steps:
                break

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    torch.save(controlled.state_dict(), save_path)
    print(f"Saved controller to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["probe", "controller"], required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/pretrain.pkl")
    parser.add_argument("--save_path", type=str, default="checkpoints/controller.pt")
    args = parser.parse_args()

    if args.mode == "probe":
        run_linear_probe(args.model_path, args.data_path)
    else:
        train_supervised_controller(args.model_path, args.data_path, args.save_path)
