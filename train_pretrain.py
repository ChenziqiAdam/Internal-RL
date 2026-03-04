"""
Phase 3: Autoregressive Sequence Model Pretraining
- AdamW, lr=3e-4, weight_decay=0.03, batch=1024, 256K steps
- Loss: CE(next-action) + λ * CE(next-obs)
- Supports 10 seeds
"""

import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import List, Optional
import pickle
from utils import get_best_device

from model import AutoregressiveTransformer, TRANSFORMER_DIM, NUM_LAYERS
from data_gen import load_dataset, load_dataset_mmap
from env import OBS_DIM, NUM_ACTIONS


# ── Dataset ──────────────────────────────────────────────────────────────────

class TrajectoryDataset(Dataset):
    def __init__(self, dataset: dict, max_len: int = 200):
        self.obs_seqs = dataset["obs_seqs"]
        self.action_seqs = dataset["action_seqs"]
        self.subgoal_seqs = dataset["subgoal_seqs"]
        self.max_len = max_len

    def __len__(self):
        return len(self.obs_seqs)

    def __getitem__(self, idx):
        obs = torch.tensor(self.obs_seqs[idx], dtype=torch.float32)
        acts = torch.tensor(self.action_seqs[idx], dtype=torch.long)
        subgoals = torch.tensor(self.subgoal_seqs[idx], dtype=torch.long)

        T = acts.shape[0]
        if T > self.max_len:
            start = np.random.randint(0, T - self.max_len)
            obs = obs[start : start + self.max_len + 1]
            acts = acts[start : start + self.max_len]
            subgoals = subgoals[start : start + self.max_len]

        return obs, acts, subgoals


class MmapTrajectoryDataset(Dataset):
    """
    Memory-mapped dataset. Only the requested slice is paged into RAM per __getitem__.
    Uses torch.from_numpy() to avoid extra copies for float32 obs.
    """
    def __init__(self, dataset: dict, max_len: int = 200):
        self.obs_flat = dataset["obs_flat"]       # np.memmap (total_obs, obs_dim)
        self.acts_flat = dataset["acts_flat"]     # np.memmap (total_steps,)
        self.sg_flat = dataset["subgoals_flat"]   # np.memmap (total_steps,)
        self.index = dataset["index"]             # (N, 2) int64 action offsets
        self.obs_index = dataset["obs_index"]     # (N, 2) int64 obs offsets
        self.max_len = max_len
        self._lengths = self.index[:, 1] - self.index[:, 0]  # episode lengths

    def __len__(self):
        return len(self.index)

    def lengths(self):
        return self._lengths

    def __getitem__(self, idx):
        a_start, a_end = self.index[idx]
        o_start, o_end = self.obs_index[idx]

        # Slice from mmap — only this region is paged in
        obs_np = np.array(self.obs_flat[o_start:o_end], dtype=np.float32)
        acts_np = np.array(self.acts_flat[a_start:a_end], dtype=np.int64)
        sg_np = np.array(self.sg_flat[a_start:a_end], dtype=np.int64)

        T = acts_np.shape[0]
        if T > self.max_len:
            start = np.random.randint(0, T - self.max_len)
            obs_np = obs_np[start: start + self.max_len + 1]
            acts_np = acts_np[start: start + self.max_len]
            sg_np = sg_np[start: start + self.max_len]

        # from_numpy avoids copy (obs is already float32)
        obs = torch.from_numpy(obs_np)
        acts = torch.from_numpy(acts_np)
        subgoals = torch.from_numpy(sg_np)
        return obs, acts, subgoals


class BucketSampler(Sampler):
    """
    Groups indices into batches of similar sequence lengths to minimize padding.
    Shuffles within and across buckets each epoch.
    """
    def __init__(self, lengths: np.ndarray, batch_size: int, drop_last: bool = True):
        self.batch_size = batch_size
        self.drop_last = drop_last
        # Sort by length, then split into batches
        self._sorted_idx = np.argsort(lengths)

    def __iter__(self):
        # Chunk sorted indices into batch_size blocks, shuffle block order, yield each block as a batch
        idx = self._sorted_idx.copy()
        chunks = [idx[i: i + self.batch_size] for i in range(0, len(idx), self.batch_size)]
        np.random.shuffle(chunks)
        for chunk in chunks:
            if self.drop_last and len(chunk) < self.batch_size:
                continue
            np.random.shuffle(chunk)
            yield chunk.tolist()

    def __len__(self):
        n = len(self._sorted_idx)
        if self.drop_last:
            return n // self.batch_size
        return math.ceil(n / self.batch_size)


def collate_fn(batch):
    """Pad sequences to same length within batch."""
    obs_list, act_list, subgoal_list = zip(*batch)
    max_T = max(a.shape[0] for a in act_list)

    obs_dim = obs_list[0].shape[-1]
    B = len(obs_list)

    obs_pad = torch.zeros(B, max_T + 1, obs_dim)
    act_pad = torch.zeros(B, max_T, dtype=torch.long)
    subgoal_pad = torch.zeros(B, max_T, dtype=torch.long)
    mask = torch.zeros(B, max_T, dtype=torch.bool)

    for i, (obs, act, sg) in enumerate(zip(obs_list, act_list, subgoal_list)):
        T = act.shape[0]
        obs_pad[i, :T+1] = obs
        act_pad[i, :T] = act
        subgoal_pad[i, :T] = sg
        mask[i, :T] = True

    return obs_pad, act_pad, subgoal_pad, mask


# ── Training ─────────────────────────────────────────────────────────────────

def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_best_device()
    print(f"Device: {device}, Seed: {args.seed}")

    # Load data — use mmap format if data_path is a directory
    if os.path.isdir(args.data_path):
        dataset = load_dataset_mmap(args.data_path)
        ds = MmapTrajectoryDataset(dataset, max_len=args.max_len)
        sampler = BucketSampler(ds.lengths(), batch_size=args.batch_size, drop_last=True)
        loader = DataLoader(
            ds, batch_sampler=sampler, collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        dataset = load_dataset(args.data_path)
        ds = TrajectoryDataset(dataset, max_len=args.max_len)
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=args.num_workers,
            drop_last=True, pin_memory=torch.cuda.is_available(),
        )

    # Model
    model = AutoregressiveTransformer(
        predict_obs=(args.lam > 0)
    ).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Cosine LR schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.total_steps
    )

    os.makedirs(args.save_dir, exist_ok=True)
    step = 0
    epoch = 0
    accum_steps = args.grad_accum

    while step < args.total_steps:
        epoch += 1
        optimizer.zero_grad()
        accum_count = 0
        for obs, acts, subgoals, mask in loader:
            obs = obs.to(device)
            acts = acts.to(device)
            mask = mask.to(device)

            action_logits, obs_logits, _ = model(obs, acts)

            # Action loss (masked)
            B, T, _ = action_logits.shape
            action_loss = F.cross_entropy(
                action_logits[mask], acts[mask]
            )

            # Observation prediction loss
            obs_loss = torch.tensor(0.0, device=device)
            if args.lam > 0 and obs_logits is not None:
                # obs_logits: (B, T, obs_dim) predicts obs_{t+1}
                target_obs = obs[:, 1:, :]  # (B, T, obs_dim)
                obs_loss = F.mse_loss(obs_logits[mask], target_obs[mask])

            loss = (action_loss + args.lam * obs_loss) / accum_steps
            loss.backward()
            accum_count += 1

            if accum_count < accum_steps:
                continue

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            accum_count = 0

            # For logging, scale back
            action_loss = action_loss.detach()
            obs_loss = obs_loss.detach()

            step += 1
            if step % args.log_every == 0:
                print(
                    f"Step {step}/{args.total_steps} | "
                    f"act_loss={action_loss.item():.4f} | "
                    f"obs_loss={obs_loss.item():.4f} | "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )

            if step % args.save_every == 0 or step == args.total_steps:
                ckpt_path = os.path.join(args.save_dir, f"seed{args.seed}_step{step}.pt")
                torch.save({
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "step": step,
                    "args": vars(args),
                }, ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

            if step >= args.total_steps:
                break

    print("Pretraining complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/pretrain.pkl")
    parser.add_argument("--save_dir", type=str, default="checkpoints/pretrain")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.03)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--grad_accum", type=int, default=1,
                        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    parser.add_argument("--total_steps", type=int, default=256000)
    parser.add_argument("--lam", type=float, default=0.01,
                        help="Weight for next-obs prediction loss (0 to disable)")
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--log_every", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=50000)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    train(args)
