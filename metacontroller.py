"""
Phase 5: Metacontroller (Unsupervised Abstract Action Discovery)
Components:
  1. Internal sequence embedder: acausal SSM over residual stream
  2. Controller encoder: GRU(dim=32) + MLP → μ_t, Σ_t for z̃_t
  3. Switching unit: MLP → β_t ∈ [0,1] (causal)
  4. Temporal integration: z_t = β_t⊙z̃_t + (1-β_t)⊙z_{t-1}
  5. Controller decoder: hypernetwork MLP → low-rank U_t

Training: NLL(next-action) + α * KL(q‖N(0,I)), base model frozen.
"""

import os
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import AutoregressiveTransformer, TRANSFORMER_DIM, NUM_LAYERS, NUM_ACTIONS
from train_pretrain import TrajectoryDataset, MmapTrajectoryDataset, BucketSampler, collate_fn
from data_gen import load_dataset, load_dataset_mmap
from utils import get_best_device
from env import OBS_DIM

# Metacontroller hyperparams
Z_DIM = 8          # latent subgoal dimension
GRU_DIM = 32
RANK = 16           # rank of controller matrix U


# ── 1. Acausal Sequence Embedder (bidirectional GRU = approximate SSM) ───────

class AcausalEmbedder(nn.Module):
    """
    Processes full residual stream sequence (acausal = bidirectional).
    Input: (B, T, dim) residual activations
    Output: (B, T, hidden_dim) context vector s_t
    """
    def __init__(self, input_dim: int = TRANSFORMER_DIM, hidden_dim: int = TRANSFORMER_DIM):
        super().__init__()
        self.bigru = nn.GRU(input_dim, hidden_dim // 2, batch_first=True, bidirectional=True)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        s, _ = self.bigru(e)
        return s  # (B, T, hidden_dim)


# ── 2. Controller Encoder ─────────────────────────────────────────────────────

class ControllerEncoder(nn.Module):
    """
    GRU(dim=32) + MLP → μ_t, log_σ_t for latent z̃_t.
    Input at each step: (e_t, s_t) concatenated.
    Also uses hidden state h_{t-1} from GRU.
    """
    def __init__(self, input_dim: int = TRANSFORMER_DIM * 2, gru_dim: int = GRU_DIM, z_dim: int = Z_DIM):
        super().__init__()
        self.gru = nn.GRUCell(input_dim, gru_dim)
        self.mu_head = nn.Linear(gru_dim, z_dim)
        self.logvar_head = nn.Linear(gru_dim, z_dim)

    def forward(self, e_t: torch.Tensor, s_t: torch.Tensor, h_prev: torch.Tensor):
        """
        e_t: (B, dim), s_t: (B, dim), h_prev: (B, gru_dim)
        Returns: mu (B, z_dim), logvar (B, z_dim), h (B, gru_dim)
        """
        inp = torch.cat([e_t, s_t], dim=-1)  # (B, 2*dim)
        h = self.gru(inp, h_prev)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar, h


# ── 3. Switching Unit (causal) ────────────────────────────────────────────────

class SwitchingUnit(nn.Module):
    """
    MLP: (e_t, h_{t-1}, z_{t-1}) → β_t ∈ [0,1]
    Causal: no future information.
    """
    def __init__(self, e_dim: int = TRANSFORMER_DIM, gru_dim: int = GRU_DIM, z_dim: int = Z_DIM):
        super().__init__()
        in_dim = e_dim + gru_dim + z_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, e_t: torch.Tensor, h_prev: torch.Tensor, z_prev: torch.Tensor) -> torch.Tensor:
        """Returns β_t (B, 1)."""
        inp = torch.cat([e_t, h_prev, z_prev], dim=-1)
        return self.mlp(inp)


# ── 5. Controller Decoder (hypernetwork) ─────────────────────────────────────

class ControllerDecoder(nn.Module):
    """
    Hypernetwork MLP: z_t → (A_t, B_t) for low-rank controller U_t = A_t @ B_t^T
    where A_t, B_t ∈ R^{dim×rank}.
    """
    def __init__(self, z_dim: int = Z_DIM, dim: int = TRANSFORMER_DIM, rank: int = RANK):
        super().__init__()
        self.dim = dim
        self.rank = rank
        out_dim = 2 * dim * rank
        self.mlp = nn.Sequential(
            nn.Linear(z_dim, 32),
            nn.Tanh(),
            nn.Linear(32, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, z_dim)
        Returns U: (B, dim, dim) low-rank perturbation matrix
        """
        out = self.mlp(z)  # (B, 2*dim*rank)
        A = out[:, :self.dim * self.rank].view(-1, self.dim, self.rank)
        B = out[:, self.dim * self.rank:].view(-1, self.dim, self.rank)
        U = torch.bmm(A, B.transpose(1, 2))  # (B, dim, dim)
        return U


# ── Full Metacontroller ───────────────────────────────────────────────────────

class Metacontroller(nn.Module):
    """
    Full metacontroller wrapping a frozen base transformer.
    Processes a trajectory and produces latent z_t at each step,
    then applies the controller to steer the base model's residual stream.
    """
    def __init__(
        self,
        base_model: AutoregressiveTransformer,
        insert_layer: int = 2,  # 0-indexed (layer 3)
        z_dim: int = Z_DIM,
        gru_dim: int = GRU_DIM,
        rank: int = RANK,
    ):
        super().__init__()
        self.base = base_model
        self.insert_layer = insert_layer
        self.z_dim = z_dim
        self.gru_dim = gru_dim

        # Freeze base model
        for p in self.base.parameters():
            p.requires_grad = False

        dim = base_model.dim
        self.embedder = AcausalEmbedder(dim, dim)
        self.encoder = ControllerEncoder(dim * 2, gru_dim, z_dim)
        self.switch = SwitchingUnit(dim, gru_dim, z_dim)
        self.decoder = ControllerDecoder(z_dim, dim, rank)

    def forward(
        self,
        obs_seq: torch.Tensor,   # (B, T+1, obs_dim)
        action_seq: torch.Tensor, # (B, T) int64
    ):
        """
        Returns:
          action_logits: (B, T, num_actions) — with controller applied
          mu_seq: (B, T, z_dim)
          logvar_seq: (B, T, z_dim)
          beta_seq: (B, T, 1)
          z_seq: (B, T, z_dim)
        """
        B, T = action_seq.shape
        device = obs_seq.device

        # Step 1: Get base model residual stream at insert_layer (acausal: full pass first)
        with torch.no_grad():
            tokens, _ = self.base.embed_sequence(obs_seq, action_seq)
            x = tokens
            for layer_idx, layer in enumerate(self.base.layers):
                x = layer(x)
                if layer_idx == self.insert_layer:
                    e_seq = x[:, 0::2, :][:, :T, :]  # (B, T, dim) obs positions
                    break  # we only need residuals up to insert layer

        # Step 2: Acausal embedder over residual stream
        s_seq = self.embedder(e_seq)  # (B, T, dim)

        # Step 3: Temporal loop — encoder + switching unit
        h = torch.zeros(B, self.gru_dim, device=device)
        z_prev = torch.zeros(B, self.z_dim, device=device)

        mu_list, logvar_list, beta_list, z_list = [], [], [], []

        for t in range(T):
            e_t = e_seq[:, t, :]   # (B, dim)
            s_t = s_seq[:, t, :]   # (B, dim)

            mu_t, logvar_t, h = self.encoder(e_t, s_t, h)
            beta_t = self.switch(e_t, h, z_prev)  # (B, 1)

            # Reparameterize
            std = (0.5 * logvar_t).exp()
            eps = torch.randn_like(mu_t)
            z_tilde = mu_t + std * eps

            # Temporal integration
            z_t = beta_t * z_tilde + (1 - beta_t) * z_prev

            mu_list.append(mu_t)
            logvar_list.append(logvar_t)
            beta_list.append(beta_t)
            z_list.append(z_t)
            z_prev = z_t.detach()

        mu_seq = torch.stack(mu_list, dim=1)     # (B, T, z_dim)
        logvar_seq = torch.stack(logvar_list, dim=1)
        beta_seq = torch.stack(beta_list, dim=1) # (B, T, 1)
        z_seq = torch.stack(z_list, dim=1)       # (B, T, z_dim)

        # Step 4: Apply controller to base model (full forward pass with perturbation)
        action_logits = self._forward_with_controller(obs_seq, action_seq, z_seq)

        return action_logits, mu_seq, logvar_seq, beta_seq, z_seq

    def _forward_with_controller(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
        z_seq: torch.Tensor,  # (B, T, z_dim)
    ) -> torch.Tensor:
        """Full forward pass inserting controller at insert_layer."""
        B, T = action_seq.shape

        tokens, _ = self.base.embed_sequence(obs_seq, action_seq)
        x = tokens

        for layer_idx, layer in enumerate(self.base.layers):
            x = layer(x)
            if layer_idx == self.insert_layer:
                # Apply per-timestep controller at obs positions
                obs_pos = x[:, 0::2, :][:, :T, :]  # (B, T, dim)

                # Decode U_t for each timestep
                z_flat = z_seq.reshape(B * T, self.z_dim)
                U_flat = self.decoder(z_flat)  # (B*T, dim, dim)
                U = U_flat.view(B, T, self.base.dim, self.base.dim)

                # Perturb: e_t <- e_t + U_t @ e_t
                perturbed = obs_pos + torch.einsum("btij,btj->bti", U, obs_pos)

                # Write back to token sequence
                x = x.clone()
                x[:, 0::2, :][:, :T, :] = perturbed  # This won't work in-place correctly
                # Correct approach:
                new_x = x.clone()
                new_x[:, 0:2*T:2, :] = perturbed
                x = new_x

        x = self.base.final_norm(x)
        obs_positions = x[:, 0::2, :][:, :T, :]
        return self.base.action_head(obs_positions)


# ── Training ─────────────────────────────────────────────────────────────────

def train_metacontroller(args):
    torch.manual_seed(args.seed)
    device = get_best_device()
    print(f"Device: {device}, Seed: {args.seed}")

    # Load frozen base model
    ckpt = torch.load(args.model_path, map_location=device)
    base_model = AutoregressiveTransformer(predict_obs=True).to(device)
    base_model.load_state_dict(ckpt["model_state"])

    model = Metacontroller(base_model).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable):,}")

    optimizer = torch.optim.AdamW(trainable, lr=3e-4, weight_decay=0.03)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if os.path.isdir(args.data_path):
        dataset = load_dataset_mmap(args.data_path)
        ds = MmapTrajectoryDataset(dataset)
        sampler = BucketSampler(ds.lengths(), batch_size=args.batch_size, drop_last=True)
        loader = DataLoader(
            ds, batch_sampler=sampler, collate_fn=collate_fn,
            num_workers=2, pin_memory=torch.cuda.is_available(),
        )
    else:
        dataset = load_dataset(args.data_path)
        ds = TrajectoryDataset(dataset)
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_fn, drop_last=True,
            num_workers=2, pin_memory=torch.cuda.is_available(),
        )

    os.makedirs(args.save_dir, exist_ok=True)
    step = 0

    while step < args.total_steps:
        for obs, acts, subgoals, mask in loader:
            obs, acts, mask = obs.to(device), acts.to(device), mask.to(device)

            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                action_logits, mu_seq, logvar_seq, beta_seq, z_seq = model(obs, acts)

                # NLL loss
                nll = F.cross_entropy(action_logits[mask], acts[mask])

                # KL divergence: KL(N(mu, sigma) || N(0, I))
                kl = -0.5 * (1 + logvar_seq - mu_seq.pow(2) - logvar_seq.exp())
                kl = kl[mask].mean()

                loss = nll + args.alpha * kl

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()

            step += 1
            if step % 500 == 0:
                avg_beta = beta_seq[mask].mean().item()
                print(f"Step {step}/{args.total_steps} | nll={nll.item():.4f} | kl={kl.item():.4f} | β̄={avg_beta:.3f}")

            if step % 10000 == 0 or step == args.total_steps:
                ckpt_path = os.path.join(args.save_dir, f"seed{args.seed}_step{step}.pt")
                torch.save(model.state_dict(), ckpt_path)
                print(f"Saved: {ckpt_path}")

            if step >= args.total_steps:
                break

    print("Metacontroller training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/pretrain.pkl")
    parser.add_argument("--save_dir", type=str, default="checkpoints/metacontroller")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="KL weight, sweep over {0, 0.05, 0.1, 0.17, 0.3, 0.5, 1}")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--total_steps", type=int, default=64000)
    args = parser.parse_args()

    train_metacontroller(args)
