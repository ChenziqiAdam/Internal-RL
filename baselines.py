"""
Phase 7: Baselines
1. Raw action RL: GRPO directly on pretrained transformer token space
2. Internal RL w/o temporal abstraction: β_t = 1 always
3. Internal RL co-train: metacontroller + base model trained jointly (not frozen)
4. CompILE: segment-and-encode alternative to metacontroller
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from model import AutoregressiveTransformer, NUM_ACTIONS
from internal_rl import InternalRLAgent, grpo_update, CausalPolicy
from metacontroller import Metacontroller, ControllerDecoder, SwitchingUnit, Z_DIM, GRU_DIM, RANK
from env import GridworldPinpad, POSTRAIN_TASK, OBS_DIM, MAX_STEPS


# ── Baseline 1: Raw Action RL (GRPO on token space) ──────────────────────────

class RawActionRLAgent(nn.Module):
    """GRPO directly on the pretrained transformer output."""
    def __init__(self, base_model: AutoregressiveTransformer):
        super().__init__()
        self.base = base_model
        # Optionally fine-tune a small adapter head
        self.head = nn.Linear(base_model.dim, NUM_ACTIONS)

    def rollout_episode(self, task: List[int], seed: Optional[int] = None) -> dict:
        device = next(self.parameters()).device
        env = GridworldPinpad(task, seed=seed)
        obs = env.reset()

        obs_history = [obs.copy()]
        action_list = []
        log_prob_list = []
        done = False
        total_reward = 0.0
        step = 0

        while not done and step < MAX_STEPS:
            obs_t = torch.tensor(np.array(obs_history), dtype=torch.float32, device=device)
            # obs_t: (T, obs_dim) → (1, T, obs_dim)
            obs_t = obs_t.unsqueeze(0)
            T = step + 1

            # Dummy past actions
            if step == 0:
                act_t = torch.zeros(1, 1, dtype=torch.long, device=device)
            else:
                act_t = torch.tensor(action_list, dtype=torch.long, device=device).unsqueeze(0)
                # Pad obs to match action count
                if obs_t.shape[1] > act_t.shape[1] + 1:
                    obs_t = obs_t[:, :act_t.shape[1] + 1]

            with torch.no_grad():
                logits, _, _ = self.base(obs_t, act_t[:, :obs_t.shape[1]-1] if step > 0 else act_t)
                last_logit = logits[0, -1]  # (num_actions,)
                action_logits = self.head(self.base.action_head.weight.T @ last_logit.unsqueeze(-1)).squeeze() if False else last_logit

            # Enable grad for policy gradient
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            dummy_act = torch.zeros(1, 1, dtype=torch.long, device=device)
            with torch.set_grad_enabled(True):
                logits_grad, _, _ = self.base(
                    torch.cat([obs_tensor], dim=1),
                    dummy_act,
                )
                probs = F.softmax(logits_grad[0, -1], dim=-1)

            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            obs_new, reward, done, _ = env.step(action.item())
            obs_history.append(obs_new.copy())
            action_list.append(action.item())
            log_prob_list.append(log_prob)
            obs = obs_new
            total_reward += reward
            step += 1

        return {
            "log_probs": torch.stack(log_prob_list) if log_prob_list else torch.tensor([]),
            "reward": total_reward,
        }


# ── Baseline 2: Internal RL w/o Temporal Abstraction (β_t = 1 always) ────────

class NoTemporalAbstractionAgent(InternalRLAgent):
    """Same as InternalRLAgent but forces β_t = 1 (new subgoal every step)."""
    def rollout_episode(self, task: List[int], seed: Optional[int] = None) -> dict:
        device = next(self.policy.parameters()).device
        env = GridworldPinpad(task, seed=seed)
        obs = env.reset()

        obs_list = [obs.copy()]
        action_list = []
        log_prob_list = []
        z_list = []

        h = self.policy.init_hidden(1, device)
        z_prev = torch.zeros(1, self.z_dim, device=device)

        done = False
        total_reward = 0.0
        step = 0

        while not done and step < MAX_STEPS:
            obs_pair = torch.zeros(1, 2, obs.shape[0], device=device)
            obs_pair[0, 0] = torch.tensor(obs, dtype=torch.float32, device=device)
            dummy_act = torch.zeros(1, 1, dtype=torch.long, device=device)

            with torch.no_grad():
                tokens, _ = self.base.embed_sequence(obs_pair, dummy_act)
                x = tokens
                for layer_idx, layer in enumerate(self.base.layers):
                    x = layer(x)
                    if layer_idx == self.insert_layer:
                        e_t = x[:, 0, :]
                        break

            mu, logvar, h = self.policy(e_t, h)
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(mu)
            z_tilde = mu + std * eps

            # Force β_t = 1 (no temporal abstraction)
            z_t = z_tilde

            with torch.no_grad():
                U = self.decoder(z_t)

            e_perturbed = e_t + torch.bmm(U, e_t.unsqueeze(-1)).squeeze(-1)
            action_logits = self.base.action_head(e_perturbed)
            probs = F.softmax(action_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            obs_new, reward, done, _ = env.step(action.item())
            obs_list.append(obs_new.copy())
            action_list.append(action.item())
            log_prob_list.append(log_prob)
            z_list.append(z_t.squeeze(0))

            obs = obs_new
            z_prev = z_t.detach()
            total_reward += reward
            step += 1

        return {
            "obs": obs_list,
            "actions": action_list,
            "log_probs": torch.stack(log_prob_list) if log_prob_list else torch.tensor([]),
            "z_seq": torch.stack(z_list) if z_list else torch.zeros(0, self.z_dim, device=device),
            "reward": total_reward,
        }


# ── Baseline 3: Co-training (base model NOT frozen) ───────────────────────────

class CotrnainMetacontroller(Metacontroller):
    """Same as Metacontroller but base model is NOT frozen."""
    def __init__(self, base_model, insert_layer=2):
        super().__init__(base_model, insert_layer)
        # Unfreeze base model
        for p in self.base.parameters():
            p.requires_grad = True


# ── Baseline 4: CompILE ───────────────────────────────────────────────────────

class CompILESegmenter(nn.Module):
    """
    CompILE-style segmentation module.
    Replaces metacontroller switching unit with a learned discrete segmenter.
    Uses a simple Bernoulli gate over residual stream.
    Reference: Appendix C.4.3
    """
    def __init__(self, dim: int = 256, z_dim: int = Z_DIM, rank: int = RANK):
        super().__init__()
        self.z_dim = z_dim

        # Segmentation gate: P(boundary_t | e_t)
        self.gate = nn.Sequential(
            nn.Linear(dim, 64), nn.Tanh(),
            nn.Linear(64, 1), nn.Sigmoid()
        )

        # Segment encoder: GRU over e_t within segment
        self.seg_gru = nn.GRUCell(dim, dim)

        # Latent head
        self.mu_head = nn.Linear(dim, z_dim)
        self.logvar_head = nn.Linear(dim, z_dim)

        # Controller decoder
        self.decoder = ControllerDecoder(z_dim, dim, rank)

    def forward(self, e_seq: torch.Tensor):
        """
        e_seq: (B, T, dim)
        Returns: z_seq (B, T, z_dim), gate_seq (B, T, 1), mu_seq, logvar_seq
        """
        B, T, dim = e_seq.shape
        device = e_seq.device

        h = torch.zeros(B, dim, device=device)
        z_prev = torch.zeros(B, self.z_dim, device=device)

        z_list, gate_list, mu_list, logvar_list = [], [], [], []

        for t in range(T):
            e_t = e_seq[:, t, :]
            gate_t = self.gate(e_t)  # (B, 1) — boundary probability

            # Hard gate via straight-through estimator
            gate_hard = (gate_t > 0.5).float()
            gate_ste = gate_hard - gate_t.detach() + gate_t  # STE

            # Reset hidden state at boundary
            h = (1 - gate_ste) * h + gate_ste * torch.zeros_like(h)
            h = self.seg_gru(e_t, h)

            mu_t = self.mu_head(h)
            logvar_t = self.logvar_head(h)
            std = (0.5 * logvar_t).exp()
            z_t = mu_t + std * torch.randn_like(mu_t)

            z_list.append(z_t)
            gate_list.append(gate_t)
            mu_list.append(mu_t)
            logvar_list.append(logvar_t)

        return (
            torch.stack(z_list, dim=1),
            torch.stack(gate_list, dim=1),
            torch.stack(mu_list, dim=1),
            torch.stack(logvar_list, dim=1),
        )


# ── Baseline Training Runner ──────────────────────────────────────────────────

def run_baseline(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_ckpt = torch.load(args.base_model_path, map_location=device)
    base_model = AutoregressiveTransformer(predict_obs=True).to(device)
    base_model.load_state_dict(base_ckpt["model_state"])

    task = POSTRAIN_TASK
    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(args.save_dir, f"{args.baseline}_seed{args.seed}_log.csv")

    if args.baseline == "raw_action_rl":
        agent = RawActionRLAgent(base_model).to(device)
        optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr)

        with open(log_path, "w") as f:
            f.write("step,mean_reward,success_rate\n")

        for step in range(args.total_steps):
            episodes = [agent.rollout_episode(task) for _ in range(args.episodes_per_update)]
            metrics = grpo_update(agent, optimizer, episodes)
            if step % 100 == 0:
                print(f"[raw_action_rl] Step {step} | success={metrics['success_rate']:.3f}")
                with open(log_path, "a") as f:
                    f.write(f"{step},{metrics['mean_reward']:.4f},{metrics['success_rate']:.4f}\n")

    elif args.baseline == "no_temporal":
        meta_ckpt = torch.load(args.meta_model_path, map_location=device)
        meta = Metacontroller(base_model).to(device)
        meta.load_state_dict(meta_ckpt)
        agent = NoTemporalAbstractionAgent(base_model, meta.switch, meta.decoder).to(device)
        optimizer = torch.optim.Adam(agent.policy.parameters(), lr=args.lr)

        with open(log_path, "w") as f:
            f.write("step,mean_reward,success_rate\n")

        for step in range(args.total_steps):
            episodes = [agent.rollout_episode(task) for _ in range(args.episodes_per_update)]
            metrics = grpo_update(agent, optimizer, episodes)
            if step % 100 == 0:
                print(f"[no_temporal] Step {step} | success={metrics['success_rate']:.3f}")
                with open(log_path, "a") as f:
                    f.write(f"{step},{metrics['mean_reward']:.4f},{metrics['success_rate']:.4f}\n")

    elif args.baseline == "cotrain":
        # Co-training: metacontroller + unfrozen base
        meta = CotrnainMetacontroller(base_model).to(device)
        optimizer = torch.optim.AdamW(meta.parameters(), lr=3e-4, weight_decay=0.03)

        from data_gen import load_dataset
        from train_pretrain import TrajectoryDataset, collate_fn
        from torch.utils.data import DataLoader

        dataset = load_dataset(args.data_path)
        ds = TrajectoryDataset(dataset)
        loader = DataLoader(ds, batch_size=512, shuffle=True, collate_fn=collate_fn, drop_last=True)

        step = 0
        while step < args.total_steps:
            for obs, acts, _, mask in loader:
                obs, acts, mask = obs.to(device), acts.to(device), mask.to(device)
                logits, mu, logvar, beta, z = meta(obs, acts)
                nll = F.cross_entropy(logits[mask], acts[mask])
                kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()))[mask].mean()
                loss = nll + 0.1 * kl
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(meta.parameters(), 1.0)
                optimizer.step()
                step += 1
                if step % 1000 == 0:
                    print(f"[cotrain] Step {step} | nll={nll.item():.4f}")
                if step >= args.total_steps:
                    break

    print(f"Baseline {args.baseline} complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", choices=["raw_action_rl", "no_temporal", "cotrain"], required=True)
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--meta_model_path", type=str, default=None)
    parser.add_argument("--data_path", type=str, default="data/pretrain.pkl")
    parser.add_argument("--save_dir", type=str, default="checkpoints/baselines")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--total_steps", type=int, default=100000)
    parser.add_argument("--episodes_per_update", type=int, default=16)
    args = parser.parse_args()

    run_baseline(args)
