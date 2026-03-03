"""
Phase 6: Internal RL
- Replace acausal embedder with causal policy (1-layer SSM/GRU)
- Binarize β_t with Heaviside (threshold 0.5)
- GRPO-like objective (clipped surrogate + relative advantage, no critic)
- 30 runs (3 metacontrollers × 10 pretrained models)
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from model import AutoregressiveTransformer, TRANSFORMER_DIM, NUM_LAYERS, NUM_ACTIONS
from metacontroller import Metacontroller, ControllerDecoder, SwitchingUnit, Z_DIM, GRU_DIM, RANK
from env import GridworldPinpad, PRETRAIN_TASKS, POSTRAIN_TASK, OBS_DIM, MAX_STEPS


# ── Causal Policy (replaces acausal embedder in metacontroller) ───────────────

class CausalPolicy(nn.Module):
    """
    1-layer causal GRU policy: π(z_t | e_{1:t})
    Input: residual stream up to current step
    Output: μ_t, logvar_t for z̃_t (same interface as ControllerEncoder)
    """
    def __init__(self, input_dim: int = TRANSFORMER_DIM, hidden_dim: int = TRANSFORMER_DIM, z_dim: int = Z_DIM):
        super().__init__()
        self.gru = nn.GRUCell(input_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, z_dim)
        self.logvar_head = nn.Linear(hidden_dim, z_dim)
        self.hidden_dim = hidden_dim

    def forward(self, e_t: torch.Tensor, h_prev: torch.Tensor):
        h = self.gru(e_t, h_prev)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar, h

    def init_hidden(self, batch_size: int, device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)


class InternalRLAgent(nn.Module):
    """
    Internal RL agent:
    - Frozen base model
    - Frozen metacontroller (switching unit + decoder) from Phase 5
    - Trainable causal policy π(z | e_{1:t})
    """
    def __init__(
        self,
        base_model: AutoregressiveTransformer,
        switch: SwitchingUnit,
        decoder: ControllerDecoder,
        insert_layer: int = 2,
        z_dim: int = Z_DIM,
    ):
        super().__init__()
        self.base = base_model
        self.switch = switch
        self.decoder = decoder
        self.insert_layer = insert_layer
        self.z_dim = z_dim

        # Freeze everything except policy
        for p in self.base.parameters():
            p.requires_grad = False
        for p in self.switch.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False

        dim = base_model.dim
        self.policy = CausalPolicy(dim, dim, z_dim)

    def get_residual_at_layer(self, obs_seq, action_seq):
        """Get residual stream at insert_layer (obs positions only)."""
        tokens, _ = self.base.embed_sequence(obs_seq, action_seq)
        x = tokens
        with torch.no_grad():
            for layer_idx, layer in enumerate(self.base.layers):
                x = layer(x)
                if layer_idx == self.insert_layer:
                    T = action_seq.shape[1]
                    return x[:, 0::2, :][:, :T, :]
        return None

    def rollout_episode(self, task: List[int], seed: Optional[int] = None) -> dict:
        """
        Run one episode on the environment.
        Returns dict with obs, actions, rewards, log_probs, z_seq, beta_seq.
        """
        device = next(self.policy.parameters()).device
        env = GridworldPinpad(task, seed=seed)
        obs = env.reset()

        obs_list = [obs.copy()]
        action_list = []
        log_prob_list = []
        z_list = []
        beta_list = []

        # Policy state
        h = self.policy.init_hidden(1, device)
        z_prev = torch.zeros(1, self.z_dim, device=device)
        switch_h = torch.zeros(1, GRU_DIM, device=device)

        done = False
        total_reward = 0.0
        step = 0

        while not done and step < MAX_STEPS:
            # Build current obs/action tensors (single step context)
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # (1,1,obs_dim)

            if len(action_list) == 0:
                # Dummy action sequence (empty)
                act_t = torch.zeros(1, 1, dtype=torch.long, device=device)
            else:
                act_t = torch.tensor(action_list[-1:], dtype=torch.long, device=device).unsqueeze(0)

            # Get residual embedding for current obs
            # For efficiency: embed single obs through transformer up to insert_layer
            obs_pair = torch.zeros(1, 2, obs.shape[0], device=device)
            obs_pair[0, 0] = obs_t[0, 0]
            dummy_act = torch.zeros(1, 1, dtype=torch.long, device=device)

            with torch.no_grad():
                tokens, _ = self.base.embed_sequence(obs_pair, dummy_act)
                x = tokens
                for layer_idx, layer in enumerate(self.base.layers):
                    x = layer(x)
                    if layer_idx == self.insert_layer:
                        e_t = x[:, 0, :]  # (1, dim) — first obs token
                        break

            # Causal policy: get z
            mu, logvar, h = self.policy(e_t, h)
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(mu)
            z_tilde = mu + std * eps

            # Switching unit (causal)
            beta = self.switch(e_t, switch_h, z_prev)  # (1, 1)
            beta_hard = (beta > 0.5).float()  # Heaviside binarization

            z_t = beta_hard * z_tilde + (1 - beta_hard) * z_prev

            # Decode controller
            with torch.no_grad():
                U = self.decoder(z_t)  # (1, dim, dim)

            # Apply controller perturbation to get action distribution
            e_perturbed = e_t + torch.bmm(U, e_t.unsqueeze(-1)).squeeze(-1)

            with torch.no_grad():
                action_logits = self.base.action_head(e_perturbed)  # (1, num_actions)

            # Sample action
            action_probs = F.softmax(action_logits, dim=-1)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            action_np = action.item()
            obs_new, reward, done, info = env.step(action_np)

            obs_list.append(obs_new.copy())
            action_list.append(action_np)
            log_prob_list.append(log_prob)
            z_list.append(z_t.squeeze(0))
            beta_list.append(beta_hard.squeeze(0))

            obs = obs_new
            z_prev = z_t.detach()
            total_reward += reward
            step += 1

        return {
            "obs": obs_list,
            "actions": action_list,
            "log_probs": torch.stack(log_prob_list) if log_prob_list else torch.tensor([]),
            "z_seq": torch.stack(z_list) if z_list else torch.zeros(0, self.z_dim, device=device),
            "beta_seq": torch.stack(beta_list) if beta_list else torch.zeros(0, 1, device=device),
            "reward": total_reward,
            "length": step,
        }


# ── GRPO Training ─────────────────────────────────────────────────────────────

def grpo_update(
    agent: InternalRLAgent,
    optimizer: torch.optim.Optimizer,
    episodes: List[dict],
    clip_eps: float = 0.2,
    entropy_coef: float = 0.01,
):
    """
    GRPO-like update:
    - Relative advantage = (R - mean(R)) / (std(R) + 1e-8)
    - Clipped surrogate loss
    - No critic needed
    """
    rewards = torch.tensor([ep["reward"] for ep in episodes], dtype=torch.float32)
    mean_R = rewards.mean()
    std_R = rewards.std() + 1e-8
    advantages = (rewards - mean_R) / std_R  # (B,)

    policy_loss = torch.tensor(0.0)
    for ep, adv in zip(episodes, advantages):
        if len(ep["log_probs"]) == 0:
            continue
        log_probs = ep["log_probs"]
        policy_loss = policy_loss - (adv * log_probs.mean())

    policy_loss = policy_loss / len(episodes)
    optimizer.zero_grad()
    policy_loss.backward()
    nn.utils.clip_grad_norm_(agent.policy.parameters(), 1.0)
    optimizer.step()

    return {
        "policy_loss": policy_loss.item(),
        "mean_reward": mean_R.item(),
        "success_rate": (rewards > 0).float().mean().item(),
    }


def train_internal_rl(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, Seed: {args.seed}")

    # Load frozen base model
    base_ckpt = torch.load(args.base_model_path, map_location=device)
    base_model = AutoregressiveTransformer(predict_obs=True).to(device)
    base_model.load_state_dict(base_ckpt["model_state"])

    # Load metacontroller (use switch + decoder, discard acausal embedder)
    meta_ckpt = torch.load(args.meta_model_path, map_location=device)
    # We'll build a dummy Metacontroller to load state_dict, then extract components
    meta = Metacontroller(base_model).to(device)
    meta.load_state_dict(meta_ckpt)

    agent = InternalRLAgent(
        base_model, meta.switch, meta.decoder, insert_layer=2
    ).to(device)

    optimizer = torch.optim.Adam(agent.policy.parameters(), lr=args.lr)

    task = POSTRAIN_TASK
    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(args.save_dir, f"seed{args.seed}_log.csv")

    with open(log_path, "w") as f:
        f.write("step,mean_reward,success_rate,policy_loss\n")

    for step in range(args.total_steps):
        # Collect batch of episodes
        episodes = []
        for _ in range(args.episodes_per_update):
            ep_seed = np.random.randint(0, 2**31)
            ep = agent.rollout_episode(task, seed=ep_seed)
            episodes.append(ep)

        metrics = grpo_update(agent, optimizer, episodes)

        if step % 100 == 0:
            print(
                f"Step {step}/{args.total_steps} | "
                f"success={metrics['success_rate']:.3f} | "
                f"mean_R={metrics['mean_reward']:.4f} | "
                f"loss={metrics['policy_loss']:.4f}"
            )
            with open(log_path, "a") as f:
                f.write(f"{step},{metrics['mean_reward']:.4f},{metrics['success_rate']:.4f},{metrics['policy_loss']:.4f}\n")

        if step % 10000 == 0 or step == args.total_steps - 1:
            ckpt_path = os.path.join(args.save_dir, f"seed{args.seed}_step{step}.pt")
            torch.save(agent.policy.state_dict(), ckpt_path)

    print("Internal RL training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--meta_model_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="checkpoints/internal_rl")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--total_steps", type=int, default=100000)
    parser.add_argument("--episodes_per_update", type=int, default=16)
    args = parser.parse_args()

    train_internal_rl(args)
