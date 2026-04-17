"""Clinical RL model + training loop. THE AGENT MODIFIES THIS FILE.

Everything is fair game: model architecture, reward shaping, RL algorithm,
hyperparameters, training strategy, multi-task selection.

Goal: maximize the primary metric for the chosen task(s).
"""

import argparse
import itertools
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    TaskRegistry,
    TaskSpec,
    compute_ddi_rate,
    compute_reward,
    count_parameters,
    evaluate_model,
    get_ddi_matrix,
    get_peak_vram_mb,
    load_task_data,
    print_results,
    set_seed,
    TIME_BUDGET_SECONDS,
)

# ============================================================
# Configuration — agent tunes these
# ============================================================

# Task selection (single task for MVP; multi-task interface is available)
TASK_NAME = "mimic4_los"

# Model architecture
EMBEDDING_DIM = 128
HIDDEN_DIM = 384
NUM_RNN_LAYERS = 2
DROPOUT = 0.3

# RL hyperparameters
RL_ALGO = "a2c_gae"    # reinforce | ppo | a2c_gae | dqn
GAMMA = 0.95          # discount factor for delayed rewards
ENTROPY_COEF = 0.01   # exploration bonus
VALUE_LOSS_COEF = 0.1 # value network weight (reduced from 0.5 — RL loss was dominating)
USE_BASELINE = True   # variance reduction
N_ACTIONS = 10        # history window for agent attention
RL_LOSS_COEF = 0.5    # scale RL loss relative to task loss

# PPO-specific
PPO_CLIP_EPS = 0.2    # clipping epsilon
PPO_MINI_EPOCHS = 3   # policy update passes per batch

# A2C-GAE-specific
GAE_LAMBDA = 0.95     # GAE lambda for bias-variance tradeoff

# DQN-specific
DQN_EPS_START = 1.0   # initial exploration rate
DQN_EPS_END = 0.05    # final exploration rate
DQN_EPS_DECAY = 0.995 # per-epoch decay
DQN_REPLAY_SIZE = 10000
DQN_TARGET_UPDATE = 5  # update target network every N epochs

# Ordinal regression (for multiclass tasks with ordered labels like LOS)
USE_ORDINAL = False

# Optimization
LR = 1e-3
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 128
MAX_GRAD_NORM = 1.0

# Input augmentation
CODE_MASK_RATE = 0.1  # randomly zero out 10% of medical codes during training

# Reproducibility
SEED = 42


# ============================================================
# Model Components — agent modifies architecture
# ============================================================

class CodeEmbedding(nn.Module):
    """Embed medical codes and aggregate per-visit.

    Input:  [batch, visits, codes] (LongTensor)
    Output: [batch, visits, embed_dim]
    """
    def __init__(self, vocab_size: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        # x: [batch, visits, codes]
        # Input augmentation: randomly mask codes during training
        if self.training and CODE_MASK_RATE > 0:
            mask = torch.rand_like(x, dtype=torch.float) > CODE_MASK_RATE
            x = x * mask.long()
        emb = self.embedding(x)            # [batch, visits, codes, embed_dim]
        emb = self.dropout(emb)
        # Sum over codes dimension (ignore padding via embedding zero)
        return emb.sum(dim=2)              # [batch, visits, embed_dim]


class PolicyAgent(nn.Module):
    """RL agent that selects relevant historical states.

    Supports multiple RL algorithms:
    - reinforce/a2c_gae/ppo: stochastic policy + value baseline
    - dqn: Q-network with epsilon-greedy exploration
    """
    def __init__(
        self,
        obs_dim: int,
        n_actions: int = 10,
        hidden_dim: int = 64,
        use_baseline: bool = True,
        rl_algo: str = "reinforce",
    ):
        super().__init__()
        self.n_actions = n_actions
        self.use_baseline = use_baseline
        self.rl_algo = rl_algo

        if rl_algo == "dqn":
            # Q-network for DQN
            self.q_net = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_actions),
            )
            # Target network (copy of q_net, updated periodically)
            self.q_target = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_actions),
            )
            self.q_target.load_state_dict(self.q_net.state_dict())
            for p in self.q_target.parameters():
                p.requires_grad = False
            self.epsilon = DQN_EPS_START
        else:
            # Policy network for REINFORCE / PPO / A2C-GAE
            self.policy_net = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, n_actions),
            )
            if use_baseline or rl_algo in ("ppo", "a2c_gae"):
                self.value_net = nn.Sequential(
                    nn.Linear(obs_dim, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, 1),
                )

    def forward(
        self, observation: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Select action from observation.

        Returns: (action, log_prob, entropy, baseline_value)
        """
        if self.rl_algo == "dqn":
            return self._forward_dqn(observation)
        return self._forward_policy(observation)

    def _forward_policy(
        self, observation: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        logits = self.policy_net(observation.detach())
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        baseline = None
        if hasattr(self, "value_net"):
            baseline = self.value_net(observation.detach()).squeeze(-1)

        return action, log_prob, entropy, baseline

    def _forward_dqn(
        self, observation: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        q_values = self.q_net(observation.detach())
        # Epsilon-greedy
        if self.training and np.random.random() < self.epsilon:
            action = torch.randint(0, self.n_actions, (observation.shape[0],),
                                   device=observation.device)
        else:
            action = q_values.argmax(dim=-1)
        # Return q_values as "log_prob" placeholder for loss computation
        return action, q_values, torch.zeros_like(action, dtype=torch.float), None

    def update_target(self):
        """Copy q_net weights to target network (DQN only)."""
        if self.rl_algo == "dqn":
            self.q_target.load_state_dict(self.q_net.state_dict())

    def decay_epsilon(self):
        """Decay exploration rate (DQN only)."""
        if self.rl_algo == "dqn":
            self.epsilon = max(DQN_EPS_END, self.epsilon * DQN_EPS_DECAY)


class ReplayBuffer:
    """Simple replay buffer for DQN."""
    def __init__(self, capacity: int = DQN_REPLAY_SIZE):
        self.capacity = capacity
        self.buffer: List[Tuple] = []
        self.pos = 0

    def push_batch(self, states, actions, rewards, next_states):
        """Push a batch of transitions (unbatched into individual samples)."""
        batch_size = states.shape[0]
        for i in range(batch_size):
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            self.buffer[self.pos] = (
                states[i], actions[i], rewards[i] if rewards.dim() > 0 else rewards,
                next_states[i],
            )
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states = zip(*[self.buffer[i] for i in indices])
        return (torch.stack(states), torch.stack(actions),
                torch.stack(rewards), torch.stack(next_states))

    def __len__(self):
        return len(self.buffer)


class ClinicalRLModel(nn.Module):
    """RL-based clinical prediction model.

    Architecture:
    1. Embed each feature type (conditions, procedures, etc.)
    2. Process visit sequence with GRU
    3. Use RL agent to select relevant historical states
    4. Predict output (drug combination / mortality / etc.)

    The agent can modify any part of this architecture.
    """
    def __init__(self, task_spec: TaskSpec, class_weights=None, pos_weight=None, rl_algo: str = "reinforce"):
        super().__init__()
        self.spec = task_spec
        self.task_type = task_spec.task_type
        self.label_key = task_spec.label_key
        self.class_weights = class_weights
        self.pos_weight = pos_weight
        self.rl_algo = rl_algo

        # Embeddings for each feature type
        self.embeddings = nn.ModuleDict()
        for key in task_spec.feature_keys:
            vocab = task_spec.feature_dims.get(key, 500)
            self.embeddings[key] = CodeEmbedding(vocab, EMBEDDING_DIM, DROPOUT)

        # Sequence encoder
        input_dim = EMBEDDING_DIM * len(task_spec.feature_keys)
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=HIDDEN_DIM,
            num_layers=NUM_RNN_LAYERS,
            batch_first=True,
            dropout=DROPOUT if NUM_RNN_LAYERS > 1 else 0,
        )

        # RL agent for history selection
        self.rl_agent = PolicyAgent(
            obs_dim=HIDDEN_DIM,
            n_actions=N_ACTIONS,
            hidden_dim=64,
            use_baseline=USE_BASELINE,
            rl_algo=rl_algo,
        )

        # DQN replay buffer
        if rl_algo == "dqn":
            self.replay_buffer = ReplayBuffer(DQN_REPLAY_SIZE)

        # Fusion layer: combines current state + agent-selected history
        self.fusion = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
        )

        # Task-specific output head
        self.use_ordinal = (task_spec.task_type == "multiclass" and USE_ORDINAL)
        if task_spec.task_type == "multilabel":
            self.output_head = nn.Linear(HIDDEN_DIM, task_spec.label_dim)
        elif task_spec.task_type == "binary":
            self.output_head = nn.Linear(HIDDEN_DIM, 2)
        elif task_spec.task_type == "multiclass":
            if self.use_ordinal:
                # Ordinal regression: K-1 cumulative thresholds
                n_thresholds = task_spec.label_dim - 1
                self.output_head = nn.Linear(HIDDEN_DIM, 1)  # shared latent
                self.thresholds = nn.Parameter(torch.linspace(-1, 1, n_thresholds))
            else:
                self.output_head = nn.Linear(HIDDEN_DIM, task_spec.label_dim)

    def encode(self, **kwargs):
        """Shared encoder: embeddings → GRU → RL history attention → fusion → last_state."""
        device = next(self.parameters()).device
        embedded = []
        for key in self.spec.feature_keys:
            if key in kwargs:
                embedded.append(self.embeddings[key](kwargs[key]))
        x = torch.cat(embedded, dim=-1)
        mask = kwargs.get("mask")
        rnn_out, _ = self.rnn(x)
        batch_size, seq_len, hidden_dim = rnn_out.shape

        rl_log_probs = []
        rl_entropies = []
        rl_baselines = []
        fused_states = []

        for t in range(seq_len):
            current = rnn_out[:, t, :]
            if t == 0:
                fused = self.fusion(torch.cat([current, current], dim=-1))
            else:
                history_len = min(t, N_ACTIONS)
                history_start = max(0, t - N_ACTIONS)
                history = rnn_out[:, history_start:t, :]
                action, log_prob, entropy, baseline = self.rl_agent(current)
                rl_log_probs.append(log_prob)
                rl_entropies.append(entropy)
                if baseline is not None:
                    rl_baselines.append(baseline)
                action_clamped = action.clamp(0, history_len - 1)
                idx = action_clamped.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, hidden_dim)
                selected = history.gather(1, idx).squeeze(1)
                fused = self.fusion(torch.cat([current, selected], dim=-1))
                if self.rl_algo == "dqn" and self.training:
                    self.replay_buffer.push_batch(
                        current.detach().cpu(), action_clamped.detach().cpu(),
                        torch.zeros(batch_size), rnn_out[:, t, :].detach().cpu(),
                    )
            fused_states.append(fused)

        fused_seq = torch.stack(fused_states, dim=1)
        if mask is not None:
            lengths = mask.long().sum(dim=1).clamp(min=1) - 1
            last_state = fused_seq[torch.arange(batch_size, device=device), lengths]
        else:
            last_state = fused_seq[:, -1, :]
        return last_state, rl_log_probs, rl_entropies, rl_baselines

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        last_state, rl_log_probs, rl_entropies, rl_baselines = self.encode(**kwargs)
        logit = self.output_head(last_state)

        # Compute loss
        result = {"logit": logit}

        if self.label_key in kwargs:
            y_true = kwargs[self.label_key]
            result["y_true"] = y_true

            # Task loss
            if self.task_type == "multilabel":
                task_loss = F.binary_cross_entropy_with_logits(
                    logit, y_true, pos_weight=self.pos_weight,
                )
                result["y_prob"] = torch.sigmoid(logit)
            elif self.task_type == "binary":
                task_loss = F.cross_entropy(logit, y_true)
                result["y_prob"] = F.softmax(logit, dim=-1)[:, 1]
            elif self.task_type == "multiclass" and self.use_ordinal:
                # Ordinal regression: cumulative logits → class probabilities
                # logit is [batch, 1], thresholds are [K-1]
                latent = logit.squeeze(-1)  # [batch]
                # Cumulative probabilities: P(Y > k) = sigmoid(latent - threshold_k)
                sorted_thresh, _ = torch.sort(self.thresholds)
                cum_probs = torch.sigmoid(
                    latent.unsqueeze(-1) - sorted_thresh.unsqueeze(0)
                )  # [batch, K-1]
                # Convert cumulative to class probs: P(Y=k) = P(Y>k-1) - P(Y>k)
                ones = torch.ones(cum_probs.shape[0], 1, device=cum_probs.device)
                zeros = torch.zeros(cum_probs.shape[0], 1, device=cum_probs.device)
                cum_full = torch.cat([ones, cum_probs, zeros], dim=-1)  # [batch, K+1]
                class_probs = cum_full[:, :-1] - cum_full[:, 1:]  # [batch, K]
                class_probs = class_probs.clamp(min=1e-7)
                # NLL loss
                task_loss = F.nll_loss(
                    class_probs.log(), y_true, weight=self.class_weights,
                )
                result["y_prob"] = class_probs
                result["logit"] = class_probs.log()  # for RL reward computation
            elif self.task_type == "multiclass":
                task_loss = F.cross_entropy(logit, y_true, weight=self.class_weights)
                result["y_prob"] = F.softmax(logit, dim=-1)

            # RL loss (REINFORCE with baseline)
            rl_loss = self._compute_rl_loss(
                rl_log_probs, rl_entropies, rl_baselines,
                task_loss.detach(), y_true, logit.detach(),
            )

            result["loss"] = task_loss + RL_LOSS_COEF * rl_loss
            result["task_loss"] = task_loss.detach()
            result["rl_loss"] = rl_loss.detach()

        return result

    def _compute_reward(self, logit: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute per-sample reward from prediction quality.

        For multiclass (LOS): ordinal-aware reward — closer predictions
        get partial credit (e.g., predicting 3-7d when true is 7-14d = 0.67).
        """
        with torch.no_grad():
            if self.task_type == "multilabel":
                y_pred = (torch.sigmoid(logit) > 0.5).float()
                intersection = (y_pred * y_true).sum(dim=-1)
                union = ((y_pred + y_true) > 0).float().sum(dim=-1)
                return intersection / union.clamp(min=1)
            elif self.task_type == "multiclass":
                pred_class = logit.argmax(dim=-1)
                n_classes = logit.shape[-1]
                distance = (pred_class - y_true).abs().float()
                # Reward: 1.0 for exact, decays linearly with distance
                return 1.0 - distance / max(n_classes - 1, 1)
            else:
                pred_class = logit.argmax(dim=-1)
                return (pred_class == y_true).float()

    def _compute_rl_loss(
        self,
        log_probs: List[torch.Tensor],
        entropies: List[torch.Tensor],
        baselines: List[torch.Tensor],
        task_loss: torch.Tensor,
        y_true: torch.Tensor,
        logit: torch.Tensor,
    ) -> torch.Tensor:
        if not log_probs:
            return torch.tensor(0.0, device=logit.device)

        if self.rl_algo == "dqn":
            return self._compute_dqn_loss(logit)
        elif self.rl_algo == "ppo":
            return self._compute_ppo_loss(log_probs, entropies, baselines, logit, y_true)
        elif self.rl_algo == "a2c_gae":
            return self._compute_a2c_gae_loss(log_probs, entropies, baselines, logit, y_true)
        else:  # reinforce
            return self._compute_reinforce_loss(log_probs, entropies, baselines, logit, y_true)

    def _compute_reinforce_loss(
        self, log_probs, entropies, baselines, logit, y_true,
    ) -> torch.Tensor:
        """Original REINFORCE with baseline."""
        reward = self._compute_reward(logit, y_true)
        T = len(log_probs)
        discounted = torch.zeros(T, reward.shape[0], device=logit.device)
        discounted[-1] = reward
        for t in reversed(range(T - 1)):
            discounted[t] = reward + GAMMA * discounted[t + 1]

        policy_loss = torch.tensor(0.0, device=logit.device)
        value_loss = torch.tensor(0.0, device=logit.device)
        entropy_bonus = torch.tensor(0.0, device=logit.device)

        for t in range(T):
            R = discounted[t]
            if baselines and USE_BASELINE:
                advantage = R - baselines[t]
                policy_loss -= (log_probs[t] * advantage.detach()).mean()
                value_loss += F.mse_loss(baselines[t], R)
            else:
                policy_loss -= (log_probs[t] * R).mean()
            entropy_bonus += entropies[t].mean()

        return (
            policy_loss / max(T, 1)
            + VALUE_LOSS_COEF * value_loss / max(T, 1)
            - ENTROPY_COEF * entropy_bonus / max(T, 1)
        )

    def _compute_ppo_loss(
        self, log_probs, entropies, baselines, logit, y_true,
    ) -> torch.Tensor:
        """PPO with clipped surrogate objective."""
        reward = self._compute_reward(logit, y_true)
        T = len(log_probs)
        discounted = torch.zeros(T, reward.shape[0], device=logit.device)
        discounted[-1] = reward
        for t in reversed(range(T - 1)):
            discounted[t] = reward + GAMMA * discounted[t + 1]

        # Store old log probs for ratio computation (detached)
        old_log_probs = [lp.detach() for lp in log_probs]

        policy_loss = torch.tensor(0.0, device=logit.device)
        value_loss = torch.tensor(0.0, device=logit.device)
        entropy_bonus = torch.tensor(0.0, device=logit.device)

        for t in range(T):
            R = discounted[t]
            if baselines:
                advantage = (R - baselines[t]).detach()
            else:
                advantage = R

            # PPO clipped ratio
            ratio = torch.exp(log_probs[t] - old_log_probs[t])
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP_EPS, 1.0 + PPO_CLIP_EPS) * advantage
            policy_loss -= torch.min(surr1, surr2).mean()

            if baselines:
                # Clipped value loss
                value_loss += F.mse_loss(baselines[t], R)

            entropy_bonus += entropies[t].mean()

        return (
            policy_loss / max(T, 1)
            + VALUE_LOSS_COEF * value_loss / max(T, 1)
            - ENTROPY_COEF * entropy_bonus / max(T, 1)
        )

    def _compute_a2c_gae_loss(
        self, log_probs, entropies, baselines, logit, y_true,
    ) -> torch.Tensor:
        """A2C with Generalized Advantage Estimation (GAE-lambda)."""
        reward = self._compute_reward(logit, y_true)
        T = len(log_probs)

        # Compute GAE advantages
        advantages = torch.zeros(T, reward.shape[0], device=logit.device)
        gae = torch.zeros(reward.shape[0], device=logit.device)

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = torch.zeros_like(reward)
            else:
                next_value = baselines[t + 1].detach() if baselines else torch.zeros_like(reward)

            current_value = baselines[t].detach() if baselines else torch.zeros_like(reward)
            # TD error: r + gamma * V(s') - V(s)
            # reward is terminal, so at last step delta = reward - V(s)
            # at other steps delta = gamma * V(s') - V(s)  (no intermediate reward)
            if t == T - 1:
                delta = reward - current_value
            else:
                delta = GAMMA * next_value - current_value
            gae = delta + GAMMA * GAE_LAMBDA * gae
            advantages[t] = gae

        policy_loss = torch.tensor(0.0, device=logit.device)
        value_loss = torch.tensor(0.0, device=logit.device)
        entropy_bonus = torch.tensor(0.0, device=logit.device)

        for t in range(T):
            adv = advantages[t].detach()
            policy_loss -= (log_probs[t] * adv).mean()
            if baselines:
                # Value target = advantage + current value estimate
                returns = advantages[t] + baselines[t].detach()
                value_loss += F.mse_loss(baselines[t], returns.detach())
            entropy_bonus += entropies[t].mean()

        return (
            policy_loss / max(T, 1)
            + VALUE_LOSS_COEF * value_loss / max(T, 1)
            - ENTROPY_COEF * entropy_bonus / max(T, 1)
        )

    def _compute_dqn_loss(self, logit: torch.Tensor) -> torch.Tensor:
        """DQN loss from replay buffer."""
        if not hasattr(self, "replay_buffer") or len(self.replay_buffer) < BATCH_SIZE:
            return torch.tensor(0.0, device=logit.device)

        device = logit.device
        states, actions, rewards, next_states = self.replay_buffer.sample(
            min(BATCH_SIZE, len(self.replay_buffer))
        )
        states = states.to(device)
        actions = actions.to(device).long()
        rewards = rewards.to(device)
        next_states = next_states.to(device)

        # Current Q values
        q_values = self.rl_agent.q_net(states)
        q_selected = q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # Target Q values (Double DQN: use online net for action selection)
        with torch.no_grad():
            next_q = self.rl_agent.q_net(next_states)
            next_actions = next_q.argmax(dim=-1, keepdim=True)
            next_q_target = self.rl_agent.q_target(next_states)
            next_q_selected = next_q_target.gather(1, next_actions).squeeze(-1)
            target = rewards + GAMMA * next_q_selected

        return F.smooth_l1_loss(q_selected, target)


# ============================================================
# Multi-Task Wrapper
# ============================================================

MULTITASK_TASKS = [
    "mimic4_mortality",
    "mimic4_readmission",
    "mimic4_los",
    "mimic4_phenotyping",
]

TASK_LOSS_WEIGHTS = {
    "mimic4_mortality": 1.0,
    "mimic4_readmission": 1.0,
    "mimic4_los": 1.5,
    "mimic4_phenotyping": 3.0,
}

TASK_SAMPLE_WEIGHTS = {
    "mimic4_mortality": 1,
    "mimic4_readmission": 1,
    "mimic4_los": 1,
    "mimic4_phenotyping": 1,
}


class MultiTaskWrapper(nn.Module):
    """Shared encoder + per-task output heads for multi-task learning."""

    def __init__(
        self,
        task_specs: Dict[str, TaskSpec],
        class_weights: Dict[str, Optional[torch.Tensor]],
        pos_weights: Dict[str, Optional[torch.Tensor]],
        rl_algo: str = "a2c_gae",
    ):
        super().__init__()
        self.task_specs = task_specs
        ref_spec = list(task_specs.values())[0]
        self.encoder = ClinicalRLModel(ref_spec, rl_algo=rl_algo)

        self.heads = nn.ModuleDict()
        self._class_weights = {}
        self._pos_weights = {}
        for name, spec in task_specs.items():
            if spec.task_type == "multilabel":
                self.heads[name] = nn.Linear(HIDDEN_DIM, spec.label_dim)
            elif spec.task_type == "binary":
                self.heads[name] = nn.Linear(HIDDEN_DIM, 2)
            elif spec.task_type == "multiclass":
                self.heads[name] = nn.Linear(HIDDEN_DIM, spec.label_dim)
            self._class_weights[name] = class_weights.get(name)
            self._pos_weights[name] = pos_weights.get(name)

    def forward_task(self, task_name: str, **kwargs) -> Dict[str, torch.Tensor]:
        spec = self.task_specs[task_name]
        old_task_type = self.encoder.task_type
        old_label_key = self.encoder.label_key
        self.encoder.task_type = spec.task_type
        self.encoder.label_key = spec.label_key

        last_state, rl_log_probs, rl_entropies, rl_baselines = self.encoder.encode(**kwargs)

        self.encoder.task_type = old_task_type
        self.encoder.label_key = old_label_key

        logit = self.heads[task_name](last_state)
        result = {"logit": logit}

        if spec.label_key in kwargs:
            y_true = kwargs[spec.label_key]
            result["y_true"] = y_true

            if spec.task_type == "multilabel":
                task_loss = F.binary_cross_entropy_with_logits(
                    logit, y_true, pos_weight=self._pos_weights.get(task_name),
                )
                result["y_prob"] = torch.sigmoid(logit)
            elif spec.task_type == "binary":
                task_loss = F.cross_entropy(logit, y_true)
                result["y_prob"] = F.softmax(logit, dim=-1)[:, 1]
            elif spec.task_type == "multiclass":
                task_loss = F.cross_entropy(
                    logit, y_true, weight=self._class_weights.get(task_name),
                )
                result["y_prob"] = F.softmax(logit, dim=-1)

            old_tt = self.encoder.task_type
            self.encoder.task_type = spec.task_type
            rl_loss = self.encoder._compute_rl_loss(
                rl_log_probs, rl_entropies, rl_baselines,
                task_loss.detach(), y_true, logit.detach(),
            )
            self.encoder.task_type = old_tt

            tw = TASK_LOSS_WEIGHTS.get(task_name, 1.0)
            result["loss"] = tw * task_loss + RL_LOSS_COEF * rl_loss
            result["task_loss"] = task_loss.detach()
            result["rl_loss"] = rl_loss.detach()

        return result


class _EvalWrapper(nn.Module):
    """Wraps MultiTaskWrapper so evaluate_model() can call model(**batch)."""

    def __init__(self, mt_wrapper: MultiTaskWrapper, task_name: str):
        super().__init__()
        self.mt = mt_wrapper
        self.task_name = task_name

    def forward(self, **kwargs):
        return self.mt.forward_task(self.task_name, **kwargs)

    def eval(self):
        self.mt.eval()
        return self

    def train(self, mode=True):
        self.mt.train(mode)
        return self


# ============================================================
# Reward Shaping — agent modifies reward design
# ============================================================

def shape_reward(
    metrics: Dict[str, float],
    task_spec: TaskSpec,
    ddi_matrix: Optional[np.ndarray] = None,
) -> float:
    """Custom reward shaping beyond the default compute_reward.

    Agent can add domain-specific reward components here,
    e.g. DDI penalty, diversity bonus, etc.
    """
    base_reward = compute_reward(metrics, task_spec)

    # Example: add DDI penalty for drug recommendation
    # (agent can modify this logic)
    if task_spec.name == "drug_recommendation" and "ddi_rate" in metrics:
        ddi_penalty = -0.5 * metrics["ddi_rate"]
        base_reward += ddi_penalty

    return base_reward


# ============================================================
# Training Loop
# ============================================================

def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Medical AutoResearch — RL training")
    parser.add_argument("--task", type=str, default=None,
                        help="Override TASK_NAME (e.g. mortality_prediction)")
    parser.add_argument("--time-budget", type=int, default=None,
                        help="Override training time budget in seconds")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--n-patients", type=int, default=2000,
                        help="Number of synthetic patients")
    parser.add_argument("--use-pyhealth", action="store_true",
                        help="Use real data via PyHealth (requires MIMIC)")
    parser.add_argument("--data-root", type=str, default=None,
                        help="MIMIC data root (for --use-pyhealth)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed")
    parser.add_argument("--rl-algo", type=str, default=None,
                        choices=["reinforce", "ppo", "a2c_gae", "dqn"],
                        help="Override RL algorithm")
    parser.add_argument("--multitask", action="store_true",
                        help="Run multi-task learning on 4 MIMIC-IV tasks")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)

    task_name = args.task or TASK_NAME
    time_budget = TIME_BUDGET_SECONDS if args.time_budget is None else args.time_budget
    batch_size = BATCH_SIZE if args.batch_size is None else args.batch_size
    rl_algo = args.rl_algo or RL_ALGO

    seed = args.seed if args.seed is not None else SEED
    set_seed(seed)
    print(f"[train] Seed: {seed}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # ---- Load task & data ----
    task_spec, train_loader, val_loader, test_loader = load_task_data(
        task_name, batch_size=batch_size,
        use_pyhealth=args.use_pyhealth, data_root=args.data_root,
        n_synthetic_patients=args.n_patients,
        return_spec=True,
    )
    print(f"[train] Task: {task_spec.name} ({task_spec.task_type})")
    print(f"[train] Primary metric: {task_spec.primary_metric} ({task_spec.metric_direction})")
    print(f"[train] Device: {device}")
    print(f"[train] Data: {len(train_loader.dataset)} train, "
          f"{len(val_loader.dataset)} val, {len(test_loader.dataset)} test")

    ddi_matrix = get_ddi_matrix(task_name) if task_name == "drug_recommendation" else None

    # ---- Compute class weights for imbalanced multiclass tasks ----
    class_weights = None
    pos_weight = None
    if task_spec.task_type == "multilabel":
        # Per-label sqrt inverse-frequency pos_weight for BCE.
        # Labels with rare positives (e.g. GI hemorrhage 1.2%) get up-weighted.
        label_key = task_spec.label_key
        n_labels = task_spec.label_dim
        pos_counts = torch.zeros(n_labels)
        n_total = 0
        for batch in train_loader:
            y = batch[label_key]
            pos_counts += y.sum(dim=0)
            n_total += y.shape[0]
        neg_counts = n_total - pos_counts
        # pos_weight = sqrt(neg / pos), clipped so a single positive label
        # cannot blow the loss up.
        ratio = neg_counts / pos_counts.clamp(min=1.0)
        pw = ratio.clamp(min=1.0).sqrt().clamp(max=10.0)
        pos_weight = pw.to(device)
        print(f"[train] Label positive rates: {(pos_counts / max(n_total, 1)).tolist()}")
        print(f"[train] BCE pos_weight (sqrt, clipped): {pw.tolist()}")
    if task_spec.task_type == "multiclass":
        from collections import Counter
        label_key = task_spec.label_key
        all_labels = []
        for batch in train_loader:
            all_labels.append(batch[label_key].numpy())
        all_labels = np.concatenate(all_labels)
        counts = Counter(all_labels.tolist())
        n_classes = task_spec.label_dim
        total = len(all_labels)
        # Sqrt inverse frequency weighting (less extreme than linear)
        weights = torch.zeros(n_classes)
        for c in range(n_classes):
            weights[c] = (total / (n_classes * max(counts.get(c, 1), 1))) ** 0.5
        class_weights = weights.to(device)
        print(f"[train] Class distribution: {dict(sorted(counts.items()))}")
        print(f"[train] Class weights: {weights.tolist()}")

    # ---- Build model ----
    print(f"[train] RL algorithm: {rl_algo}")
    model = ClinicalRLModel(task_spec, class_weights=class_weights, pos_weight=pos_weight, rl_algo=rl_algo).to(device)
    n_params = count_parameters(model)
    print(f"[train] Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
    )
    if task_spec.task_type == "multilabel":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        scheduler_uses_score = False
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2,
        )
        scheduler_uses_score = True

    # ---- Training loop with time budget ----
    total_start = time.time()
    training_start = time.time()
    step = 0
    epoch = 0
    best_score = -float("inf") if task_spec.metric_direction == "max" else float("inf")
    best_state = None

    print(f"[train] Training for {time_budget}s...")

    while time.time() - training_start < time_budget:
        model.train()
        epoch_loss = 0.0
        epoch_task_loss = 0.0
        epoch_rl_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            if time.time() - training_start >= time_budget:
                break

            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            optimizer.zero_grad()
            output = model(**batch)
            loss = output["loss"]
            loss.backward()

            if MAX_GRAD_NORM > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), MAX_GRAD_NORM,
                )
            optimizer.step()

            epoch_loss += loss.item()
            epoch_task_loss += output.get("task_loss", loss).item()
            epoch_rl_loss += output.get("rl_loss", torch.tensor(0.0)).item()
            n_batches += 1
            step += 1


        if n_batches == 0:
            break

        epoch += 1
        avg_loss = epoch_loss / n_batches
        avg_task = epoch_task_loss / n_batches
        avg_rl = epoch_rl_loss / n_batches

        # Validate every epoch
        val_metrics = evaluate_model(model, val_loader, task_spec, device)
        score = val_metrics.get(task_spec.primary_metric, 0.0)

        is_better = (
            (score > best_score) if task_spec.metric_direction == "max"
            else (score < best_score)
        )
        if is_better:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if scheduler_uses_score:
            scheduler.step(score)
        else:
            scheduler.step()

        # DQN: update target network and decay epsilon
        if rl_algo == "dqn":
            if epoch % DQN_TARGET_UPDATE == 0:
                model.rl_agent.update_target()
            model.rl_agent.decay_epsilon()
            # Update replay buffer rewards with validation score as delayed reward
            if hasattr(model, "replay_buffer") and len(model.replay_buffer) > 0:
                reward_val = torch.tensor(score, dtype=torch.float32)
                for i in range(len(model.replay_buffer.buffer)):
                    if model.replay_buffer.buffer[i] is not None:
                        s, a, _, ns = model.replay_buffer.buffer[i]
                        model.replay_buffer.buffer[i] = (s, a, reward_val, ns)

        elapsed = time.time() - training_start
        eps_str = f" | eps {model.rl_agent.epsilon:.3f}" if rl_algo == "dqn" else ""
        print(
            f"[train] Epoch {epoch:3d} | loss {avg_loss:.4f} "
            f"(task {avg_task:.4f} + rl {avg_rl:.4f}) | "
            f"val_{task_spec.primary_metric} {score:.4f} | "
            f"best {best_score:.4f} | {elapsed:.0f}s/{time_budget}s{eps_str}"
        )

    training_seconds = time.time() - training_start

    # ---- Load best model & evaluate on test ----
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    test_metrics = evaluate_model(model, test_loader, task_spec, device)

    # Add DDI rate for drug recommendation
    if ddi_matrix is not None:
        # Collect predictions for DDI computation
        model.eval()
        all_preds = []
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                out = model(**batch)
                if "y_prob" in out:
                    all_preds.append((out["y_prob"].cpu().numpy() > 0.5).astype(float))
        if all_preds:
            y_pred_all = np.concatenate(all_preds)
            test_metrics["ddi_rate"] = compute_ddi_rate(y_pred_all, ddi_matrix)

    total_seconds = time.time() - total_start

    # ---- Print results ----
    print_results(
        metrics=test_metrics,
        task_spec=task_spec,
        training_seconds=training_seconds,
        total_seconds=total_seconds,
        peak_vram_mb=get_peak_vram_mb(),
        num_params=n_params,
        epochs=epoch,
        steps=step,
    )


def main_multitask(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    time_budget = TIME_BUDGET_SECONDS if args.time_budget is None else args.time_budget
    batch_size = BATCH_SIZE if args.batch_size is None else args.batch_size
    rl_algo = args.rl_algo or RL_ALGO
    seed = args.seed if args.seed is not None else SEED
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print(f"[multitask] Seed: {seed}, device: {device}")
    print(f"[multitask] Tasks: {MULTITASK_TASKS}")

    # Load all tasks
    task_specs: Dict[str, TaskSpec] = {}
    val_loaders: Dict[str, object] = {}
    test_loaders: Dict[str, object] = {}
    train_loaders: Dict[str, object] = {}

    for tname in MULTITASK_TASKS:
        spec, tr, va, te = load_task_data(tname, batch_size=batch_size, return_spec=True)
        task_specs[tname] = spec
        train_loaders[tname] = tr
        val_loaders[tname] = va
        test_loaders[tname] = te
        print(f"[multitask]   {tname}: {len(tr.dataset)} train, {len(va.dataset)} val, {len(te.dataset)} test")

    # Compute per-task class_weights / pos_weights
    all_class_weights: Dict[str, Optional[torch.Tensor]] = {}
    all_pos_weights: Dict[str, Optional[torch.Tensor]] = {}
    for tname, spec in task_specs.items():
        if spec.task_type == "multilabel":
            pos_counts = torch.zeros(spec.label_dim)
            n_total = 0
            for batch in train_loaders[tname]:
                y = batch[spec.label_key]
                pos_counts += y.sum(dim=0)
                n_total += y.shape[0]
            neg_counts = n_total - pos_counts
            ratio = neg_counts / pos_counts.clamp(min=1.0)
            pw = ratio.clamp(min=1.0).sqrt().clamp(max=10.0).to(device)
            all_pos_weights[tname] = pw
            all_class_weights[tname] = None
        elif spec.task_type == "multiclass":
            from collections import Counter
            all_labels = []
            for batch in train_loaders[tname]:
                all_labels.append(batch[spec.label_key].numpy())
            all_labels = np.concatenate(all_labels)
            counts = Counter(all_labels.tolist())
            total = len(all_labels)
            weights = torch.zeros(spec.label_dim)
            for c in range(spec.label_dim):
                weights[c] = (total / (spec.label_dim * max(counts.get(c, 1), 1))) ** 0.5
            all_class_weights[tname] = weights.to(device)
            all_pos_weights[tname] = None
        else:
            all_class_weights[tname] = None
            all_pos_weights[tname] = None

    # Build multi-task model
    model = MultiTaskWrapper(task_specs, all_class_weights, all_pos_weights, rl_algo=rl_algo).to(device)
    n_params = count_parameters(model)
    print(f"[multitask] Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    # Round-robin infinite iterators
    task_iters = {}
    for tname in MULTITASK_TASKS:
        task_iters[tname] = iter(itertools.cycle(train_loaders[tname]))

    # Training loop
    total_start = time.time()
    training_start = time.time()
    step = 0
    epoch = 0
    steps_per_epoch = max(len(train_loaders[t]) for t in MULTITASK_TASKS)
    best_combined = -float("inf")
    best_state = None

    print(f"[multitask] Training for {time_budget}s, ~{steps_per_epoch} steps/epoch...", flush=True)

    while time.time() - training_start < time_budget:
        model.train()
        epoch_losses = {t: [] for t in MULTITASK_TASKS}

        task_schedule = []
        for t in MULTITASK_TASKS:
            task_schedule.extend([t] * TASK_SAMPLE_WEIGHTS.get(t, 1))

        for i in range(steps_per_epoch):
            if time.time() - training_start >= time_budget:
                break
            tname = task_schedule[i % len(task_schedule)]
            batch = next(task_iters[tname])
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            optimizer.zero_grad()
            output = model.forward_task(tname, **batch)
            loss = output["loss"]
            loss.backward()
            if MAX_GRAD_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            epoch_losses[tname].append(output["task_loss"].item())
            step += 1

            if step % 500 == 0:
                elapsed = time.time() - training_start
                print(f"[multitask]   step {step} | {tname} loss {output['task_loss'].item():.4f} | {elapsed:.0f}s", flush=True)

        if step == 0:
            break

        epoch += 1
        scheduler.step()

        # Validate all tasks
        val_scores = {}
        for tname, spec in task_specs.items():
            wrapper = _EvalWrapper(model, tname)
            metrics = evaluate_model(wrapper, val_loaders[tname], spec, device)
            val_scores[tname] = metrics.get(spec.primary_metric, 0.0)

        combined = np.mean(list(val_scores.values()))
        if combined > best_combined:
            best_combined = combined
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, "results/multitask_best.pt")

        elapsed = time.time() - training_start
        loss_strs = " | ".join(
            f"{t.split('_', 1)[1][:4]} {np.mean(epoch_losses[t]):.4f}/{val_scores[t]:.4f}"
            for t in MULTITASK_TASKS
        )
        print(f"[multitask] Epoch {epoch:3d} | {loss_strs} | combined {combined:.4f} | best {best_combined:.4f} | {elapsed:.0f}s/{time_budget}s", flush=True)

    training_seconds = time.time() - training_start

    # Load best & test
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    print("\n===== Multi-Task Test Results =====")
    for tname, spec in task_specs.items():
        wrapper = _EvalWrapper(model, tname)
        test_metrics = evaluate_model(wrapper, test_loaders[tname], spec, device)
        score = test_metrics.get(spec.primary_metric, 0.0)
        print(f"  {tname}: {spec.primary_metric} = {score:.4f}")
        for k, v in sorted(test_metrics.items()):
            if k != spec.primary_metric:
                print(f"    {k} = {v:.4f}")

    total_seconds = time.time() - total_start
    print(f"\n[multitask] {epoch} epochs, {step} steps, {training_seconds:.0f}s training, {total_seconds:.0f}s total")
    print(f"[multitask] Peak VRAM: {get_peak_vram_mb():.0f} MB")
    print(f"[multitask] Parameters: {n_params:,}")
    print(f"[multitask] Best combined val score: {best_combined:.4f}")

    # Print in a format the loop can parse
    for tname, spec in task_specs.items():
        wrapper = _EvalWrapper(model, tname)
        test_metrics = evaluate_model(wrapper, test_loaders[tname], spec, device)
        score = test_metrics.get(spec.primary_metric, 0.0)
        print(f"primary_metric\t{tname}\t{score:.6f}")


if __name__ == "__main__":
    args = parse_args()
    if getattr(args, "multitask", False):
        main_multitask()
    else:
        main()
