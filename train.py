"""Clinical RL model + training loop. THE AGENT MODIFIES THIS FILE.

Everything is fair game: model architecture, reward shaping, RL algorithm,
hyperparameters, training strategy, multi-task selection.

Goal: maximize the primary metric for the chosen task(s).
"""

import argparse
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
HIDDEN_DIM = 128
NUM_RNN_LAYERS = 1
DROPOUT = 0.3

# RL hyperparameters
GAMMA = 0.95          # discount factor for delayed rewards
ENTROPY_COEF = 0.01   # exploration bonus
VALUE_LOSS_COEF = 0.5 # value network weight
USE_BASELINE = True   # variance reduction
N_ACTIONS = 10        # history window for agent attention

# Optimization
LR = 1e-3
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 64
MAX_GRAD_NORM = 1.0

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
        emb = self.embedding(x)            # [batch, visits, codes, embed_dim]
        emb = self.dropout(emb)
        # Sum over codes dimension (ignore padding via embedding zero)
        return emb.sum(dim=2)              # [batch, visits, embed_dim]


class PolicyAgent(nn.Module):
    """RL agent that selects relevant historical states.

    At each timestep, observes the current context and chooses
    which historical hidden state to attend to. This implements
    a learned attention mechanism via REINFORCE.
    """
    def __init__(
        self,
        obs_dim: int,
        n_actions: int = 10,
        hidden_dim: int = 64,
        use_baseline: bool = True,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.use_baseline = use_baseline

        self.policy_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_actions),
        )
        if use_baseline:
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
        logits = self.policy_net(observation.detach())
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        baseline = None
        if self.use_baseline:
            baseline = self.value_net(observation.detach()).squeeze(-1)

        return action, log_prob, entropy, baseline


class ClinicalRLModel(nn.Module):
    """RL-based clinical prediction model.

    Architecture:
    1. Embed each feature type (conditions, procedures, etc.)
    2. Process visit sequence with GRU
    3. Use RL agent to select relevant historical states
    4. Predict output (drug combination / mortality / etc.)

    The agent can modify any part of this architecture.
    """
    def __init__(self, task_spec: TaskSpec, class_weights=None):
        super().__init__()
        self.spec = task_spec
        self.task_type = task_spec.task_type
        self.label_key = task_spec.label_key
        self.class_weights = class_weights

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
        )

        # Fusion layer: combines current state + agent-selected history
        self.fusion = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
        )

        # Task-specific output head
        if task_spec.task_type == "multilabel":
            self.output_head = nn.Linear(HIDDEN_DIM, task_spec.label_dim)
        elif task_spec.task_type == "binary":
            self.output_head = nn.Linear(HIDDEN_DIM, 2)
        elif task_spec.task_type == "multiclass":
            self.output_head = nn.Linear(HIDDEN_DIM, task_spec.label_dim)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device

        # 1. Embed and concatenate features
        embedded = []
        for key in self.spec.feature_keys:
            if key in kwargs:
                embedded.append(self.embeddings[key](kwargs[key]))
        x = torch.cat(embedded, dim=-1)  # [batch, visits, input_dim]

        mask = kwargs.get("mask")  # [batch, visits]

        # 2. Encode with RNN
        rnn_out, _ = self.rnn(x)  # [batch, visits, hidden_dim]
        batch_size, seq_len, hidden_dim = rnn_out.shape

        # 3. RL agent: select relevant historical states at each step
        rl_log_probs = []
        rl_entropies = []
        rl_baselines = []
        fused_states = []

        for t in range(seq_len):
            current = rnn_out[:, t, :]  # [batch, hidden_dim]

            if t == 0:
                # No history at first step, use current as both
                fused = self.fusion(torch.cat([current, current], dim=-1))
            else:
                # Agent selects from history buffer
                history_len = min(t, N_ACTIONS)
                history_start = max(0, t - N_ACTIONS)
                history = rnn_out[:, history_start:t, :]  # [batch, <=N_ACTIONS, hidden]

                action, log_prob, entropy, baseline = self.rl_agent(current)
                rl_log_probs.append(log_prob)
                rl_entropies.append(entropy)
                if baseline is not None:
                    rl_baselines.append(baseline)

                # Clamp action to valid range
                action_clamped = action.clamp(0, history_len - 1)
                # Gather selected historical state
                idx = action_clamped.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, hidden_dim)
                selected = history.gather(1, idx).squeeze(1)  # [batch, hidden]

                fused = self.fusion(torch.cat([current, selected], dim=-1))

            fused_states.append(fused)

        fused_seq = torch.stack(fused_states, dim=1)  # [batch, visits, hidden]

        # 4. Get last valid state for prediction
        if mask is not None:
            lengths = mask.long().sum(dim=1).clamp(min=1) - 1
            last_state = fused_seq[
                torch.arange(batch_size, device=device), lengths
            ]
        else:
            last_state = fused_seq[:, -1, :]

        # 5. Predict
        logit = self.output_head(last_state)

        # 6. Compute loss
        result = {"logit": logit}

        if self.label_key in kwargs:
            y_true = kwargs[self.label_key]
            result["y_true"] = y_true

            # Task loss
            if self.task_type == "multilabel":
                task_loss = F.binary_cross_entropy_with_logits(logit, y_true)
                result["y_prob"] = torch.sigmoid(logit)
            elif self.task_type == "binary":
                task_loss = F.cross_entropy(logit, y_true)
                result["y_prob"] = F.softmax(logit, dim=-1)[:, 1]
            elif self.task_type == "multiclass":
                task_loss = F.cross_entropy(logit, y_true, weight=self.class_weights)
                result["y_prob"] = F.softmax(logit, dim=-1)

            # RL loss (REINFORCE with baseline)
            rl_loss = self._compute_rl_loss(
                rl_log_probs, rl_entropies, rl_baselines,
                task_loss.detach(), y_true, logit.detach(),
            )

            result["loss"] = task_loss + rl_loss
            result["task_loss"] = task_loss.detach()
            result["rl_loss"] = rl_loss.detach()

        return result

    def _compute_rl_loss(
        self,
        log_probs: List[torch.Tensor],
        entropies: List[torch.Tensor],
        baselines: List[torch.Tensor],
        task_loss: torch.Tensor,
        y_true: torch.Tensor,
        logit: torch.Tensor,
    ) -> torch.Tensor:
        """Compute RL loss using REINFORCE with reward shaping.

        Reward signal is derived from prediction quality:
        - multilabel: per-sample Jaccard similarity
        - binary/multiclass: correctness indicator
        """
        if not log_probs:
            return torch.tensor(0.0, device=logit.device)

        # Compute per-sample reward
        with torch.no_grad():
            if self.task_type == "multilabel":
                y_pred = (torch.sigmoid(logit) > 0.5).float()
                intersection = (y_pred * y_true).sum(dim=-1)
                union = ((y_pred + y_true) > 0).float().sum(dim=-1)
                reward = intersection / union.clamp(min=1)
            elif self.task_type == "binary":
                pred_class = logit.argmax(dim=-1)
                reward = (pred_class == y_true).float()
            elif self.task_type == "multiclass":
                pred_class = logit.argmax(dim=-1)
                reward = (pred_class == y_true).float()

        # Discount rewards backward through time
        T = len(log_probs)
        discounted = torch.zeros(T, reward.shape[0], device=logit.device)
        discounted[-1] = reward
        for t in reversed(range(T - 1)):
            discounted[t] = reward + GAMMA * discounted[t + 1]

        # REINFORCE loss
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

        total_rl_loss = (
            policy_loss / max(T, 1)
            + VALUE_LOSS_COEF * value_loss / max(T, 1)
            - ENTROPY_COEF * entropy_bonus / max(T, 1)
        )

        return total_rl_loss


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
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)

    task_name = args.task or TASK_NAME
    time_budget = TIME_BUDGET_SECONDS if args.time_budget is None else args.time_budget
    batch_size = BATCH_SIZE if args.batch_size is None else args.batch_size

    set_seed(SEED)
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
        # Inverse frequency weighting
        weights = torch.zeros(n_classes)
        for c in range(n_classes):
            weights[c] = total / (n_classes * max(counts.get(c, 1), 1))
        class_weights = weights.to(device)
        print(f"[train] Class distribution: {dict(sorted(counts.items()))}")
        print(f"[train] Class weights: {weights.tolist()}")

    # ---- Build model ----
    model = ClinicalRLModel(task_spec, class_weights=class_weights).to(device)
    n_params = count_parameters(model)
    print(f"[train] Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2,
    )

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

        scheduler.step(score)

        elapsed = time.time() - training_start
        print(
            f"[train] Epoch {epoch:3d} | loss {avg_loss:.4f} "
            f"(task {avg_task:.4f} + rl {avg_rl:.4f}) | "
            f"val_{task_spec.primary_metric} {score:.4f} | "
            f"best {best_score:.4f} | {elapsed:.0f}s/{time_budget}s"
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


if __name__ == "__main__":
    main()
