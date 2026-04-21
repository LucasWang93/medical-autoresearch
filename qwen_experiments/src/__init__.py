"""qwen_experiments — LLM-based clinical prediction pipeline.

Three tracks:
  A. Few-shot baseline (inference only, no training)
  B. Single-task GRPO LoRA + autonomous iter loop
  C. Multi-task GRPO LoRA (shared adapter, per-task group normalization)

Reuses medical-autoresearch MIMIC-IV data loaders; produces fully traceable
run directories and an append-only leaderboard.tsv.
"""
