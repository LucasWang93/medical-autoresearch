"""Quick current-state snapshot of all iter 2 runs."""
import json, glob, os, sys

ITER = "iter02" if len(sys.argv) < 2 else sys.argv[1]
root = "/nfs/roberts/project/pi_yz875/sw2572/codes/auto-research-health/medical-autoresearch/qwen_experiments/runs"

# Baseline iter 0 for comparison
BASELINE = {
    ("qwen35-4b","mimic4_mortality"): 0.924,
    ("qwen35-4b","mimic4_readmission"): 0.519,
    ("qwen35-4b","mimic4_los"): 0.182,
    ("qwen35-4b","mimic4_phenotyping"): 0.133,
    ("qwen35-4b","mimic4_drugrec"): 0.180,
    ("qwen35-9b","mimic4_mortality"): 0.505,
    ("qwen35-9b","mimic4_readmission"): 0.500,
    ("qwen35-9b","mimic4_los"): 0.182,
    ("qwen35-9b","mimic4_phenotyping"): 0.130,
    ("qwen35-9b","mimic4_drugrec"): 0.180,
}

print(f"{'run_id':80s} {'steps':>5s} {'evals':>5s} {'parse_last':>10s} {'prim_last':>10s} {'vs base':>8s}")
print("-" * 130)

dirs = sorted(glob.glob(f"{root}/B_single/*{ITER}*") + glob.glob(f"{root}/C_multi/*{ITER}*"))
for d in dirs:
    tl = os.path.join(d, "train_log.jsonl")
    if not os.path.exists(tl):
        continue
    steps, evals = [], []
    for ln in open(tl):
        try: r = json.loads(ln)
        except: continue
        if r.get("event") == "eval":
            evals.append(r)
        elif "step" in r and "parse_rate" in r:
            steps.append(r)
    name = os.path.basename(d)
    # Derive model_tag + task from name
    parts = name.split("__")
    mtag = parts[0]
    task = parts[1] if len(parts) > 1 else "multi"
    base = BASELINE.get((mtag, task), None)
    last_eval = evals[-1] if evals else None
    primary_keys = ("auroc","f1_macro","f1_samples","jaccard")
    if last_eval:
        prim_v = None
        for k in primary_keys:
            if k in last_eval:
                prim_v = last_eval[k]; break
        pr = last_eval.get("parse_rate", float("nan"))
    else:
        prim_v = None
        pr = steps[-1].get("parse_rate") if steps else None
    prim_s = f"{prim_v:.3f}" if isinstance(prim_v, float) else "-"
    pr_s = f"{pr:.3f}" if isinstance(pr, float) else "-"
    delta = f"{prim_v - base:+.3f}" if (base is not None and isinstance(prim_v, float)) else ""
    print(f"{name:80s} {len(steps):>5d} {len(evals):>5d} {pr_s:>10s} {prim_s:>10s} {delta:>8s}")
