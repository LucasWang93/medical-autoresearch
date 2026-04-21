"""Prompt builders + answer parsers for the 5 MIMIC-IV tasks.

Design:
  * Each task has a fixed system prompt describing the task and required
    answer format.
  * The user prompt is a compact textual render of the patient's visit
    history, truncated to keep within context.
  * Binary tasks (mortality, readmission) are answered with 'yes'/'no'
    so we can extract a probability from logits at the answer position.
  * Multiclass LOS answered with '<3d' / '3-7d' / '7-14d' / '>14d'.
  * Multilabel phenotyping answered as a JSON list of phenotype keys
    (small vocabulary — 25 entries).
  * Drug recommendation answered as a JSON list of drug name strings.

The few-shot baseline selects K demonstrations from the train split.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from .data import TextSample


LOS_OPTIONS = ["<3d", "3-7d", "7-14d", ">14d"]

# 25 phenotype keys — must match prepare.py _MIMIC4_PHENOTYPES order
PHENOTYPES = [
    "acute_renal_failure",
    "acute_cerebrovascular",
    "acute_mi",
    "cardiac_dysrhythmias",
    "chronic_kidney_disease",
    "copd",
    "complications_of_care",
    "conduction_disorders",
    "chf",
    "coronary_atherosclerosis",
    "dm_complicated",
    "dm_uncomplicated",
    "lipid_metabolism",
    "essential_hypertension",
    "fluid_electrolyte",
    "gi_hemorrhage",
    "hypertension_complicated",
    "liver_disease",
    "other_lower_respiratory",
    "other_upper_respiratory",
    "pleurisy_pneumothorax",
    "pneumonia",
    "respiratory_failure",
    "septicemia",
    "shock",
]


# ------- Rendering the visit sequence -------

def render_visits(
    visits: List[Dict[str, List[str]]],
    current_idx: int,
    max_visits: int = 10,
    max_codes_per_visit: int = 20,
) -> str:
    """Text render; keeps the most recent `max_visits` visits."""
    start = max(0, len(visits) - max_visits)
    lines = []
    for t in range(start, len(visits)):
        v = visits[t]
        cond = v.get("conditions", [])[:max_codes_per_visit]
        proc = v.get("procedures", [])[:max_codes_per_visit]
        drugs = v.get("drugs", [])[:max_codes_per_visit] if "drugs" in v else None
        tag = "[current]" if t == current_idx else f"[v{t+1}]"
        line = f"{tag} icd_dx: {', '.join(cond) if cond else '—'}"
        if proc:
            line += f" | icd_px: {', '.join(proc)}"
        if drugs is not None and drugs:
            line += f" | rx_hist: {', '.join(drugs)}"
        lines.append(line)
    return "\n".join(lines)


# ------- Per-task system prompts -------

SYSTEM_PROMPTS = {
    "mimic4_mortality": (
        "You are a clinical prediction assistant. Given a patient's ICD "
        "diagnosis and procedure history, predict in-hospital mortality "
        "for the current admission. Answer with a single word: 'yes' "
        "(patient dies) or 'no' (patient survives)."
    ),
    "mimic4_readmission": (
        "You are a clinical prediction assistant. Given a patient's ICD "
        "diagnosis and procedure history, predict whether the patient "
        "will be readmitted to the hospital within 30 days of discharge. "
        "Answer with a single word: 'yes' or 'no'."
    ),
    "mimic4_los": (
        "You are a clinical prediction assistant. Given a patient's ICD "
        "diagnosis and procedure history, predict the length of stay "
        "bucket for the current admission. Answer with exactly one of: "
        f"{', '.join(LOS_OPTIONS)}."
    ),
    "mimic4_phenotyping": (
        "You are a clinical prediction assistant. Given a patient's past "
        "ICD history, forecast which of the following 25 phenotypes "
        "will be diagnosed in the next admission. "
        "Phenotype keys: " + ", ".join(PHENOTYPES) + ". "
        "Answer with a JSON list of phenotype keys, e.g. "
        '["chf", "pneumonia"]. Output [] if none are expected.'
    ),
    "mimic4_drugrec": (
        "You are a clinical prediction assistant. Given a patient's ICD "
        "history and past medications, recommend the drug list for the "
        "current admission. Answer with a JSON list of drug name "
        'strings (exact lower-case names, up to 25), e.g. ["metformin", '
        '"lisinopril"].'
    ),
}


# ------- Answer format descriptors -------

def answer_format(task: str) -> str:
    if task in ("mimic4_mortality", "mimic4_readmission"):
        return "Answer: yes / no"
    if task == "mimic4_los":
        return f"Answer: one of {LOS_OPTIONS}"
    if task == "mimic4_phenotyping":
        return 'Answer: JSON list of phenotype keys, e.g. ["chf", "pneumonia"]'
    if task == "mimic4_drugrec":
        return 'Answer: JSON list of drug names, e.g. ["metformin", "lisinopril"]'
    raise ValueError(task)


# ------- User prompt builder -------

def build_user_prompt(
    sample: TextSample,
    max_visits: int = 10,
    max_codes_per_visit: int = 20,
) -> str:
    visit_block = render_visits(
        sample.visits,
        sample.current_idx,
        max_visits=max_visits,
        max_codes_per_visit=max_codes_per_visit,
    )
    return (
        "Patient visit history (chronological):\n"
        f"{visit_block}\n\n"
        f"{answer_format(sample.task)}"
    )


def label_as_answer(task: str, label: Any) -> str:
    if task == "mimic4_mortality":
        return "yes" if int(label) == 1 else "no"
    if task == "mimic4_readmission":
        return "yes" if int(label) == 1 else "no"
    if task == "mimic4_los":
        return LOS_OPTIONS[int(label)]
    if task == "mimic4_phenotyping":
        if isinstance(label, list) and label and isinstance(label[0], int):
            keys = [PHENOTYPES[i] for i, v in enumerate(label) if v]
        else:
            keys = list(label or [])
        return json.dumps(keys)
    if task == "mimic4_drugrec":
        drugs = list(label or [])[:25]
        return json.dumps([d.lower() for d in drugs])
    raise ValueError(task)


def build_chat_messages(
    sample: TextSample,
    demos: Optional[List[TextSample]] = None,
    max_visits: int = 10,
    max_codes_per_visit: int = 20,
) -> List[Dict[str, str]]:
    """Few-shot-style chat messages. Demos come before the target user turn."""
    messages = [{"role": "system", "content": SYSTEM_PROMPTS[sample.task]}]
    if demos:
        for d in demos:
            messages.append({
                "role": "user",
                "content": build_user_prompt(d, max_visits, max_codes_per_visit),
            })
            messages.append({
                "role": "assistant",
                "content": label_as_answer(d.task, d.label),
            })
    messages.append({
        "role": "user",
        "content": build_user_prompt(sample, max_visits, max_codes_per_visit),
    })
    return messages


# ------- Answer parsers -------

_YES_PAT = re.compile(r"\b(yes|positive|mortality|die|readmit(ted)?)\b", re.I)
_NO_PAT = re.compile(r"\b(no|negative|survive|not)\b", re.I)


def parse_binary(text: str) -> Optional[int]:
    t = text.strip().lower()
    if t.startswith("yes"):
        return 1
    if t.startswith("no"):
        return 0
    if _YES_PAT.search(t) and not _NO_PAT.search(t):
        return 1
    if _NO_PAT.search(t) and not _YES_PAT.search(t):
        return 0
    return None


def parse_los(text: str) -> Optional[int]:
    t = text.strip().lower().replace(" ", "")
    for i, opt in enumerate(LOS_OPTIONS):
        if opt.lower().replace(" ", "") in t:
            return i
    return None


def _extract_json_list(text: str) -> Optional[List[str]]:
    m = re.search(r"\[.*?\]", text, re.DOTALL)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
        if isinstance(arr, list):
            return [str(x) for x in arr]
    except Exception:
        return None
    return None


def parse_phenotypes(text: str) -> Optional[List[int]]:
    items = _extract_json_list(text)
    if items is None:
        # fallback: search for phenotype keywords directly
        items = [p for p in PHENOTYPES if re.search(r"\b" + re.escape(p) + r"\b", text, re.I)]
        if not items:
            return None
    multihot = [0] * len(PHENOTYPES)
    pset = set(PHENOTYPES)
    for x in items:
        k = x.strip().lower()
        if k in pset:
            multihot[PHENOTYPES.index(k)] = 1
    return multihot


def parse_drugs(text: str, drug_vocab: Optional[List[str]] = None) -> Optional[List[str]]:
    items = _extract_json_list(text)
    if items is None:
        return None
    cleaned = []
    for x in items:
        s = x.strip().lower()
        if not s:
            continue
        if drug_vocab is not None and s not in drug_vocab:
            # keep anyway — Jaccard against true list tolerates OOV
            pass
        cleaned.append(s)
    return cleaned


def parse_answer(task: str, text: str) -> Any:
    if task in ("mimic4_mortality", "mimic4_readmission"):
        return parse_binary(text)
    if task == "mimic4_los":
        return parse_los(text)
    if task == "mimic4_phenotyping":
        return parse_phenotypes(text)
    if task == "mimic4_drugrec":
        return parse_drugs(text)
    raise ValueError(task)
