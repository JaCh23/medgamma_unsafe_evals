"""Shared paths, v1 result-file registry and loaders for the v2 revision notebooks."""
import os, json, math, hashlib, collections, pathlib
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent          # the folder containing this file
DATA_DIR = HERE.parent / "data"                         # local folder holding curated/, results/ and model_res/ (read-only)
D = str(DATA_DIR) + "/"

DATA_V2 = HERE / "data_v2"; DATA_V2.mkdir(exist_ok=True)   # every file created by this extension lives here
GOLD_FILE = str(DATA_V2 / "gold_set_excl_degenerate100.jsonl")
REVIEW_FILE = str(DATA_V2 / "gold_set_excl_degenerate100_clinician_review.csv")
OUT_DIR = DATA_V2 / "outputs"; OUT_DIR.mkdir(exist_ok=True)   # scored tables written by notebooks 02-06
RUNS_DIR = DATA_V2 / "runs"                                   # raw model responses written by notebook 05

MODELS = {  # name: (Open-Ended file, MCQ file)
    "Llama-3.3-70B":    ("results/exp2_nb_meta_llama3.3_results.jsonl",    "model_res/llama3_70b_results.jsonl"),
    "Deepseek-v3.1":    ("results/exp2_nb_alibaba_deepseek_results.jsonl", "model_res/deepseek_results.jsonl"),
    "Mistral-Small":    ("results/exp2_nb_mistral_results.jsonl",          "model_res/mistral_small_results.jsonl"),
    "Gemini-2.5-Flash": ("results/exp2_nb_google_gemini_results.jsonl",    "model_res/gemini_flash_results.jsonl"),
    "Claude-3-Haiku":   ("results/exp2_nb4_claude_haiku_results.jsonl",    "model_res/claude_haiku_results.jsonl"),
    "GPT-4o-mini":      ("results/exp2_nb4_gpt_baseline_results.jsonl",    "model_res/gpt_baseline_results.jsonl"),
    "Qwen-2.5-7B":      ("results/exp2_nb_alibaba_qwen_results.jsonl",     "model_res/qwen_results.jsonl"),
    "MedGemma-1.5-4B":  ("results/exp2_medgemma_results.jsonl",            "model_res/medgemma_results.jsonl"),
}


def load_gold_file(path=GOLD_FILE):
    """The exported 54-item set. Raises a clear error if notebook 01 has not been run."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found: run 01_curate_gold_set.ipynb first")
    gold = [json.loads(l) for l in open(path, encoding="utf-8")]
    return gold


def gold_sha(path=GOLD_FILE):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]


def rd(f):
    return [json.loads(l) for l in open(D + f, encoding="utf-8")]


def resp_key(r):
    return next(k for k in ("model_response", "gpt_response", "model_raw_response") if k in r)


def choice_key(r):
    return next(k for k in r if k.endswith("choice"))


def rows_for(model, cond, gold):
    """Result rows aligned to the gold set. MCQ files already hold the 54 in order; OE files hold 147, subset via idx147."""
    r = rd(MODELS[model][0 if cond == "OE" else 1])
    if cond == "OE":
        r = [r[g["idx147"]] for g in gold]
    return r


def duplicate_oe_models():
    """Models whose Open-Ended file is byte-identical to another model's (invalid as independent runs)."""
    h = {m: hashlib.md5("".join(str(x[resp_key(x)]) for x in rd(MODELS[m][0])).encode()).hexdigest() for m in MODELS}
    groups = collections.defaultdict(list)
    for m, v in h.items():
        groups[v].append(m)
    return {m for g in groups.values() if len(g) > 1 for m in g}


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); a = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - a) / d * 100, (c + a) / d * 100