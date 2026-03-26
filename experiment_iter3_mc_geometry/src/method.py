#!/usr/bin/env python3
"""Phase 1: Monte Carlo Estimator Validation & Geometric Confound Analysis.

Task A: Monte Carlo simulation of 1,020,000 AR(1) sequences across 6 lengths x 17 phi
values to compare 3 autocorrelation estimators, establish minimum detectable effects,
and verify bias-cancellation in excess measures.

Task B: Three-pronged geometric confound analysis on synthetic canonical trees and real
UD dependency trees to determine whether tree geometry alone trivially explains observed
anti-correlation through all baseline tiers.
"""

import gc
import hashlib
import itertools
import json
import math
import os
import resource
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import scipy.stats
from scipy.optimize import minimize_scalar
from loguru import logger

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════════════
WORKSPACE = Path(
    "/ai-inventor/aii_pipeline/runs/comp-ling-dobrovoljc_osk"
    "/3_invention_loop/iter_3/gen_art/exp_id1_it3__opus"
)
DATA_DEP_PATH = Path(
    "/ai-inventor/aii_pipeline/runs/comp-ling-dobrovoljc_ebw"
    "/3_invention_loop/iter_1/gen_art/data_id3_it1__opus"
)

LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ═══════════════════════════════════════════════════════════════════════════════
# HARDWARE DETECTION & RESOURCE LIMITS
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_cpus() -> int:
    """Detect actual CPU allocation (container-aware)."""
    try:
        parts = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if parts[0] != "max":
            return math.ceil(int(parts[0]) / int(parts[1]))
    except (FileNotFoundError, ValueError):
        pass
    try:
        q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        p = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if q > 0:
            return math.ceil(q / p)
    except (FileNotFoundError, ValueError):
        pass
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    return os.cpu_count() or 1


def _container_ram_gb() -> float | None:
    """Read RAM limit from cgroup (container/pod)."""
    for p in ["/sys/fs/cgroup/memory.max",
              "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None


NUM_CPUS = _detect_cpus()
TOTAL_RAM_GB = _container_ram_gb() or 29.0

# Set virtual-memory ceiling — 70 % of container RAM, x3 for virtual overcommit
RAM_BUDGET_BYTES = int(TOTAL_RAM_GB * 0.70 * 1e9)
resource.setrlimit(resource.RLIMIT_AS,
                   (RAM_BUDGET_BYTES * 3, RAM_BUDGET_BYTES * 3))

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, "
            f"budget {RAM_BUDGET_BYTES / 1e9:.1f} GB")

# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════
LENGTHS = [10, 15, 20, 25, 30, 40]
PHIS = np.round(np.arange(-0.40, 0.45, 0.05), 2)  # 17 values
PHIS[PHIS == 0] = 0.0                              # avoid -0.0
N_REPS = 10_000
MLE_SUBSAMPLE = 200
CELLS = list(itertools.product(LENGTHS, [float(p) + 0.0 for p in PHIS]))

# Task B
TREE_SIZES = [20, 30, 40]
N_TREES_SYNTHETIC = 500
N_RPL_PERMS = 100
MAX_SENTENCES_REAL = 2000

# Bias-cancellation test
BIAS_CANCEL_CELLS = [
    (20, -0.20), (20, -0.10), (20, 0.0),
    (20,  0.10), (30, -0.20), (30, 0.0),
]
N_PAIRS = 1000
N_BASELINE_SAMPLES = 50

sys.setrecursionlimit(5000)

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def make_seed(*args) -> int:
    """Deterministic seed from arbitrary arguments."""
    h = hashlib.sha256(str(args).encode()).hexdigest()
    return int(h[:8], 16) % (2**31)


def to_native(obj):
    """Recursively convert numpy types to Python builtins for JSON."""
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# ═══════════════════════════════════════════════════════════════════════════════
# AUTOCORRELATION ESTIMATORS
# ═══════════════════════════════════════════════════════════════════════════════

def r1_standard_batch(X: np.ndarray) -> np.ndarray:
    """Vectorised standard (Yule-Walker) lag-1 autocorrelation per row."""
    Xc = X - X.mean(axis=1, keepdims=True)
    num = np.sum(Xc[:, :-1] * Xc[:, 1:], axis=1)
    den = np.sum(Xc ** 2, axis=1)
    den = np.where(den == 0, 1e-30, den)
    return num / den


def r1_corrected_batch(X: np.ndarray) -> np.ndarray:
    """Marriott-Pope analytical correction: r1 + (1 + 3*r1) / n."""
    r1 = r1_standard_batch(X)
    n = X.shape[1]
    return r1 + (1.0 + 3.0 * r1) / n


def compute_r1(seq) -> float:
    """Standard lag-1 autocorrelation of a 1-D numeric sequence."""
    x = np.asarray(seq, dtype=float)
    if len(x) < 4:
        return np.nan
    xc = x - x.mean()
    den = np.dot(xc, xc)
    if den == 0:
        return 0.0
    return float(np.dot(xc[:-1], xc[1:]) / den)


def estimator_mle_scipy(x: np.ndarray) -> float:
    """Profile-MLE for AR(1) phi via scipy bounded minimisation."""
    n = len(x)
    if n < 4:
        return np.nan
    y = x - np.mean(x)
    y0sq = y[0] ** 2
    y_lag = y[:-1]
    y_lead = y[1:]

    def neg_prof_ll(phi):
        if abs(phi) >= 0.999:
            return 1e10
        resid = y_lead - phi * y_lag
        ssr = (1 - phi ** 2) * y0sq + np.dot(resid, resid)
        if ssr <= 0:
            return 1e10
        return n * np.log(ssr / n) - np.log(1 - phi ** 2)

    try:
        res = minimize_scalar(neg_prof_ll, bounds=(-0.99, 0.99),
                              method="bounded")
        return float(res.x) if res.success else np.nan
    except Exception:
        return np.nan


# ═══════════════════════════════════════════════════════════════════════════════
# AR(1) SEQUENCE GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_ar1_batch(n: int, phi: float, n_reps: int,
                       rng: np.random.Generator) -> np.ndarray:
    """Generate *n_reps* AR(1) sequences of length *n* (rows of a 2-D array).

    The initial observation is drawn from the stationary distribution
    N(0, 1/(1-phi^2)) so there is no burn-in transient.
    """
    eps = rng.normal(0.0, 1.0, size=(n_reps, n))
    x = np.empty_like(eps)
    if abs(phi) < 0.999:
        x[:, 0] = eps[:, 0] / np.sqrt(1 - phi ** 2)
    else:
        x[:, 0] = eps[:, 0]
    for t in range(1, n):
        x[:, t] = phi * x[:, t - 1] + eps[:, t]
    return x


# ═══════════════════════════════════════════════════════════════════════════════
# TASK A — MONTE CARLO CELL WORKER
# ═══════════════════════════════════════════════════════════════════════════════

def process_cell(args):
    """Compute bias / RMSE / coverage / power for one (n, phi) cell."""
    n, phi, seed, n_reps, mle_sub = args
    rng = np.random.default_rng(seed)
    X = generate_ar1_batch(n, phi, n_reps, rng)

    # Vectorised estimators
    r1_vals = r1_standard_batch(X)
    r1c_vals = r1_corrected_batch(X)

    # MLE on a subsample (scipy profile-MLE, much faster than statsmodels)
    mle_sub_actual = min(mle_sub, n_reps)
    mle_vals = np.array([estimator_mle_scipy(X[i])
                         for i in range(mle_sub_actual)])
    mle_vals = mle_vals[~np.isnan(mle_vals)]

    se_null = 1.0 / np.sqrt(n)
    results = {}

    for name, vals in [("r1_standard", r1_vals),
                       ("r1_corrected", r1c_vals),
                       ("mle", mle_vals)]:
        bias  = float(np.mean(vals) - phi)
        rmse  = float(np.sqrt(np.mean((vals - phi) ** 2)))
        se_est = float(np.std(vals))
        ci_lo = vals - 1.96 * se_est
        ci_hi = vals + 1.96 * se_est
        coverage = float(np.mean((ci_lo <= phi) & (phi <= ci_hi)))
        z = np.abs(vals) / se_null
        power = float(np.mean(z > 1.96))

        results[name] = {
            "bias":          bias,
            "rmse":          rmse,
            "coverage_95":   coverage,
            "power_alpha05": power,
            "mean_estimate": float(np.mean(vals)),
            "std_estimate":  se_est,
            "n_valid":       int(len(vals)),
        }

    return (n, float(phi), results)


# ═══════════════════════════════════════════════════════════════════════════════
# TASK B — TREE UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def build_children_map(head_array):
    """Return (children dict, root index).

    *head_array* uses **1-indexed** head positions; 0 means root.
    """
    children: dict[int, list[int]] = defaultdict(list)
    root = None
    for i, h in enumerate(head_array):
        if h == 0:
            root = i
        else:
            children[h - 1].append(i)
    return children, root


def rpl_linearize(children, node, rng):
    """Random Projective Linearisation — Gildea-Temperley method.

    At each node, randomly assign each dependent to LEFT or RIGHT of head,
    randomly order within each side, then recurse.  Produces a projective
    linearisation (all subtrees are contiguous).
    """
    kids = children.get(node, [])
    if not kids:
        return [node]
    left, right = [], []
    for k in kids:
        (left if rng.random() < 0.5 else right).append(k)
    rng.shuffle(left)
    rng.shuffle(right)
    out: list[int] = []
    for k in left:
        out.extend(rpl_linearize(children, k, rng))
    out.append(node)
    for k in right:
        out.extend(rpl_linearize(children, k, rng))
    return out


def compute_dd_from_linearization(linearization, head_array):
    """DD sequence from a linearisation (excludes root token)."""
    pos_map = {orig: new + 1
               for new, orig in enumerate(linearization)}
    dd: list[int] = []
    for new_idx, orig in enumerate(linearization):
        h = head_array[orig]
        if h == 0:
            continue                       # root — skip
        head_orig = h - 1
        if head_orig in pos_map:
            dd.append(abs((new_idx + 1) - pos_map[head_orig]))
    return dd


def compute_dd_filtered(linearization, head_array, deprel_array):
    """DD sequence excluding root **and** punct tokens."""
    pos_map = {orig: new + 1
               for new, orig in enumerate(linearization)}
    dd: list[int] = []
    for new_idx, orig in enumerate(linearization):
        h = head_array[orig]
        if h == 0:
            continue
        if deprel_array[orig] == "punct":
            continue
        head_orig = h - 1
        if head_orig in pos_map:
            dd.append(abs((new_idx + 1) - pos_map[head_orig]))
    return dd


# ── Synthetic tree generators ───────────────────────────────────────────────

def make_star_tree(n: int) -> list[int]:
    """Star: root at 0, every other token depends on root."""
    head = [0] * n
    for i in range(1, n):
        head[i] = 1          # 1-indexed → root is position 0
    return head


def make_caterpillar_tree(n: int) -> list[int]:
    """Caterpillar: spine 0→1→…→floor(n/2)-1, remaining are leaves."""
    head = [0] * n
    spine = n // 2
    for i in range(1, spine):
        head[i] = i           # each spine node → previous (1-indexed)
    leaf = spine
    for i in range(spine):
        if leaf < n:
            head[leaf] = i + 1
            leaf += 1
    return head


def make_balanced_tree(n: int) -> list[int]:
    """Complete binary tree (node i has children 2i+1, 2i+2)."""
    head = [0] * n
    for i in range(1, n):
        head[i] = (i - 1) // 2 + 1
    return head


def make_random_projective_tree(n: int, rng) -> list[int]:
    """Random projective tree via recursive partitioning."""
    head = [0] * n

    def _build(positions, parent_pos):
        if not positions:
            return
        ri = int(rng.integers(0, len(positions)))
        rp = positions[ri]
        if parent_pos is not None:
            head[rp] = parent_pos + 1       # 1-indexed head
        left  = [p for p in positions if p < rp]
        right = [p for p in positions if p > rp]
        _build(left,  rp)
        _build(right, rp)

    _build(list(range(n)), None)
    return head


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING (for Task B real-tree analysis)
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_sentences(data: dict, min_tokens: int) -> list[dict]:
    """Pull qualifying sentences from one data-file dict."""
    sents: list[dict] = []
    for ds in data.get("datasets", []):
        for ex in ds.get("examples", []):
            try:
                inp = json.loads(ex["input"])
                out = json.loads(ex["output"])
                head  = inp["head_array"]
                deprl = inp["deprel_array"]
                keep  = [i for i, d in enumerate(deprl) if d != "punct"]
                if len(keep) < min_tokens:
                    continue
                sents.append({
                    "head_array":   head,
                    "deprel_array": deprl,
                    "keep_idx":     keep,
                    "token_count":  out["token_count"],
                    "treebank_id":  ex.get("metadata_treebank_id", "unknown"),
                })
            except (json.JSONDecodeError, KeyError):
                continue
    return sents


def load_sentences(data_dir: Path, max_sentences: int,
                   min_tokens: int = 15) -> list[dict]:
    """Load up to *max_sentences* qualifying sentences.

    Tries mini_data first, then progressively loads full_data parts
    until the target count is reached.
    """
    sents: list[dict] = []

    # 1) mini data
    mini = data_dir / "mini_data_out.json"
    if mini.exists():
        logger.info(f"Loading {mini.name}")
        d = json.loads(mini.read_text())
        sents.extend(_extract_sentences(d, min_tokens))
        logger.info(f"  {len(sents)} qualifying sentences from mini")
        del d; gc.collect()

    # 2) full data parts (one-by-one to conserve memory)
    if len(sents) < max_sentences:
        for part in range(1, 17):
            if len(sents) >= max_sentences:
                break
            fp = data_dir / f"data_out/full_data_out_{part}.json"
            if not fp.exists():
                logger.warning(f"  {fp.name} not found — skipping")
                continue
            logger.info(f"  Loading {fp.name} "
                        f"({len(sents)}/{max_sentences} so far)")
            try:
                d = json.loads(fp.read_text())
                new = _extract_sentences(d, min_tokens)
                sents.extend(new)
                logger.info(f"    +{len(new)} → {len(sents)} total")
                del d; gc.collect()
            except Exception as exc:
                logger.error(f"    Failed: {exc}")

    # 3) sub-sample if we collected more than needed
    if len(sents) > max_sentences:
        rng = np.random.default_rng(123)
        idx = rng.choice(len(sents), size=max_sentences, replace=False)
        sents = [sents[i] for i in sorted(idx)]

    logger.info(f"Final sentence count: {len(sents)}")
    return sents


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

@logger.catch
def main() -> None:
    t0 = time.time()

    # ──────────────────────────────────────────────────────────────────────
    # TASK A — MONTE CARLO ESTIMATOR VALIDATION
    # ──────────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("TASK A: Monte Carlo Estimator Validation")
    logger.info(f"  {len(LENGTHS)} lengths x {len(PHIS)} phi = "
                f"{len(CELLS)} cells, {N_REPS:,} reps/cell, "
                f"MLE sub-sample {MLE_SUBSAMPLE}")

    base_rng   = np.random.default_rng(42)
    cell_seeds = base_rng.integers(0, 2**31, size=len(CELLS)).tolist()
    cell_args  = [(n, float(phi), seed, N_REPS, MLE_SUBSAMPLE)
                  for (n, phi), seed in zip(CELLS, cell_seeds)]

    mc_results: dict[str, dict] = {}
    n_workers = min(NUM_CPUS, 4)
    ta_start  = time.time()
    done      = 0

    logger.info(f"  Launching {n_workers} workers …")
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(process_cell, a): a for a in cell_args}
        for fut in as_completed(futs):
            try:
                n_val, phi_val, res = fut.result()
                mc_results[f"n{n_val}_phi{phi_val:.2f}"] = res
                done += 1
                if done % 20 == 0 or done == len(CELLS):
                    logger.info(f"    {done}/{len(CELLS)} cells "
                                f"({time.time()-ta_start:.0f}s)")
            except Exception as exc:
                logger.error(f"    Cell {futs[fut][:2]} failed: {exc}")

    ta_time = time.time() - ta_start
    logger.info(f"  Task A done in {ta_time:.1f}s "
                f"({len(mc_results)} cells)")

    # ── MDE table ────────────────────────────────────────────────────────
    mde_table: dict[str, float | None] = {}
    for n_val in LENGTHS:
        for est in ("r1_standard", "r1_corrected", "mle"):
            mde = None
            for pv in sorted(float(p) for p in PHIS if p > 0):
                key = f"n{n_val}_phi{pv:.2f}"
                if (key in mc_results
                        and mc_results[key][est]["power_alpha05"] >= 0.80):
                    mde = pv
                    break
            mde_table[f"n{n_val}_{est}"] = mde
    logger.info(f"  MDE table:\n{json.dumps(to_native(mde_table), indent=2)}")

    # ── Bias-cancellation test ───────────────────────────────────────────
    logger.info("  Bias-cancellation test …")
    bias_cancel: dict[str, dict] = {}
    for n_val, phi_val in BIAS_CANCEL_CELLS:
        rng = np.random.default_rng(make_seed("bias_cancel", n_val, phi_val))
        X_A  = generate_ar1_batch(n_val, phi_val, N_PAIRS, rng)
        r1_A = r1_standard_batch(X_A)
        exc  = np.empty(N_PAIRS)
        for i in range(N_PAIRS):
            X_B = generate_ar1_batch(n_val, phi_val, N_BASELINE_SAMPLES, rng)
            exc[i] = r1_A[i] - float(np.mean(r1_standard_batch(X_B)))
        k = f"n{n_val}_phi{phi_val:.2f}"
        lo, hi = float(np.percentile(exc, 2.5)), float(np.percentile(exc, 97.5))
        bias_cancel[k] = {
            "mean_excess":        float(np.mean(exc)),
            "std_excess":         float(np.std(exc)),
            "ci_95_low":          lo,
            "ci_95_high":         hi,
            "proportion_negative": float(np.mean(exc < 0)),
            "zero_in_ci":         bool(lo <= 0 <= hi),
        }
        logger.info(f"    {k}: mean={bias_cancel[k]['mean_excess']:.6f}  "
                    f"zero_in_ci={bias_cancel[k]['zero_in_ci']}")

    # ── Task A plots ─────────────────────────────────────────────────────
    logger.info("  Generating Task A plots …")

    # Power curves
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Power Curves by Sequence Length", fontsize=14)
    for idx, n_val in enumerate(LENGTHS):
        ax = axes[idx // 3, idx % 3]
        for est, clr in [("r1_standard", "#2196F3"),
                         ("r1_corrected", "#FF9800"),
                         ("mle",          "#4CAF50")]:
            pw = [mc_results.get(f"n{n_val}_phi{p:.2f}", {})
                            .get(est, {})
                            .get("power_alpha05", np.nan)
                  for p in PHIS]
            ax.plot(PHIS, pw, color=clr, label=est, marker=".", ms=4, lw=1.5)
        ax.axhline(0.80, color="red",  ls="--", alpha=.5, label="80 % power")
        ax.axhline(0.05, color="gray", ls=":",  alpha=.5, label="alpha=0.05")
        ax.set_title(f"n = {n_val}", fontsize=11)
        ax.set_xlabel("True phi"); ax.set_ylabel("Power")
        ax.legend(fontsize=7); ax.set_ylim(-0.05, 1.05); ax.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(str(WORKSPACE / "power_curves.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # Bias curves
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Bias Curves by Sequence Length", fontsize=14)
    for idx, n_val in enumerate(LENGTHS):
        ax = axes[idx // 3, idx % 3]
        for est, clr in [("r1_standard", "#2196F3"),
                         ("r1_corrected", "#FF9800"),
                         ("mle",          "#4CAF50")]:
            bi = [mc_results.get(f"n{n_val}_phi{p:.2f}", {})
                            .get(est, {})
                            .get("bias", np.nan)
                  for p in PHIS]
            ax.plot(PHIS, bi, color=clr, label=est, marker=".", ms=4, lw=1.5)
        # Theoretical first-order bias for standard r1
        theo = [-(1 + 3 * float(p)) / n_val for p in PHIS]
        ax.plot(PHIS, theo, color="#2196F3", ls=":", alpha=.5,
                label="theoretical r1 bias")
        ax.axhline(0, color="black", ls="-", alpha=.3)
        ax.set_title(f"Bias at n = {n_val}", fontsize=11)
        ax.set_xlabel("True phi"); ax.set_ylabel("Bias  (E[est] - phi)")
        ax.legend(fontsize=7); ax.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(str(WORKSPACE / "bias_curves.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Task A plots saved")

    # ──────────────────────────────────────────────────────────────────────
    # TASK B — GEOMETRIC CONFOUND ANALYSIS
    # ──────────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("TASK B: Geometric Confound Analysis")

    # ── Prong 1: synthetic canonical trees ───────────────────────────────
    logger.info("  Prong 1 — synthetic trees …")
    synthetic_results: dict[str, dict] = {}
    tree_gens: list[tuple] = [
        ("star",              make_star_tree,              False),
        ("caterpillar",       make_caterpillar_tree,       False),
        ("balanced",          make_balanced_tree,          False),
        ("random_projective", make_random_projective_tree, True),
    ]

    for ttype, gen_fn, needs_rng in tree_gens:
        for sz in TREE_SIZES:
            excesses: list[float] = []
            obs_r1s:  list[float] = []
            rpl_means: list[float] = []

            for t_idx in range(N_TREES_SYNTHETIC):
                rng = np.random.default_rng(make_seed(ttype, sz, t_idx))
                harr = gen_fn(sz, rng) if needs_rng else gen_fn(sz)

                children, root = build_children_map(harr)
                if root is None:
                    continue

                # Canonical (natural position) linearisation
                dd_can = compute_dd_from_linearization(list(range(sz)), harr)
                r1_can = compute_r1(dd_can)

                # RPL permutations
                rpl_r1_list: list[float] = []
                for _ in range(N_RPL_PERMS):
                    try:
                        lin = rpl_linearize(children, root, rng)
                        r1v = compute_r1(
                            compute_dd_from_linearization(lin, harr))
                        if not np.isnan(r1v):
                            rpl_r1_list.append(r1v)
                    except RecursionError:
                        continue

                if rpl_r1_list and not np.isnan(r1_can):
                    ex = r1_can - float(np.mean(rpl_r1_list))
                    excesses.append(ex)
                    obs_r1s.append(r1_can)
                    rpl_means.append(float(np.mean(rpl_r1_list)))

            key = f"{ttype}_n{sz}"
            if excesses:
                ea = np.array(excesses)
                synthetic_results[key] = {
                    "mean_excess":        float(np.mean(ea)),
                    "std_excess":         float(np.std(ea)),
                    "median_excess":      float(np.median(ea)),
                    "mean_obs_r1":        float(np.mean(obs_r1s)),
                    "mean_rpl_r1":        float(np.mean(rpl_means)),
                    "n_trees_computed":   len(excesses),
                    "proportion_negative": float(np.mean(ea < 0)),
                }
            else:
                synthetic_results[key] = {
                    "mean_excess": 0.0, "std_excess": 0.0,
                    "median_excess": 0.0, "mean_obs_r1": 0.0,
                    "mean_rpl_r1": 0.0, "n_trees_computed": 0,
                    "proportion_negative": 0.0,
                }
            logger.info(f"    {key}: "
                        f"mean_excess={synthetic_results[key]['mean_excess']:.4f}  "
                        f"n={synthetic_results[key]['n_trees_computed']}")

    # ── Prong 2 & 3: real UD dependency trees ────────────────────────────
    logger.info("  Prong 2-3 — real UD trees …")
    all_sents = load_sentences(DATA_DEP_PATH, MAX_SENTENCES_REAL)

    rpl_vs_rpl_exc:  list[float] = []
    obs_vs_rpl_exc:  list[float] = []
    shuf_vs_rpl_exc: list[float] = []

    tb_start = time.time()
    for s_idx, sent in enumerate(all_sents):
        harr  = sent["head_array"]
        drels = sent["deprel_array"]
        children, root = build_children_map(harr)
        if root is None:
            continue

        # Observed DD (punct-filtered, original word order)
        keep = sent["keep_idx"]
        obs_dd = [abs(i + 1 - harr[i]) for i in keep if harr[i] != 0]
        r1_obs = compute_r1(obs_dd)

        # Generate 100 RPL permutations
        rng_r = np.random.default_rng(make_seed("rpl", s_idx))
        rpl_r1s: list[float] = []
        for _ in range(100):
            try:
                lin = rpl_linearize(children, root, rng_r)
                dd_f = compute_dd_filtered(lin, harr, drels)
                val = compute_r1(dd_f)
                if not np.isnan(val):
                    rpl_r1s.append(val)
            except RecursionError:
                continue

        if len(rpl_r1s) < 50:
            continue

        # Prong 2 — RPL-vs-RPL geometric null
        h1 = float(np.mean(rpl_r1s[:50]))
        h2 = float(np.mean(rpl_r1s[50:]))
        rpl_vs_rpl_exc.append(h1 - h2)

        # Prong 3 — observed vs RPL, shuffled vs RPL
        if not np.isnan(r1_obs):
            rpl_m = float(np.mean(rpl_r1s))
            obs_vs_rpl_exc.append(r1_obs - rpl_m)

            shuf_dd = list(rng_r.permutation(obs_dd))
            r1_shuf = compute_r1(shuf_dd)
            if not np.isnan(r1_shuf):
                shuf_vs_rpl_exc.append(r1_shuf - rpl_m)

        if (s_idx + 1) % 200 == 0:
            logger.info(f"    {s_idx+1}/{len(all_sents)} sentences  "
                        f"({time.time()-tb_start:.0f}s)")

    logger.info(f"  Real-tree results: "
                f"RPL-vs-RPL={len(rpl_vs_rpl_exc)}, "
                f"obs-vs-RPL={len(obs_vs_rpl_exc)}, "
                f"shuf-vs-RPL={len(shuf_vs_rpl_exc)}")

    # ── Assemble geometric results ───────────────────────────────────────
    geo_results: dict[str, dict] = {}

    if len(rpl_vs_rpl_exc) > 1:
        geo_results["rpl_vs_rpl"] = {
            "mean_excess":  float(np.mean(rpl_vs_rpl_exc)),
            "std_excess":   float(np.std(rpl_vs_rpl_exc)),
            "p_value_ttest": float(
                scipy.stats.ttest_1samp(rpl_vs_rpl_exc, 0).pvalue),
            "n_sentences":  len(rpl_vs_rpl_exc),
            "near_zero":    bool(abs(np.mean(rpl_vs_rpl_exc)) < 0.01),
        }

    if len(obs_vs_rpl_exc) > 1:
        geo_results["observed_vs_rpl"] = {
            "mean_excess":        float(np.mean(obs_vs_rpl_exc)),
            "std_excess":         float(np.std(obs_vs_rpl_exc)),
            "p_value_ttest":      float(
                scipy.stats.ttest_1samp(obs_vs_rpl_exc, 0).pvalue),
            "n_sentences":        len(obs_vs_rpl_exc),
            "median_excess":      float(np.median(obs_vs_rpl_exc)),
            "proportion_negative": float(
                np.mean(np.array(obs_vs_rpl_exc) < 0)),
        }

    if len(shuf_vs_rpl_exc) > 1:
        geo_results["shuffled_vs_rpl"] = {
            "mean_excess":  float(np.mean(shuf_vs_rpl_exc)),
            "std_excess":   float(np.std(shuf_vs_rpl_exc)),
            "p_value_ttest": float(
                scipy.stats.ttest_1samp(shuf_vs_rpl_exc, 0).pvalue),
            "n_sentences":  len(shuf_vs_rpl_exc),
        }

    if len(obs_vs_rpl_exc) > 1 and len(shuf_vs_rpl_exc) > 1:
        geo_results["observed_vs_shuffled_comparison"] = {
            "mean_diff":    float(np.mean(obs_vs_rpl_exc)
                                  - np.mean(shuf_vs_rpl_exc)),
            "ttest_pvalue": float(
                scipy.stats.ttest_ind(
                    obs_vs_rpl_exc, shuf_vs_rpl_exc).pvalue),
        }

    # ── Task B plots ─────────────────────────────────────────────────────
    logger.info("  Generating Task B plots …")

    # Synthetic-tree excess bar chart
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Synthetic Tree Excess r1 (Canonical vs RPL)", fontsize=14)
    colours = ["#2196F3", "#FF9800", "#4CAF50"]
    for idx, ttype in enumerate(["star", "caterpillar",
                                  "balanced", "random_projective"]):
        ax = axes[idx]
        means = [synthetic_results.get(f"{ttype}_n{s}", {})
                 .get("mean_excess", 0) for s in TREE_SIZES]
        stds  = [synthetic_results.get(f"{ttype}_n{s}", {})
                 .get("std_excess",  0) for s in TREE_SIZES]
        ax.bar([str(s) for s in TREE_SIZES], means, yerr=stds,
               capsize=5, alpha=.7, color=colours)
        ax.axhline(0, color="red", ls="--", lw=1)
        ax.set_title(ttype, fontsize=11)
        ax.set_ylabel("Excess r1"); ax.set_xlabel("n")
        ax.grid(alpha=.3, axis="y")
    plt.tight_layout()
    plt.savefig(str(WORKSPACE / "geometric_synthetic.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # Real-tree excess histograms
    fig, ax = plt.subplots(figsize=(10, 6))
    nbins = lambda lst: min(50, max(10, len(lst) // 5)) if lst else 10
    if rpl_vs_rpl_exc:
        ax.hist(rpl_vs_rpl_exc,  bins=nbins(rpl_vs_rpl_exc),  alpha=.5,
                density=True, color="#2196F3",
                label=f"RPL-vs-RPL (n={len(rpl_vs_rpl_exc)})")
    if obs_vs_rpl_exc:
        ax.hist(obs_vs_rpl_exc,  bins=nbins(obs_vs_rpl_exc),  alpha=.5,
                density=True, color="#F44336",
                label=f"Observed-vs-RPL (n={len(obs_vs_rpl_exc)})")
    if shuf_vs_rpl_exc:
        ax.hist(shuf_vs_rpl_exc, bins=nbins(shuf_vs_rpl_exc), alpha=.5,
                density=True, color="#4CAF50",
                label=f"Shuffled-vs-RPL (n={len(shuf_vs_rpl_exc)})")
    ax.axvline(0, color="black", ls="--", lw=1)
    ax.legend(fontsize=10)
    ax.set_xlabel("Excess lag-1 autocorrelation")
    ax.set_ylabel("Density")
    ax.set_title("Geometric Confound Check: Real UD Sentences")
    ax.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(str(WORKSPACE / "geometric_real_trees.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Task B plots saved")

    # ──────────────────────────────────────────────────────────────────────
    # ASSEMBLE OUTPUT  (exp_gen_sol_out schema)
    # ──────────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Assembling method_out.json …")

    # Interpretations
    bias_cancels = all(v["zero_in_ci"] for v in bias_cancel.values())
    chosen_est   = "r1_standard" if bias_cancels else "r1_corrected"
    rationale    = ("Bias cancels in excess measure (real minus RPL); "
                    "r1_standard preferred for lowest variance."
                    if bias_cancels else
                    "Bias does not fully cancel; r1_corrected provides "
                    "analytical Marriott-Pope correction.")

    geo_absorbed = all(
        abs(synthetic_results.get(f"{tt}_n{s}", {})
            .get("mean_excess", 1.0)) < 0.05
        for tt in ("star", "caterpillar", "balanced", "random_projective")
        for s in TREE_SIZES
    )
    rpl_null_ok = (geo_results.get("rpl_vs_rpl", {})
                              .get("p_value_ttest", 0) > 0.05)
    obs_p = geo_results.get("observed_vs_rpl", {}).get("p_value_ttest", 1)
    obs_m = geo_results.get("observed_vs_rpl", {}).get("mean_excess", 0)
    shuf_p = geo_results.get("shuffled_vs_rpl", {}).get("p_value_ttest", 1)
    shuf_m = geo_results.get("shuffled_vs_rpl", {}).get("mean_excess", 0)
    # Sequential effect is genuine if observed excess is significantly
    # more negative than shuffled excess (the key comparison).
    comp_p = (geo_results.get("observed_vs_shuffled_comparison", {})
                         .get("ttest_pvalue", 1.0))
    comp_d = (geo_results.get("observed_vs_shuffled_comparison", {})
                         .get("mean_diff", 0.0))
    seq_genuine = (obs_p < 0.05 and obs_m < 0
                   and comp_p < 0.05 and comp_d < 0)

    output: dict = {
        "metadata": {
            "method_name": ("Phase 1: Monte Carlo Estimator Validation "
                            "& Geometric Confound Analysis"),
            "description": (
                "Task A: MC simulation of AR(1) sequences comparing "
                "3 autocorrelation estimators across 102 parameter cells. "
                "Task B: Three-pronged geometric confound analysis on "
                "synthetic and real dependency trees."
            ),
            "task_a_monte_carlo": {
                "simulation_params": {
                    "lengths":     LENGTHS,
                    "phi_values":  [float(p) for p in PHIS],
                    "n_reps_r1":   N_REPS,
                    "n_reps_mle":  MLE_SUBSAMPLE,
                    "n_cells":     len(CELLS),
                },
                "per_cell_results":  to_native(mc_results),
                "mde_table":         to_native(mde_table),
                "bias_cancellation": to_native(bias_cancel),
                "estimator_recommendation": {
                    "chosen_estimator":     chosen_est,
                    "rationale":            rationale,
                    "bias_cancels_in_excess": bias_cancels,
                },
                "runtime_seconds": round(ta_time, 2),
            },
            "task_b_geometric_confounds": {
                "prong1_synthetic":   to_native(synthetic_results),
                "prong2_3_real":      to_native(geo_results),
                "interpretation": {
                    "geometry_absorbed_by_rpl":  geo_absorbed,
                    "rpl_null_valid":            rpl_null_ok,
                    "sequential_effect_genuine": seq_genuine,
                },
            },
            "figures": [
                "power_curves.png", "bias_curves.png",
                "geometric_synthetic.png", "geometric_real_trees.png",
            ],
            "total_runtime_seconds": round(time.time() - t0, 2),
        },
        "datasets": [
            {   # ──── Dataset 1: Task A ────
                "dataset": "monte_carlo_estimator_validation",
                "examples": [],
            },
            {   # ──── Dataset 2: Task B ────
                "dataset": "geometric_confound_analysis",
                "examples": [],
            },
        ],
    }

    # ── Populate Task A examples ─────────────────────────────────────────
    for key in sorted(mc_results):
        res   = mc_results[key]
        parts = key.split("_phi")
        nv    = int(parts[0].replace("n", ""))
        pv    = float(parts[1])
        # Build per-estimator predictions
        pred_std = json.dumps(to_native({
            "estimator": "r1_standard",
            "bias": res["r1_standard"]["bias"],
            "rmse": res["r1_standard"]["rmse"],
            "power": res["r1_standard"]["power_alpha05"],
        }))
        pred_cor = json.dumps(to_native({
            "estimator": "r1_corrected",
            "bias": res["r1_corrected"]["bias"],
            "rmse": res["r1_corrected"]["rmse"],
            "power": res["r1_corrected"]["power_alpha05"],
        }))
        pred_mle = json.dumps(to_native({
            "estimator": "mle",
            "bias": res["mle"]["bias"],
            "rmse": res["mle"]["rmse"],
            "power": res["mle"]["power_alpha05"],
        }))
        output["datasets"][0]["examples"].append({
            "input":  json.dumps({"task": "mc_cell", "n": nv,
                                  "phi": pv, "n_reps": N_REPS,
                                  "mle_sub": MLE_SUBSAMPLE}),
            "output": json.dumps(to_native(res)),
            "predict_r1_standard":  pred_std,
            "predict_r1_corrected": pred_cor,
            "predict_mle":          pred_mle,
            "metadata_cell_key":   key,
            "metadata_seq_length": nv,
            "metadata_true_phi":   pv,
        })

    # Summary rows
    mde_pred = json.dumps(to_native(mde_table))
    output["datasets"][0]["examples"].append({
        "input":  json.dumps({"task": "mde_table"}),
        "output": json.dumps(to_native(mde_table)),
        "predict_r1_standard":  mde_pred,
        "predict_r1_corrected": mde_pred,
        "predict_mle":          mde_pred,
        "metadata_cell_key": "mde_summary",
    })
    bc_pred = json.dumps(to_native(bias_cancel))
    output["datasets"][0]["examples"].append({
        "input":  json.dumps({"task": "bias_cancellation"}),
        "output": json.dumps(to_native(bias_cancel)),
        "predict_r1_standard":  bc_pred,
        "predict_r1_corrected": bc_pred,
        "predict_mle":          bc_pred,
        "metadata_cell_key": "bias_cancellation",
    })
    rec_pred = json.dumps({"chosen": chosen_est,
                            "rationale": rationale,
                            "bias_cancels": bias_cancels})
    output["datasets"][0]["examples"].append({
        "input":  json.dumps({"task": "estimator_recommendation"}),
        "output": rec_pred,
        "predict_r1_standard":  rec_pred,
        "predict_r1_corrected": rec_pred,
        "predict_mle":          rec_pred,
        "metadata_cell_key": "recommendation",
    })

    # ── Populate Task B examples ─────────────────────────────────────────
    for key in sorted(synthetic_results):
        res   = synthetic_results[key]
        parts = key.rsplit("_n", 1)
        tt    = parts[0]
        sz    = int(parts[1])
        synth_pred = json.dumps(to_native({
            "mean_excess": res.get("mean_excess", 0.0),
            "proportion_negative": res.get("proportion_negative", 0.0),
            "absorbed": abs(res.get("mean_excess", 1.0)) < 0.05,
        }))
        output["datasets"][1]["examples"].append({
            "input":  json.dumps({"task": "synthetic_excess",
                                  "tree_type": tt, "n": sz,
                                  "n_trees": N_TREES_SYNTHETIC,
                                  "n_rpl": N_RPL_PERMS}),
            "output": json.dumps(to_native(res)),
            "predict_rpl_baseline": synth_pred,
            "metadata_cell_key":  key,
            "metadata_tree_type": tt,
            "metadata_tree_size": sz,
        })

    geo_pred = json.dumps(to_native(geo_results))
    output["datasets"][1]["examples"].append({
        "input":  json.dumps({"task": "geometric_null_real_trees",
                              "n_sentences": len(all_sents),
                              "n_rpl": 100}),
        "output": json.dumps(to_native(geo_results)),
        "predict_rpl_baseline": geo_pred,
        "metadata_cell_key": "real_tree_null",
    })
    interp_pred = json.dumps({
        "geometry_absorbed":  geo_absorbed,
        "rpl_null_valid":     rpl_null_ok,
        "sequential_genuine": seq_genuine,
    })
    output["datasets"][1]["examples"].append({
        "input":  json.dumps({"task": "interpretation"}),
        "output": interp_pred,
        "predict_rpl_baseline": interp_pred,
        "metadata_cell_key": "interpretation",
    })

    # ── Save ─────────────────────────────────────────────────────────────
    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2))
    sz = out_path.stat().st_size
    logger.info(f"Saved {out_path.name}  ({sz / 1e6:.2f} MB)")
    logger.info(f"Total runtime: {time.time() - t0:.1f}s")

    if sz > 45_000_000:
        logger.warning("Output exceeds 45 MB — splitting may be needed")


if __name__ == "__main__":
    main()
