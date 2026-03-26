#!/usr/bin/env python3
"""
Robustness & Sensitivity: 4-Variant Punct/Threshold/Projectivity/Aggregation Analysis.

Runs four pipeline variants on ~50 typologically diverse treebanks using 100 RPL
permutations each. Produces scope quantification, within-language aggregation
sensitivity, and concrete compensatory-pattern examples.

Output: method_out.json conforming to exp_gen_sol_out schema.
"""

import gc
import json
import math
import os
import resource
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from loguru import logger
from scipy.stats import t as t_dist, wilcoxon

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ---------------------------------------------------------------------------
# Hardware detection (container-aware)
# ---------------------------------------------------------------------------
def _detect_cpus() -> int:
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
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None


NUM_CPUS = _detect_cpus()
TOTAL_RAM_GB = _container_ram_gb() or 16.0

# Set memory limit: use ~80% of container RAM
RAM_BUDGET_BYTES = int(TOTAL_RAM_GB * 0.80 * 1024**3)
try:
    resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET_BYTES * 3, RAM_BUDGET_BYTES * 3))
except (ValueError, resource.error):
    logger.warning("Could not set RLIMIT_AS")

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, budget={RAM_BUDGET_BYTES / 1e9:.1f} GB")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).resolve().parent
DEP_BASE = Path("/ai-inventor/aii_pipeline/runs/comp-ling-dobrovoljc_ebw/3_invention_loop/iter_1/gen_art")

DATA_ID3_DIR = DEP_BASE / "data_id3_it1__opus" / "data_out"
DATA_ID4_PATH = DEP_BASE / "data_id4_it1__opus" / "full_data_out.json"
DATA_ID5_PATH = DEP_BASE / "data_id5_it1__opus" / "full_data_out.json"

N_PERMS = 100
MIN_SENTENCES_META = 30  # Minimum sentences per treebank for meta-analysis
MIN_SENTENCES_EFFECT = 10  # Minimum sentences for any variant estimate
TARGET_TREEBANKS = 50
sys.setrecursionlimit(5000)

# ---------------------------------------------------------------------------
# Core computation functions
# ---------------------------------------------------------------------------

def compute_dd_punct_excluded(head_array: list[int], deprel_array: list[str]) -> list[int]:
    """Compute DD sequence excluding punctuation tokens."""
    dd_seq = []
    for i in range(len(head_array)):
        if deprel_array[i] == "punct":
            continue
        h = head_array[i]
        if h == 0:
            continue
        dd_seq.append(abs((i + 1) - h))
    return dd_seq


def compute_dd_punct_included(head_array: list[int]) -> list[int]:
    """Original DD sequence including all non-root tokens."""
    dd_seq = []
    for i in range(len(head_array)):
        h = head_array[i]
        if h == 0:
            continue
        dd_seq.append(abs((i + 1) - h))
    return dd_seq


def check_projectivity(head_array: list[int]) -> bool:
    """Check if dependency tree has no crossing arcs."""
    n = len(head_array)
    for i in range(n):
        h = head_array[i]
        if h == 0:
            continue
        pos = i + 1
        lo, hi = min(pos, h), max(pos, h)
        for j_pos in range(lo + 1, hi):
            hj = head_array[j_pos - 1]
            if hj < lo or hj > hi:
                return False
    return True


def build_children_map(head_array: list[int]):
    """Build parent->children mapping. Returns (root_0based, children_dict)."""
    children = defaultdict(list)
    root = None
    for i, h in enumerate(head_array):
        if h == 0:
            root = i
        else:
            children[h - 1].append(i)
    return root, children


def random_projective_linearize(node: int, children: dict, rng) -> list[int]:
    """Recursively produce a random projective linearization (iterative to avoid deep recursion)."""
    # Use iterative approach with explicit stack
    result = []
    # We need a stack-based approach: for each node, we shuffle subtrees + head marker
    # and then process them in order.
    # Stack entries: either an int (emit this node) or a tuple (node_to_expand,)
    stack = [(node,)]

    while stack:
        item = stack.pop()
        if isinstance(item, int):
            result.append(item)
        else:
            nd = item[0]
            child_list = children.get(nd, [])
            if not child_list:
                result.append(nd)
            else:
                # Recursively linearize each child subtree - but we need subtrees first
                # For iterative approach, we pre-compute subtrees recursively but with limit
                subtrees = []
                for c in child_list:
                    st = _rpl_recursive(c, children, rng)
                    subtrees.append(st)
                items = list(subtrees) + [None]
                rng.shuffle(items)
                for it in items:
                    if it is None:
                        result.append(nd)
                    else:
                        result.extend(it)
    return result


def _rpl_recursive(node: int, children: dict, rng) -> list[int]:
    """Recursive RPL helper."""
    child_list = children.get(node, [])
    if not child_list:
        return [node]
    subtrees = [_rpl_recursive(c, children, rng) for c in child_list]
    items = list(subtrees) + [None]
    rng.shuffle(items)
    result = []
    for it in items:
        if it is None:
            result.append(node)
        else:
            result.extend(it)
    return result


def compute_dd_from_linearization(
    order: list[int],
    head_array: list[int],
    deprel_array: list[str],
    exclude_punct: bool,
) -> list[int]:
    """Compute DD sequence from a random linearization."""
    pos_map = {}
    for new_pos_0, orig_idx in enumerate(order):
        pos_map[orig_idx] = new_pos_0 + 1

    dd_seq = []
    for orig_idx in order:
        if exclude_punct and deprel_array[orig_idx] == "punct":
            continue
        h = head_array[orig_idx]
        if h == 0:
            continue
        head_orig_idx = h - 1
        dd = abs(pos_map[orig_idx] - pos_map[head_orig_idx])
        dd_seq.append(dd)
    return dd_seq


def lag1_autocorrelation(seq) -> float:
    """Standard Yule-Walker lag-1 autocorrelation (r1)."""
    if len(seq) < 4:
        return float("nan")
    x = np.array(seq, dtype=np.float64)
    x_bar = x.mean()
    denom = np.sum((x - x_bar) ** 2)
    if denom == 0:
        return 0.0
    numer = np.sum((x[:-1] - x_bar) * (x[1:] - x_bar))
    return float(numer / denom)


# ---------------------------------------------------------------------------
# Per-sentence processing for all 4 variants
# ---------------------------------------------------------------------------

def process_sentence_all_variants(
    head_array: list[int],
    deprel_array: list[str],
    dd_sequence_original: list[int],
    token_count: int,
    n_perms: int = 100,
    rng=None,
) -> dict:
    """Process one sentence for all 4 variants. Returns dict with keys A/B/C/D."""
    rng = rng or np.random.default_rng()
    results = {}

    is_ge20 = token_count >= 20
    is_ge15 = token_count >= 15
    is_projective = check_projectivity(head_array)

    # Compute observed DD sequences
    dd_punct_excl = compute_dd_punct_excluded(head_array, deprel_array)
    dd_punct_incl = dd_sequence_original

    r1_obs_excl = lag1_autocorrelation(dd_punct_excl) if len(dd_punct_excl) >= 4 else float("nan")
    r1_obs_incl = lag1_autocorrelation(dd_punct_incl) if len(dd_punct_incl) >= 4 else float("nan")

    # Build tree for RPL
    root, children = build_children_map(head_array)
    if root is None:
        return {}

    # Generate RPL permutations
    rpl_r1_excl = []
    rpl_r1_incl = []
    for _ in range(n_perms):
        try:
            order = _rpl_recursive(root, children, rng)
        except RecursionError:
            continue
        if len(order) != len(head_array):
            continue
        dd_rpl_excl = compute_dd_from_linearization(order, head_array, deprel_array, True)
        dd_rpl_incl = compute_dd_from_linearization(order, head_array, deprel_array, False)
        if len(dd_rpl_excl) >= 4:
            rpl_r1_excl.append(lag1_autocorrelation(dd_rpl_excl))
        if len(dd_rpl_incl) >= 4:
            rpl_r1_incl.append(lag1_autocorrelation(dd_rpl_incl))

    mean_rpl_excl = float(np.nanmean(rpl_r1_excl)) if rpl_r1_excl else float("nan")
    mean_rpl_incl = float(np.nanmean(rpl_r1_incl)) if rpl_r1_incl else float("nan")

    # Variant A: punct excluded, >=20 tokens
    if is_ge20 and not np.isnan(r1_obs_excl) and not np.isnan(mean_rpl_excl):
        results["A"] = {
            "observed_r1": r1_obs_excl,
            "baseline_r1": mean_rpl_excl,
            "excess": r1_obs_excl - mean_rpl_excl,
            "dd_length": len(dd_punct_excl),
        }

    # Variant B: punct included, >=20 tokens
    if is_ge20 and not np.isnan(r1_obs_incl) and not np.isnan(mean_rpl_incl):
        results["B"] = {
            "observed_r1": r1_obs_incl,
            "baseline_r1": mean_rpl_incl,
            "excess": r1_obs_incl - mean_rpl_incl,
            "dd_length": len(dd_punct_incl),
        }

    # Variant C: punct excluded, >=15 tokens
    if is_ge15 and not np.isnan(r1_obs_excl) and not np.isnan(mean_rpl_excl):
        results["C"] = {
            "observed_r1": r1_obs_excl,
            "baseline_r1": mean_rpl_excl,
            "excess": r1_obs_excl - mean_rpl_excl,
            "dd_length": len(dd_punct_excl),
        }

    # Variant D: punct excluded, >=20 tokens, projective only
    if is_ge20 and is_projective and not np.isnan(r1_obs_excl) and not np.isnan(mean_rpl_excl):
        results["D"] = {
            "observed_r1": r1_obs_excl,
            "baseline_r1": mean_rpl_excl,
            "excess": r1_obs_excl - mean_rpl_excl,
            "dd_length": len(dd_punct_excl),
        }

    return results


# ---------------------------------------------------------------------------
# Treebank-level processing (runs in worker process)
# ---------------------------------------------------------------------------

def process_treebank(treebank_id: str, sentences: list[dict], n_perms: int = 100) -> dict:
    """Process all sentences for one treebank, all 4 variants."""
    rng = np.random.default_rng(hash(treebank_id) % (2**32))

    variant_results = {v: [] for v in ["A", "B", "C", "D"]}
    n_skipped = 0

    for sent in sentences:
        try:
            results = process_sentence_all_variants(
                sent["head_array"],
                sent["deprel_array"],
                sent["dd_sequence"],
                sent["token_count"],
                n_perms,
                rng,
            )
            for v, res in results.items():
                variant_results[v].append(res)
        except Exception:
            n_skipped += 1
            continue

    # Aggregate per treebank per variant
    treebank_summary = {"n_skipped": n_skipped}
    for v in ["A", "B", "C", "D"]:
        vals = variant_results[v]
        if len(vals) < MIN_SENTENCES_EFFECT:
            treebank_summary[v] = None
            continue

        excess_arr = np.array([r["excess"] for r in vals])
        treebank_summary[v] = {
            "n_sentences": len(vals),
            "mean_excess": float(np.mean(excess_arr)),
            "var_excess": float(np.var(excess_arr, ddof=1)),
            "se_excess": float(np.std(excess_arr, ddof=1) / np.sqrt(len(vals))),
            "median_excess": float(np.median(excess_arr)),
            "prop_negative": float(np.mean(excess_arr < 0)),
            "iqr_excess": float(
                np.percentile(excess_arr, 75) - np.percentile(excess_arr, 25)
            ),
            "mean_observed_r1": float(np.mean([r["observed_r1"] for r in vals])),
            "mean_baseline_r1": float(np.mean([r["baseline_r1"] for r in vals])),
            "mean_dd_length": float(np.mean([r["dd_length"] for r in vals])),
        }

    return treebank_summary


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_typology(path: Path) -> dict:
    """Load data_id4 typological classification table."""
    logger.info(f"Loading typology from {path}")
    raw = json.loads(path.read_text())
    typology = {}
    for ds in raw["datasets"]:
        for ex in ds["examples"]:
            tb_id = ex["metadata_treebank_id"]
            typology[tb_id] = {
                "language_name": ex.get("metadata_language_name", ""),
                "language_family": ex.get("metadata_language_family"),
                "modality": ex.get("metadata_modality", "written"),
                "wals_case_category": ex.get("metadata_wals_case_category"),
                "wals_word_order_label": ex.get("metadata_wals_word_order_label"),
                "iso_639_3": ex.get("metadata_iso_639_3", ""),
            }
    logger.info(f"Loaded typology for {len(typology)} treebanks")
    return typology


def load_grammar_profiles(path: Path) -> dict:
    """Load data_id5 grammar profiles."""
    logger.info(f"Loading grammar profiles from {path}")
    raw = json.loads(path.read_text())
    profiles = {}
    for ds in raw["datasets"]:
        for ex in ds["examples"]:
            inp = json.loads(ex["input"])
            tb_id = inp["treebank_id"]
            out = json.loads(ex["output"])
            profiles[tb_id] = {
                "proportion_projective": out.get("structural_stats", {}).get(
                    "proportion_projective", 0.0
                ),
                "mean_tree_depth": out.get("structural_stats", {}).get(
                    "mean_tree_depth", 0.0
                ),
                "mean_branching_factor": out.get("structural_stats", {}).get(
                    "mean_branching_factor", 0.0
                ),
            }
    logger.info(f"Loaded profiles for {len(profiles)} treebanks")
    return profiles


def load_treebank_summaries(data_id3_dir: Path) -> dict:
    """Load treebank_summaries from any data_id3 part file's metadata."""
    # Read metadata from part 1
    part_path = data_id3_dir / "full_data_out_1.json"
    logger.info(f"Loading treebank summaries from {part_path}")
    raw = json.loads(part_path.read_text())
    summaries = raw.get("metadata", {}).get("treebank_summaries", {})
    logger.info(f"Loaded summaries for {len(summaries)} treebanks")
    return summaries


# ---------------------------------------------------------------------------
# STEP 1: Scope quantification
# ---------------------------------------------------------------------------

def compute_scope_quantification(
    data_id3_dir: Path, summaries: dict, typology: dict
) -> dict:
    """Compute scope quantification across all 233 qualifying treebanks."""
    logger.info("=== STEP 1: Scope Quantification ===")

    # Collect token counts from all parts for ge25 and ge30 thresholds
    tb_token_counts = defaultdict(list)
    for part_idx in range(1, 17):
        part_path = data_id3_dir / f"full_data_out_{part_idx}.json"
        if not part_path.exists():
            logger.warning(f"Part {part_idx} not found, skipping")
            continue
        logger.info(f"Scanning part {part_idx}/16 for scope quantification")
        raw = json.loads(part_path.read_text())
        for ds in raw["datasets"]:
            for ex in ds["examples"]:
                tb_id = ex["metadata_treebank_id"]
                tc = int(ex["metadata_token_count"])
                tb_token_counts[tb_id].append(tc)
        del raw
        gc.collect()

    # Compute per-treebank scope
    qualifying_tbs = {
        tb_id: s
        for tb_id, s in summaries.items()
        if s.get("qualifies_primary", False)
    }

    per_treebank_scope = []
    for tb_id, s in qualifying_tbs.items():
        total = s["total_sentences"]
        ge15 = s["sentences_ge15"]
        ge20 = s["sentences_ge20"]
        counts = tb_token_counts.get(tb_id, [])
        ge25 = sum(1 for tc in counts if tc >= 25)
        ge30 = sum(1 for tc in counts if tc >= 30)
        modality = typology.get(tb_id, {}).get("modality", "written")

        per_treebank_scope.append({
            "treebank_id": tb_id,
            "total_sentences": total,
            "ge15": ge15,
            "ge20": ge20,
            "ge25": ge25,
            "ge30": ge30,
            "prop_ge20": ge20 / total if total > 0 else 0.0,
            "modality": modality,
        })

    # By modality
    spoken_props = [
        t["prop_ge20"] for t in per_treebank_scope if t["modality"] == "spoken"
    ]
    written_props = [
        t["prop_ge20"] for t in per_treebank_scope if t["modality"] == "written"
    ]

    def _stats(arr):
        if not arr:
            return {"n": 0, "median": 0, "mean": 0, "p25": 0, "p50": 0, "p75": 0}
        a = np.array(arr)
        return {
            "n": len(a),
            "median": float(np.median(a)),
            "mean": float(np.mean(a)),
            "p25": float(np.percentile(a, 25)),
            "p50": float(np.percentile(a, 50)),
            "p75": float(np.percentile(a, 75)),
        }

    total_ge20 = sum(t["ge20"] for t in per_treebank_scope)
    total_ge15 = sum(t["ge15"] for t in per_treebank_scope)

    scope = {
        "total_treebanks": len(qualifying_tbs),
        "thresholds": {
            "ge15": {
                "total_sentences": total_ge15,
                "n_treebanks_with_50plus": sum(
                    1 for t in per_treebank_scope if t["ge15"] >= 50
                ),
            },
            "ge20": {
                "total_sentences": total_ge20,
                "n_treebanks_with_50plus": sum(
                    1 for t in per_treebank_scope if t["ge20"] >= 50
                ),
            },
            "ge25": {
                "total_sentences": sum(t["ge25"] for t in per_treebank_scope),
                "n_treebanks_with_50plus": sum(
                    1 for t in per_treebank_scope if t["ge25"] >= 50
                ),
            },
            "ge30": {
                "total_sentences": sum(t["ge30"] for t in per_treebank_scope),
                "n_treebanks_with_50plus": sum(
                    1 for t in per_treebank_scope if t["ge30"] >= 50
                ),
            },
        },
        "by_modality": {
            "spoken": _stats(spoken_props),
            "written": _stats(written_props),
        },
        "per_treebank_scope": sorted(
            per_treebank_scope, key=lambda x: x["ge20"], reverse=True
        ),
    }

    logger.info(
        f"Scope: {len(qualifying_tbs)} treebanks, {total_ge20} total ge20 sentences, "
        f"{len(spoken_props)} spoken, {len(written_props)} written"
    )
    return scope


# ---------------------------------------------------------------------------
# STEP 2: Treebank selection
# ---------------------------------------------------------------------------

def select_treebanks(
    summaries: dict, typology: dict, target: int = TARGET_TREEBANKS
) -> list[dict]:
    """Select ~50 typologically diverse treebanks."""
    logger.info("=== STEP 2: Treebank Selection ===")

    qualifying = {
        tb_id: s
        for tb_id, s in summaries.items()
        if s.get("qualifies_primary", False) and s["sentences_ge20"] >= 50
    }

    # Mandatory: all spoken with >=50 ge20
    selected_ids = set()
    selected_meta = []

    for tb_id, s in qualifying.items():
        modality = typology.get(tb_id, {}).get("modality", "written")
        if modality == "spoken":
            selected_ids.add(tb_id)
            selected_meta.append(_tb_meta(tb_id, s, typology))

    logger.info(f"Selected {len(selected_ids)} spoken treebanks")

    # Written treebank sampling
    written_candidates = {
        tb_id: s
        for tb_id, s in qualifying.items()
        if tb_id not in selected_ids
        and typology.get(tb_id, {}).get("modality", "written") != "spoken"
    }

    # Round 1: one per family (largest treebank)
    family_groups = defaultdict(list)
    for tb_id, s in written_candidates.items():
        fam = typology.get(tb_id, {}).get("language_family") or "Unknown"
        family_groups[fam].append((tb_id, s))

    families_sorted = sorted(
        family_groups.keys(),
        key=lambda f: max(s["sentences_ge20"] for _, s in family_groups[f]),
        reverse=True,
    )

    for fam in families_sorted[:20]:
        best_tb, best_s = max(family_groups[fam], key=lambda x: x[1]["sentences_ge20"])
        if best_tb not in selected_ids:
            selected_ids.add(best_tb)
            selected_meta.append(_tb_meta(best_tb, best_s, typology))

    logger.info(f"After Round 1 (families): {len(selected_ids)} treebanks")

    # Round 2: fill typological gaps (word order coverage)
    word_orders_present = {
        typology.get(tb, {}).get("wals_word_order_label")
        for tb in selected_ids
    } - {None}

    all_word_orders = {
        typology.get(tb, {}).get("wals_word_order_label")
        for tb in written_candidates
    } - {None}

    missing_wo = all_word_orders - word_orders_present
    for wo in missing_wo:
        candidates = [
            (tb_id, s)
            for tb_id, s in written_candidates.items()
            if tb_id not in selected_ids
            and typology.get(tb_id, {}).get("wals_word_order_label") == wo
        ]
        if candidates:
            best_tb, best_s = max(candidates, key=lambda x: x[1]["sentences_ge20"])
            selected_ids.add(best_tb)
            selected_meta.append(_tb_meta(best_tb, best_s, typology))

    logger.info(f"After Round 2 (word orders): {len(selected_ids)} treebanks")

    # Round 3: add multi-treebank languages
    multi_lang_targets = ["en", "cs", "fr", "de", "ja", "zh", "ar", "ru", "es", "fi", "ko", "tr"]
    for lang_prefix in multi_lang_targets:
        lang_tbs = [
            (tb_id, s)
            for tb_id, s in written_candidates.items()
            if tb_id not in selected_ids and tb_id.startswith(lang_prefix + "_")
        ]
        if lang_tbs:
            # Check if we already have one from this language
            existing = [
                tb for tb in selected_ids if tb.startswith(lang_prefix + "_")
            ]
            if existing:
                # Add second-best
                best_tb, best_s = max(lang_tbs, key=lambda x: x[1]["sentences_ge20"])
                selected_ids.add(best_tb)
                selected_meta.append(_tb_meta(best_tb, best_s, typology))
            elif len(lang_tbs) >= 2:
                # Add top 2
                sorted_tbs = sorted(lang_tbs, key=lambda x: x[1]["sentences_ge20"], reverse=True)
                for tb_id, s in sorted_tbs[:2]:
                    selected_ids.add(tb_id)
                    selected_meta.append(_tb_meta(tb_id, s, typology))

    logger.info(f"After Round 3 (multi-lang): {len(selected_ids)} treebanks")

    # Round 4: fill to target with largest remaining
    remaining = [
        (tb_id, s)
        for tb_id, s in written_candidates.items()
        if tb_id not in selected_ids
    ]
    remaining.sort(key=lambda x: x[1]["sentences_ge20"], reverse=True)

    needed = target - len(selected_ids)
    for tb_id, s in remaining[:max(0, needed)]:
        selected_ids.add(tb_id)
        selected_meta.append(_tb_meta(tb_id, s, typology))

    logger.info(f"Final selection: {len(selected_ids)} treebanks")
    return selected_meta


def _tb_meta(tb_id: str, summary: dict, typology: dict) -> dict:
    t = typology.get(tb_id, {})
    return {
        "id": tb_id,
        "language": t.get("language_name", ""),
        "family": t.get("language_family", "Unknown"),
        "modality": t.get("modality", "written"),
        "case_cat": t.get("wals_case_category"),
        "word_order": t.get("wals_word_order_label"),
        "n_ge20": summary["sentences_ge20"],
    }


# ---------------------------------------------------------------------------
# STEP 4: Load sentence data and process treebanks in parallel
# ---------------------------------------------------------------------------

def load_sentences_for_treebanks(
    data_id3_dir: Path, selected_ids: set[str]
) -> dict[str, list[dict]]:
    """Load sentence data from data_id3 parts for selected treebanks."""
    logger.info("=== Loading sentence data ===")
    treebank_sentences = defaultdict(list)

    for part_idx in range(1, 17):
        part_path = data_id3_dir / f"full_data_out_{part_idx}.json"
        if not part_path.exists():
            logger.warning(f"Part {part_idx} not found")
            continue
        logger.info(f"Loading part {part_idx}/16")
        raw = json.loads(part_path.read_text())
        for ds in raw["datasets"]:
            for ex in ds["examples"]:
                tb_id = ex["metadata_treebank_id"]
                if tb_id not in selected_ids:
                    continue
                try:
                    inp = json.loads(ex["input"])
                    out = json.loads(ex["output"])
                    treebank_sentences[tb_id].append({
                        "head_array": inp["head_array"],
                        "deprel_array": inp["deprel_array"],
                        "dd_sequence": out["dd_sequence"],
                        "token_count": out["token_count"],
                        "sentence_id": ex.get("metadata_sentence_id", ""),
                    })
                except (json.JSONDecodeError, KeyError) as e:
                    logger.debug(f"Skipping bad example in {tb_id}: {e}")
                    continue
        del raw
        gc.collect()

    total_sents = sum(len(v) for v in treebank_sentences.values())
    logger.info(
        f"Loaded {total_sents} sentences across {len(treebank_sentences)} treebanks"
    )
    return dict(treebank_sentences)


def run_parallel_processing(
    treebank_sentences: dict[str, list[dict]], n_perms: int, n_workers: int
) -> dict[str, dict]:
    """Process all treebanks in parallel."""
    logger.info(
        f"=== STEP 4: Parallel Processing ({len(treebank_sentences)} treebanks, "
        f"{n_workers} workers, {n_perms} perms) ==="
    )

    all_results = {}
    tb_items = list(treebank_sentences.items())

    # Use ProcessPoolExecutor for CPU-bound work
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        for tb_id, sents in tb_items:
            fut = executor.submit(process_treebank, tb_id, sents, n_perms)
            futures[fut] = tb_id

        done_count = 0
        for fut in as_completed(futures):
            tb_id = futures[fut]
            done_count += 1
            try:
                result = fut.result(timeout=600)
                all_results[tb_id] = result
                n_skip = result.get("n_skipped", 0)
                a_n = result.get("A", {})
                a_n_str = f"A:{a_n.get('n_sentences', 0)}" if a_n else "A:None"
                logger.info(
                    f"  [{done_count}/{len(tb_items)}] {tb_id}: {a_n_str}, "
                    f"skipped={n_skip}"
                )
            except Exception as e:
                logger.error(f"Failed processing {tb_id}: {e}")
                all_results[tb_id] = {v: None for v in ["A", "B", "C", "D"]}

    return all_results


# ---------------------------------------------------------------------------
# STEP 5: Meta-analysis
# ---------------------------------------------------------------------------

def run_meta_analysis(treebank_results: dict, variant_key: str) -> dict:
    """Run REML meta-analysis for one variant across treebanks."""
    effect_sizes = []
    variances = []
    treebank_ids = []

    for tb_id, results in treebank_results.items():
        r = results.get(variant_key)
        if r is None:
            continue
        if r["n_sentences"] < MIN_SENTENCES_META:
            continue
        effect_sizes.append(r["mean_excess"])
        variances.append(r["var_excess"] / r["n_sentences"])
        treebank_ids.append(tb_id)

    if len(effect_sizes) < 3:
        logger.warning(f"Variant {variant_key}: only {len(effect_sizes)} treebanks, skipping meta")
        return {"error": f"Too few treebanks ({len(effect_sizes)})"}

    y = np.array(effect_sizes)
    v = np.array(variances)
    # Clamp tiny variances to avoid division issues
    v = np.maximum(v, 1e-12)

    K = len(y)

    # Try PyMARE REML first, fallback to DerSimonian-Laird
    pooled = se_re = ci_low = ci_up = p_val = tau2 = float("nan")
    method_used = "none"

    try:
        from pymare import Dataset
        from pymare.estimators import VarianceBasedLikelihoodEstimator

        dset = Dataset(y=y, v=v)
        reml = VarianceBasedLikelihoodEstimator(method="REML")
        reml.fit_dataset(dset)
        summary_df = reml.summary().to_df()
        pooled = float(np.asarray(summary_df["estimate"].values[0]).item())
        se_re = float(np.asarray(summary_df["se"].values[0]).item())
        ci_low = float(np.asarray(summary_df["ci_0.025"].values[0]).item())
        ci_up = float(np.asarray(summary_df["ci_0.975"].values[0]).item())
        p_val = float(np.asarray(summary_df["p-value"].values[0]).item())
        tau2 = float(np.asarray(reml.params_["tau2"]).ravel()[0])
        method_used = "REML"
    except Exception as e:
        logger.warning(f"REML failed for variant {variant_key}: {e}, trying DL")
        try:
            from pymare import Dataset
            from pymare.estimators import DerSimonianLaird

            dset = Dataset(y=y, v=v)
            dl = DerSimonianLaird()
            dl.fit_dataset(dset)
            summary_df = dl.summary().to_df()
            pooled = float(np.asarray(summary_df["estimate"].values[0]).item())
            se_re = float(np.asarray(summary_df["se"].values[0]).item())
            ci_low = float(np.asarray(summary_df["ci_0.025"].values[0]).item())
            ci_up = float(np.asarray(summary_df["ci_0.975"].values[0]).item())
            p_val = float(np.asarray(summary_df["p-value"].values[0]).item())
            tau2 = float(np.asarray(dl.params_["tau2"]).ravel()[0])
            method_used = "DerSimonian-Laird"
        except Exception as e2:
            logger.error(f"DL also failed for variant {variant_key}: {e2}")
            # Manual DL calculation
            try:
                w = 1.0 / v
                theta_fe = np.sum(w * y) / np.sum(w)
                Q = np.sum(w * (y - theta_fe) ** 2)
                c = np.sum(w) - np.sum(w**2) / np.sum(w)
                tau2 = max(0.0, float((Q - (K - 1)) / c))
                w_re = 1.0 / (v + tau2)
                pooled = float(np.sum(w_re * y) / np.sum(w_re))
                se_re = float(1.0 / np.sqrt(np.sum(w_re)))
                ci_low = float(pooled - 1.96 * se_re)
                ci_up = float(pooled + 1.96 * se_re)
                z = pooled / se_re if se_re > 0 else 0
                from scipy.stats import norm
                p_val = float(2 * (1 - norm.cdf(abs(z))))
                method_used = "manual_DL"
            except Exception as e3:
                logger.error(f"Manual DL failed: {e3}")
                return {"error": str(e3)}

    # Heterogeneity
    I2 = tau2 / (tau2 + np.mean(v)) if (tau2 + np.mean(v)) > 0 else 0.0

    # Prediction interval
    try:
        pi_half = t_dist.ppf(0.975, max(K - 2, 1)) * np.sqrt(se_re**2 + tau2)
        pi = [float(pooled - pi_half), float(pooled + pi_half)]
    except Exception:
        pi = [float("nan"), float("nan")]

    return {
        "pooled_estimate": pooled,
        "pooled_se": se_re,
        "pooled_ci_lower": ci_low,
        "pooled_ci_upper": ci_up,
        "pooled_p_value": p_val,
        "tau2": float(tau2),
        "I2": float(I2),
        "prediction_interval": pi,
        "K_treebanks": K,
        "method_used": method_used,
        "per_treebank_effects": {
            tb: {"effect": float(e), "variance": float(va)}
            for tb, e, va in zip(treebank_ids, y, v)
        },
        "median_effect": float(np.median(y)),
        "prop_negative": float(np.mean(y < 0)),
    }


def compute_variant_comparisons(
    treebank_results: dict, meta_results: dict
) -> dict:
    """Compute pairwise variant comparisons."""
    logger.info("Computing variant comparisons")
    comparisons = {}

    # A vs B: punctuation effect
    paired_diffs_ab = []
    for tb_id, res in treebank_results.items():
        a = res.get("A")
        b = res.get("B")
        if a is not None and b is not None:
            paired_diffs_ab.append(a["mean_excess"] - b["mean_excess"])

    if len(paired_diffs_ab) >= 5:
        diffs = np.array(paired_diffs_ab)
        try:
            stat, p = wilcoxon(diffs)
        except Exception:
            stat, p = float("nan"), float("nan")
        comparisons["A_vs_B_punct_effect"] = {
            "mean_diff": float(np.mean(diffs)),
            "median_diff": float(np.median(diffs)),
            "CI_95": [
                float(np.percentile(diffs, 2.5)),
                float(np.percentile(diffs, 97.5)),
            ],
            "p_wilcoxon": float(p),
            "n_paired": len(diffs),
            "interpretation": (
                "Punctuation exclusion makes excess MORE negative"
                if np.mean(diffs) < 0
                else "Punctuation exclusion makes excess LESS negative"
            ),
        }

    # A vs C: threshold sensitivity
    a_meta = meta_results.get("A", {})
    c_meta = meta_results.get("C", {})
    if "pooled_estimate" in a_meta and "pooled_estimate" in c_meta:
        a_pool = a_meta["pooled_estimate"]
        c_pool = c_meta["pooled_estimate"]
        ratio = c_pool / a_pool if abs(a_pool) > 1e-10 else float("nan")
        comparisons["A_vs_C_threshold_sensitivity"] = {
            "A_pooled": a_pool,
            "C_pooled": c_pool,
            "ratio_C_over_A": float(ratio),
            "delta_pooled": float(c_pool - a_pool),
            "interpretation": (
                "Lower threshold attenuates effect"
                if abs(c_pool) < abs(a_pool)
                else "Lower threshold strengthens or maintains effect"
            ),
        }

    # A vs D: projectivity effect
    d_meta = meta_results.get("D", {})
    if "pooled_estimate" in a_meta and "pooled_estimate" in d_meta:
        a_pool = a_meta["pooled_estimate"]
        d_pool = d_meta["pooled_estimate"]
        # Count proportion excluded
        n_a = sum(
            1 for r in treebank_results.values()
            if r.get("A") is not None
        )
        n_d = sum(
            1 for r in treebank_results.values()
            if r.get("D") is not None
        )
        comparisons["A_vs_D_projectivity_effect"] = {
            "A_pooled": a_pool,
            "D_pooled": d_pool,
            "delta_pooled": float(a_pool - d_pool),
            "treebanks_A": n_a,
            "treebanks_D": n_d,
            "interpretation": (
                "Non-projective sentences contribute to the effect"
                if abs(a_pool) > abs(d_pool)
                else "Effect is robust to excluding non-projective sentences"
            ),
        }

    return comparisons


# ---------------------------------------------------------------------------
# STEP 6: Within-language aggregation sensitivity
# ---------------------------------------------------------------------------

def within_language_sensitivity(
    treebank_results: dict, typology: dict, meta_a: dict
) -> dict:
    """Compare per-treebank vs per-language meta-analysis."""
    logger.info("=== STEP 6: Within-Language Sensitivity ===")

    # Group by language
    lang_map = defaultdict(list)
    for tb_id in treebank_results:
        r = treebank_results[tb_id].get("A")
        if r is None:
            continue
        if r["n_sentences"] < MIN_SENTENCES_META:
            continue
        lang = typology.get(tb_id, {}).get("language_name", tb_id.split("_")[0])
        lang_map[lang].append(tb_id)

    multi_lang = {lang: tbs for lang, tbs in lang_map.items() if len(tbs) > 1}
    logger.info(f"Found {len(multi_lang)} languages with multiple treebanks")

    # Compute per-language aggregated effects
    lang_effects = []
    lang_variances = []
    lang_ids = []
    multi_details = []

    for lang, tbs in lang_map.items():
        if len(tbs) == 1:
            tb = tbs[0]
            r = treebank_results[tb]["A"]
            eff = r["mean_excess"]
            var = r["var_excess"] / r["n_sentences"]
            lang_effects.append(eff)
            lang_variances.append(var)
            lang_ids.append(lang)
        else:
            # Inverse-variance weighted
            effects = []
            vars_ = []
            for tb in tbs:
                r = treebank_results[tb]["A"]
                e = r["mean_excess"]
                v = r["var_excess"] / r["n_sentences"]
                effects.append(e)
                vars_.append(v)

            w = np.array([1.0 / max(v, 1e-12) for v in vars_])
            theta_lang = float(np.sum(w * np.array(effects)) / np.sum(w))
            v_lang = float(1.0 / np.sum(w))
            lang_effects.append(theta_lang)
            lang_variances.append(v_lang)
            lang_ids.append(lang)

            multi_details.append({
                "language": lang,
                "treebanks": tbs,
                "per_lang_effect": theta_lang,
                "per_lang_var": v_lang,
                "per_treebank_effects": [
                    {"tb": tb, "effect": e, "var": v}
                    for tb, e, v in zip(tbs, effects, vars_)
                ],
            })

    # Run meta-analysis on per-language effects
    if len(lang_effects) >= 3:
        lang_meta = run_meta_analysis_raw(
            np.array(lang_effects), np.array(lang_variances), lang_ids
        )
    else:
        lang_meta = {"error": "Too few languages"}

    # Compare to per-treebank meta
    comparison = {}
    if "pooled_estimate" in meta_a and "pooled_estimate" in lang_meta:
        comparison = {
            "delta_pooled": float(
                lang_meta["pooled_estimate"] - meta_a["pooled_estimate"]
            ),
            "delta_I2": float(lang_meta.get("I2", 0) - meta_a.get("I2", 0)),
            "delta_CI_width": float(
                (lang_meta.get("pooled_ci_upper", 0) - lang_meta.get("pooled_ci_lower", 0))
                - (meta_a.get("pooled_ci_upper", 0) - meta_a.get("pooled_ci_lower", 0))
            ),
        }

    return {
        "multi_treebank_languages": multi_details,
        "per_language_meta_analysis": lang_meta,
        "comparison_to_per_treebank": comparison,
    }


def run_meta_analysis_raw(y: np.ndarray, v: np.ndarray, ids: list) -> dict:
    """Run meta-analysis on raw effect/variance arrays."""
    K = len(y)
    v = np.maximum(v, 1e-12)

    try:
        from pymare import Dataset
        from pymare.estimators import VarianceBasedLikelihoodEstimator

        dset = Dataset(y=y, v=v)
        reml = VarianceBasedLikelihoodEstimator(method="REML")
        reml.fit_dataset(dset)
        summary_df = reml.summary().to_df()
        pooled = float(np.asarray(summary_df["estimate"].values[0]).item())
        se_re = float(np.asarray(summary_df["se"].values[0]).item())
        ci_low = float(np.asarray(summary_df["ci_0.025"].values[0]).item())
        ci_up = float(np.asarray(summary_df["ci_0.975"].values[0]).item())
        p_val = float(np.asarray(summary_df["p-value"].values[0]).item())
        tau2 = float(np.asarray(reml.params_["tau2"]).ravel()[0])
        method_used = "REML"
    except Exception:
        # Fallback to manual DL
        w = 1.0 / v
        theta_fe = np.sum(w * y) / np.sum(w)
        Q = np.sum(w * (y - theta_fe) ** 2)
        c = np.sum(w) - np.sum(w**2) / np.sum(w)
        tau2 = max(0.0, float((Q - (K - 1)) / c))
        w_re = 1.0 / (v + tau2)
        pooled = float(np.sum(w_re * y) / np.sum(w_re))
        se_re = float(1.0 / np.sqrt(np.sum(w_re)))
        ci_low = float(pooled - 1.96 * se_re)
        ci_up = float(pooled + 1.96 * se_re)
        z = pooled / se_re if se_re > 0 else 0
        from scipy.stats import norm
        p_val = float(2 * (1 - norm.cdf(abs(z))))
        method_used = "manual_DL"

    I2 = tau2 / (tau2 + float(np.mean(v))) if (tau2 + float(np.mean(v))) > 0 else 0.0

    try:
        pi_half = t_dist.ppf(0.975, max(K - 2, 1)) * np.sqrt(se_re**2 + tau2)
        pi = [float(pooled - pi_half), float(pooled + pi_half)]
    except Exception:
        pi = [float("nan"), float("nan")]

    return {
        "pooled_estimate": pooled,
        "pooled_se": se_re,
        "pooled_ci_lower": ci_low,
        "pooled_ci_upper": ci_up,
        "pooled_p_value": p_val,
        "tau2": tau2,
        "I2": I2,
        "prediction_interval": pi,
        "K": K,
        "method_used": method_used,
        "median_effect": float(np.median(y)),
        "prop_negative": float(np.mean(y < 0)),
    }


# ---------------------------------------------------------------------------
# STEP 7: Concrete examples
# ---------------------------------------------------------------------------

def extract_concrete_examples(
    treebank_results: dict,
    treebank_sentences: dict[str, list[dict]],
    n_top: int = 5,
    n_examples_per: int = 10,
) -> dict:
    """Extract top example sentences showing compensatory anti-correlation."""
    logger.info("=== STEP 7: Concrete Examples ===")

    # Find top-N treebanks by most negative mean_excess (variant A)
    tb_effects = []
    for tb_id, res in treebank_results.items():
        a = res.get("A")
        if a is not None and a["n_sentences"] >= MIN_SENTENCES_META:
            tb_effects.append((tb_id, a["mean_excess"]))

    tb_effects.sort(key=lambda x: x[1])
    top_tbs = [tb_id for tb_id, _ in tb_effects[:n_top]]
    logger.info(f"Top {n_top} treebanks: {top_tbs}")

    examples = []
    for tb_id in top_tbs:
        sents = treebank_sentences.get(tb_id, [])
        if not sents:
            continue

        rng = np.random.default_rng(hash(tb_id) % (2**32))

        # Process each sentence for variant A to get per-sentence metrics
        sent_data = []
        for sent in sents:
            if sent["token_count"] < 20:
                continue
            dd_excl = compute_dd_punct_excluded(sent["head_array"], sent["deprel_array"])
            if len(dd_excl) < 4:
                continue
            r1_obs = lag1_autocorrelation(dd_excl)
            if np.isnan(r1_obs):
                continue

            # Quick baseline (use fewer perms for example selection)
            root, children = build_children_map(sent["head_array"])
            if root is None:
                continue
            rpl_r1s = []
            for _ in range(30):
                try:
                    order = _rpl_recursive(root, children, rng)
                except RecursionError:
                    continue
                if len(order) != len(sent["head_array"]):
                    continue
                dd_rpl = compute_dd_from_linearization(
                    order, sent["head_array"], sent["deprel_array"], True
                )
                if len(dd_rpl) >= 4:
                    rpl_r1s.append(lag1_autocorrelation(dd_rpl))
            if not rpl_r1s:
                continue
            baseline_r1 = float(np.nanmean(rpl_r1s))
            excess = r1_obs - baseline_r1

            # Alternation score: count sign changes in diff(DD)
            dd_arr = np.array(dd_excl, dtype=float)
            if len(dd_arr) >= 2:
                diffs = np.diff(dd_arr)
                sign_changes = np.sum(np.abs(np.diff(np.sign(diffs))) > 0)
                alt_score = sign_changes / len(dd_arr) if len(dd_arr) > 0 else 0
            else:
                alt_score = 0

            sent_data.append({
                "sentence_id": sent["sentence_id"],
                "token_count": sent["token_count"],
                "head_array": sent["head_array"],
                "deprel_array": sent["deprel_array"],
                "dd_sequence": dd_excl,
                "observed_r1": r1_obs,
                "baseline_r1": baseline_r1,
                "excess": excess,
                "alt_score": alt_score,
            })

        # Select sentences with clearly negative excess and high alternation
        if not sent_data:
            continue
        median_excess = np.median([s["excess"] for s in sent_data])
        qualifying = [s for s in sent_data if s["excess"] < median_excess]
        qualifying.sort(key=lambda s: s["alt_score"], reverse=True)

        for s in qualifying[:n_examples_per]:
            # Build alternation markup
            dd = s["dd_sequence"]
            markup_parts = []
            for idx, d in enumerate(dd):
                if idx > 0:
                    arrow = "v" if d < dd[idx - 1] else "^"
                    markup_parts.append(f" {arrow} ")
                markup_parts.append(f"dd={d}")
            markup = "".join(markup_parts)

            examples.append({
                "treebank_id": tb_id,
                "sentence_id": s["sentence_id"],
                "token_count": s["token_count"],
                "deprel_array": s["deprel_array"],
                "head_array": s["head_array"],
                "dd_sequence": s["dd_sequence"],
                "observed_r1": round(s["observed_r1"], 4),
                "baseline_r1": round(s["baseline_r1"], 4),
                "excess": round(s["excess"], 4),
                "alternation_markup": markup[:500],
            })

    logger.info(f"Extracted {len(examples)} concrete examples")
    return {"top_5_treebanks": top_tbs, "examples": examples}


# ---------------------------------------------------------------------------
# STEP 8: Build within-treebank distributions
# ---------------------------------------------------------------------------

def build_within_treebank_distributions(treebank_results: dict) -> dict:
    """Build per-variant within-treebank distributions."""
    distributions = {}
    for v in ["A", "B", "C", "D"]:
        all_prop_neg = []
        per_tb = []
        for tb_id, res in treebank_results.items():
            r = res.get(v)
            if r is None:
                continue
            all_prop_neg.append(r["prop_negative"])
            per_tb.append({"treebank_id": tb_id, "prop_negative": r["prop_negative"]})

        if all_prop_neg:
            distributions[v] = {
                "global_prop_negative_sentences": float(np.mean(all_prop_neg)),
                "global_median_per_treebank_prop_negative": float(np.median(all_prop_neg)),
                "per_treebank_prop_negative": sorted(
                    per_tb, key=lambda x: x["prop_negative"]
                ),
            }
        else:
            distributions[v] = {"error": "No data"}

    return {"per_variant": distributions}


# ---------------------------------------------------------------------------
# Format output to exp_gen_sol_out schema
# ---------------------------------------------------------------------------

def format_output(
    scope: dict,
    selected_treebanks: list[dict],
    variant_comparison: dict,
    meta_results: dict,
    pairwise_comparisons: dict,
    within_lang: dict,
    concrete_examples: dict,
    within_tb_dist: dict,
) -> dict:
    """Format all results into exp_gen_sol_out schema."""

    # Build the full analysis result as a single JSON string for the output field
    full_result = {
        "experiment_id": "experiment_iter3_dir3",
        "title": "Robustness & Sensitivity Analyses",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scope_quantification": scope,
        "selected_treebanks": {
            "count": len(selected_treebanks),
            "selection_criteria": (
                "All qualifying spoken treebanks + stratified written sample: "
                "diverse families, word orders, case categories, "
                "multi-treebank languages, filled to ~50 total."
            ),
            "treebanks": selected_treebanks,
        },
        "variant_comparison": {
            "A_punct_excl_ge20": {
                "description": "Primary: punct excluded, >=20 tokens, all sentences",
                "meta_analysis": meta_results.get("A", {}),
            },
            "B_punct_incl_ge20": {
                "description": "Punct included, >=20 tokens",
                "meta_analysis": meta_results.get("B", {}),
            },
            "C_punct_excl_ge15": {
                "description": "Punct excluded, >=15 tokens (lower threshold)",
                "meta_analysis": meta_results.get("C", {}),
            },
            "D_projective_ge20": {
                "description": "Punct excluded, >=20 tokens, projective only",
                "meta_analysis": meta_results.get("D", {}),
            },
            "pairwise_comparisons": pairwise_comparisons,
        },
        "within_language_sensitivity": within_lang,
        "concrete_examples": concrete_examples,
        "within_treebank_distributions": within_tb_dist,
    }

    # Schema: { datasets: [{ dataset: str, examples: [{ input: str, output: str, metadata_*: ... }] }] }
    # We produce one dataset with one example per treebank
    examples = []

    # Example 1: The full analysis summary
    examples.append({
        "input": json.dumps({
            "analysis_type": "robustness_sensitivity_4variant",
            "variants": ["A_punct_excl_ge20", "B_punct_incl_ge20", "C_punct_excl_ge15", "D_projective_ge20"],
            "n_treebanks": len(selected_treebanks),
            "n_permutations": N_PERMS,
        }),
        "output": json.dumps(full_result),
        "metadata_experiment_id": "experiment_iter3_dir3",
        "metadata_analysis_type": "robustness_sensitivity",
        "predict_variant_A": json.dumps({
            "pooled_excess": meta_results.get("A", {}).get("pooled_estimate"),
            "I2": meta_results.get("A", {}).get("I2"),
        }),
        "predict_baseline_rpl": json.dumps({
            "description": "Random projective linearization baseline (100 permutations per sentence)",
            "pooled_baseline_r1": "computed per-sentence, subtracted to get excess",
        }),
    })

    # Per-treebank examples (one per selected treebank with variant A results)
    for tb_meta in selected_treebanks:
        tb_id = tb_meta["id"]
        a_result = meta_results.get("A", {}).get("per_treebank_effects", {}).get(tb_id, {})
        b_result = meta_results.get("B", {}).get("per_treebank_effects", {}).get(tb_id, {})
        c_result = meta_results.get("C", {}).get("per_treebank_effects", {}).get(tb_id, {})
        d_result = meta_results.get("D", {}).get("per_treebank_effects", {}).get(tb_id, {})

        tb_input = {
            "treebank_id": tb_id,
            "language": tb_meta.get("language", ""),
            "family": tb_meta.get("family", ""),
            "modality": tb_meta.get("modality", ""),
            "n_sentences_ge20": tb_meta.get("n_ge20", 0),
        }

        tb_output = {
            "variant_A_effect": a_result.get("effect"),
            "variant_A_variance": a_result.get("variance"),
            "variant_B_effect": b_result.get("effect"),
            "variant_C_effect": c_result.get("effect"),
            "variant_D_effect": d_result.get("effect"),
        }

        examples.append({
            "input": json.dumps(tb_input),
            "output": json.dumps(tb_output),
            "metadata_treebank_id": tb_id,
            "metadata_language": tb_meta.get("language", ""),
            "metadata_family": tb_meta.get("family", ""),
            "metadata_modality": tb_meta.get("modality", ""),
            "predict_variant_A": str(round(a_result["effect"], 6)) if a_result.get("effect") is not None else "null",
            "predict_baseline_rpl": "0.0",
        })

    return {
        "metadata": {
            "experiment_id": "experiment_iter3_dir3",
            "title": "Robustness & Sensitivity: 4-Variant Analysis",
            "method_name": "compensatory_anti_correlation_4variant",
            "description": (
                "Four pipeline variants testing sensitivity to punctuation, "
                "length threshold, and projectivity constraints on dependency "
                "distance autocorrelation excess."
            ),
            "n_permutations": N_PERMS,
            "n_treebanks_selected": len(selected_treebanks),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "variants": {
                "A": "punct excluded, >=20 tokens",
                "B": "punct included, >=20 tokens",
                "C": "punct excluded, >=15 tokens",
                "D": "punct excluded, >=20 tokens, projective only",
            },
        },
        "datasets": [
            {
                "dataset": "robustness_sensitivity_analysis",
                "examples": examples,
            }
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@logger.catch
def main():
    t_start = time.time()
    logger.info("=" * 70)
    logger.info("Starting Robustness & Sensitivity Analysis")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Load reference data
    # ------------------------------------------------------------------
    typology = load_typology(DATA_ID4_PATH)
    profiles = load_grammar_profiles(DATA_ID5_PATH)
    summaries = load_treebank_summaries(DATA_ID3_DIR)

    # ------------------------------------------------------------------
    # STEP 1: Scope quantification
    # ------------------------------------------------------------------
    scope = compute_scope_quantification(DATA_ID3_DIR, summaries, typology)

    # ------------------------------------------------------------------
    # STEP 2: Treebank selection
    # ------------------------------------------------------------------
    selected_meta = select_treebanks(summaries, typology, TARGET_TREEBANKS)
    selected_ids = {t["id"] for t in selected_meta}
    logger.info(f"Selected {len(selected_ids)} treebanks: {sorted(selected_ids)[:10]}...")

    # ------------------------------------------------------------------
    # STEP 3-4: Load data and process in parallel
    # ------------------------------------------------------------------
    treebank_sentences = load_sentences_for_treebanks(DATA_ID3_DIR, selected_ids)

    # Adjust n_perms based on data size and time budget
    total_sents = sum(len(v) for v in treebank_sentences.values())
    elapsed = time.time() - t_start
    logger.info(f"Data loading took {elapsed:.1f}s. Total sentences: {total_sents}")

    n_workers = max(1, NUM_CPUS)
    n_perms = N_PERMS

    # Adjust perms only if extremely large
    if total_sents > 500_000:
        n_perms = 75
        logger.warning(f"Very large dataset ({total_sents} sents), reducing to {n_perms} perms")

    treebank_results = run_parallel_processing(treebank_sentences, n_perms, n_workers)

    elapsed = time.time() - t_start
    logger.info(f"Processing took {elapsed:.1f}s total so far")

    # ------------------------------------------------------------------
    # STEP 5: Meta-analysis for each variant
    # ------------------------------------------------------------------
    logger.info("=== STEP 5: Meta-Analysis ===")
    meta_results = {}
    for variant in ["A", "B", "C", "D"]:
        logger.info(f"Running meta-analysis for variant {variant}")
        meta_results[variant] = run_meta_analysis(treebank_results, variant)
        pooled = meta_results[variant].get("pooled_estimate", "N/A")
        I2 = meta_results[variant].get("I2", "N/A")
        K = meta_results[variant].get("K_treebanks", "N/A")
        logger.info(f"  Variant {variant}: pooled={pooled}, I2={I2}, K={K}")

    # ------------------------------------------------------------------
    # Pairwise comparisons
    # ------------------------------------------------------------------
    pairwise = compute_variant_comparisons(treebank_results, meta_results)

    # ------------------------------------------------------------------
    # STEP 6: Within-language sensitivity
    # ------------------------------------------------------------------
    within_lang = within_language_sensitivity(
        treebank_results, typology, meta_results.get("A", {})
    )

    # ------------------------------------------------------------------
    # STEP 7: Concrete examples
    # ------------------------------------------------------------------
    concrete = extract_concrete_examples(treebank_results, treebank_sentences)

    # ------------------------------------------------------------------
    # Within-treebank distributions
    # ------------------------------------------------------------------
    within_tb_dist = build_within_treebank_distributions(treebank_results)

    # ------------------------------------------------------------------
    # STEP 8: Format and write output
    # ------------------------------------------------------------------
    logger.info("=== STEP 8: Writing Output ===")
    output = format_output(
        scope,
        selected_meta,
        treebank_results,
        meta_results,
        pairwise,
        within_lang,
        concrete,
        within_tb_dist,
    )

    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Wrote output to {out_path}")

    # Check file size
    size_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info(f"Output size: {size_mb:.1f} MB")

    elapsed = time.time() - t_start
    logger.info(f"Total runtime: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    logger.info("DONE")


if __name__ == "__main__":
    main()
