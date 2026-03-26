#!/usr/bin/env python3
"""Core Pipeline: Three-Tier Baselines, Excess Autocorrelation, Meta-Analysis & Typological Regression.

Computes bias-corrected lag-1 autocorrelation of dependency-distance sequences,
subtracts baselines from tree-linearization permutations (RPL/FHD/SOP),
aggregates per-treebank via inverse-variance weighting, runs random-effects
meta-analysis and mixed-effects typological regression.
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
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ---------------------------------------------------------------------------
# Hardware detection (cgroup-aware)
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
TOTAL_RAM_GB = _container_ram_gb() or 29.0
RAM_BUDGET_BYTES = int(TOTAL_RAM_GB * 0.75 * 1e9)  # 75% of container RAM
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET_BYTES * 3, RAM_BUDGET_BYTES * 3))

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, budget {RAM_BUDGET_BYTES/1e9:.1f} GB")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).resolve().parent
DEP_BASE = Path("/ai-inventor/aii_pipeline/runs/comp-ling-dobrovoljc_ebw/3_invention_loop/iter_1/gen_art")
DATA3_DIR = DEP_BASE / "data_id3_it1__opus" / "data_out"
DATA3_MINI = DEP_BASE / "data_id3_it1__opus" / "mini_data_out.json"
DATA5_FULL = DEP_BASE / "data_id5_it1__opus" / "full_data_out.json"
DATA4_FULL = DEP_BASE / "data_id4_it1__opus" / "full_data_out.json"
PART_FILES = sorted(DATA3_DIR.glob("full_data_out_*.json"))

MAX_COMPUTE_SECONDS = 300 * 60  # 300 minutes hard limit
START_TIME = time.time()

# ---------------------------------------------------------------------------
# STEP 0: Data Loading
# ---------------------------------------------------------------------------

def load_treebank_summaries() -> dict:
    """Load treebank summaries from mini_data_out.json metadata."""
    logger.info("Loading treebank summaries...")
    data = json.loads(DATA3_MINI.read_text())
    ts = data["metadata"]["treebank_summaries"]
    logger.info(f"  Loaded summaries for {len(ts)} treebanks")
    return ts


def load_grammar_profiles() -> dict:
    """Load grammar profiles from data_id5."""
    logger.info("Loading grammar profiles from data_id5...")
    data = json.loads(DATA5_FULL.read_text())
    profiles = {}
    for ex in data["datasets"][0]["examples"]:
        inp = json.loads(ex["input"])
        tb_id = inp["treebank_id"]
        out = json.loads(ex["output"])
        profiles[tb_id] = out
    logger.info(f"  Loaded profiles for {len(profiles)} treebanks")
    return profiles


def load_typology() -> dict:
    """Load typology table from data_id4."""
    logger.info("Loading typology from data_id4...")
    data = json.loads(DATA4_FULL.read_text())
    typology = {}
    for ex in data["datasets"][0]["examples"]:
        tb_id = ex["metadata_treebank_id"]
        typology[tb_id] = {
            "language_name": ex.get("metadata_language_name"),
            "wals_case_category": ex.get("metadata_wals_case_category"),
            "wals_word_order_label": ex.get("metadata_wals_word_order_label"),
            "language_family": ex.get("metadata_language_family"),
            "modality": ex.get("metadata_modality"),
            "ud_case_proportion": ex.get("metadata_ud_case_proportion"),
            "iso639_3": ex.get("metadata_iso_639_3"),
            "glottocode": ex.get("metadata_glottocode"),
        }
    logger.info(f"  Loaded typology for {len(typology)} treebanks")
    return typology


def load_sentences_for_treebanks(target_treebanks: set, max_per_tb: int | None = None) -> dict:
    """Load ge20 sentences for target treebanks from the 16 part files.

    Each part file contains MULTIPLE datasets (one per treebank).
    Returns dict[treebank_id] -> list of sentence dicts.
    Each sentence dict has keys: head_array, deprel_array.
    """
    result = defaultdict(list)
    found_tbs = set()
    for part_path in PART_FILES:
        logger.debug(f"  Scanning {part_path.name}...")
        raw = json.loads(part_path.read_text())
        # Each part file has multiple datasets, one per treebank
        for ds in raw["datasets"]:
            ds_name = ds.get("dataset", "")
            # Quick skip: if dataset name is a treebank ID and not in targets
            if ds_name and ds_name not in target_treebanks:
                continue
            for ex in ds["examples"]:
                tb_id = ex.get("metadata_treebank_id", ds_name)
                if tb_id not in target_treebanks:
                    continue
                if ex.get("metadata_length_bucket") != "ge20":
                    continue
                if max_per_tb is not None and len(result[tb_id]) >= max_per_tb:
                    continue
                inp = json.loads(ex["input"])
                result[tb_id].append({
                    "head_array": inp["head_array"],
                    "deprel_array": inp["deprel_array"],
                })
                found_tbs.add(tb_id)
        del raw
        gc.collect()
        # Early exit if all target treebanks found and capped
        if max_per_tb is not None and all(
            len(result.get(tb, [])) >= max_per_tb for tb in target_treebanks if tb in found_tbs
        ) and found_tbs == target_treebanks:
            break
    return dict(result)


# ---------------------------------------------------------------------------
# STEP 0.5: Core Algorithm Functions
# ---------------------------------------------------------------------------

def filter_punctuation(head_array: list[int], deprel_array: list[str]):
    """Remove punct tokens and reindex heads.

    Returns (new_head_array, new_deprel_array) both 1-indexed, 0=root.
    """
    n = len(head_array)
    non_punct_idx = [i for i in range(n) if deprel_array[i] != "punct"]
    if not non_punct_idx:
        return [], []

    old_to_new = {0: 0}
    for new_i, old_i in enumerate(non_punct_idx):
        old_to_new[old_i + 1] = new_i + 1

    new_heads = []
    new_deprels = []
    for old_i in non_punct_idx:
        old_head = head_array[old_i]
        visited = set()
        while old_head != 0 and old_head not in old_to_new and old_head not in visited:
            visited.add(old_head)
            old_head = head_array[old_head - 1]
        new_heads.append(old_to_new.get(old_head, 0))
        new_deprels.append(deprel_array[old_i])

    return new_heads, new_deprels


def compute_dd_consecutive(head_array: list[int]) -> list[int]:
    """Compute DD sequence using consecutive positions 1..n.

    DD_i = |i - head(i)| for non-root tokens.
    """
    dd = []
    for i, h in enumerate(head_array):
        if h == 0:
            continue
        dd.append(abs((i + 1) - h))
    return dd


def lag1_autocorrelation(seq: list) -> float:
    """Standard lag-1 autocorrelation r1."""
    x = np.asarray(seq, dtype=np.float64)
    n = len(x)
    if n < 4:
        return np.nan
    xbar = x.mean()
    diffs = x - xbar
    denom = np.dot(diffs, diffs)
    if denom == 0:
        return np.nan
    numer = np.dot(diffs[:-1], diffs[1:])
    return numer / denom


def r1_plus(seq: list) -> float:
    """Bias-corrected r1+ = r1 + 1/n (Huitema & McKean 1991)."""
    r1 = lag1_autocorrelation(seq)
    if np.isnan(r1):
        return np.nan
    return r1 + 1.0 / len(seq)


compute_autocorrelation = r1_plus


def build_children_map(head_array: list[int]) -> tuple[dict, int | None]:
    """Build parent -> children map. Returns (children_map, root_node).

    head_array is 1-indexed, 0 = root.
    """
    children = defaultdict(list)
    root_node = None
    for i, h in enumerate(head_array):
        node_id = i + 1
        children[h].append(node_id)
        if h == 0:
            root_node = node_id
    return dict(children), root_node


# ---------------------------------------------------------------------------
# Baseline Linearization Functions
# ---------------------------------------------------------------------------

def rpl_linearize(children_map: dict, root: int, rng: np.random.RandomState) -> list[int]:
    """Random Projective Linearization."""
    def _lin(node):
        kids = children_map.get(node, [])
        if not kids:
            return [node]
        left, right = [], []
        for kid in kids:
            if rng.random() < 0.5:
                left.append(kid)
            else:
                right.append(kid)
        rng.shuffle(left)
        rng.shuffle(right)
        result = []
        for kid in left:
            result.extend(_lin(kid))
        result.append(node)
        for kid in right:
            result.extend(_lin(kid))
        return result
    return _lin(root)


def fhd_linearize(children_map: dict, deprel_map: dict, root: int,
                   head_dir_table: dict, rng: np.random.RandomState) -> list[int]:
    """Fixed Head-Direction Linearization."""
    def _lin(node):
        kids = children_map.get(node, [])
        if not kids:
            return [node]
        left, right = [], []
        for kid in kids:
            dep = deprel_map.get(kid, "dep")
            direction = head_dir_table.get(dep, "initial")
            if direction == "final":
                left.append(kid)
            else:
                right.append(kid)
        rng.shuffle(left)
        rng.shuffle(right)
        result = []
        for kid in left:
            result.extend(_lin(kid))
        result.append(node)
        for kid in right:
            result.extend(_lin(kid))
        return result
    return _lin(root)


def sop_linearize(children_map: dict, deprel_map: dict, root: int,
                  head_dir_table: dict, sop_lookup: dict,
                  rng: np.random.RandomState) -> list[int]:
    """Sibling-Order-Preserving Linearization.

    sop_lookup: dict[frozenset_of_deprels] -> {majority_order: [...], frequency: int}
    Falls back to FHD when template not found or frequency < 5.
    """
    def _lin_fhd(node):
        kids = children_map.get(node, [])
        if not kids:
            return [node]
        left, right = [], []
        for kid in kids:
            dep = deprel_map.get(kid, "dep")
            direction = head_dir_table.get(dep, "initial")
            if direction == "final":
                left.append(kid)
            else:
                right.append(kid)
        rng.shuffle(left)
        rng.shuffle(right)
        result = []
        for kid in left:
            result.extend(_lin(kid))
        result.append(node)
        for kid in right:
            result.extend(_lin(kid))
        return result

    def _lin(node):
        kids = children_map.get(node, [])
        if not kids:
            return [node]

        kid_deprels = frozenset(deprel_map.get(k, "dep") for k in kids)
        template = sop_lookup.get(kid_deprels)

        if template is None or template.get("frequency", 0) < 5:
            return _lin_fhd(node)

        ordering = template["majority_order"]
        deprel_to_kids = defaultdict(list)
        for kid in kids:
            deprel_to_kids[deprel_map.get(kid, "dep")].append(kid)

        ordered_result = []
        for slot in ordering:
            if slot == "HEAD":
                ordered_result.append(("HEAD", node))
            elif slot in deprel_to_kids and deprel_to_kids[slot]:
                kid = deprel_to_kids[slot].pop(0)
                ordered_result.append(("KID", kid))

        # If HEAD wasn't placed by template, insert it
        head_placed = any(t == "HEAD" for t, _ in ordered_result)
        if not head_placed:
            ordered_result.append(("HEAD", node))

        # Place unplaced kids via FHD
        placed_kids = {v for t, v in ordered_result if t == "KID"}
        for kid in kids:
            if kid not in placed_kids:
                dep = deprel_map.get(kid, "dep")
                direction = head_dir_table.get(dep, "initial")
                head_idx = next(
                    (i for i, (t, _) in enumerate(ordered_result) if t == "HEAD"),
                    len(ordered_result)
                )
                if direction == "final":
                    ordered_result.insert(head_idx, ("KID", kid))
                else:
                    ordered_result.insert(head_idx + 1, ("KID", kid))

        # Expand: recursively linearize kids
        final = []
        for tag, val in ordered_result:
            if tag == "HEAD":
                final.append(val)
            else:
                final.extend(_lin(val))
        return final

    return _lin(root)


def dd_from_linearization(linear_order: list[int], head_array: list[int]) -> list[int]:
    """Compute DD from a linearized order.

    linear_order: permutation of [1..n].
    head_array: 0-indexed list, head_array[i] = head of node (i+1), 0=root.
    """
    pos_map = {}
    for new_pos, node in enumerate(linear_order, 1):
        pos_map[node] = new_pos

    dd = []
    for new_pos, node in enumerate(linear_order, 1):
        head_node = head_array[node - 1]
        if head_node == 0:
            continue
        head_new_pos = pos_map.get(head_node)
        if head_new_pos is None:
            continue
        dd.append(abs(new_pos - head_new_pos))
    return dd


# ---------------------------------------------------------------------------
# Grammar Profile Extraction
# ---------------------------------------------------------------------------

def extract_head_dir_table(grammar_profile: dict) -> dict:
    """Extract majority head direction per deprel.

    Returns dict: deprel -> "initial" or "final".
    """
    hdp = grammar_profile.get("head_direction_profile", {})
    table = {}
    for deprel, stats in hdp.items():
        hi = stats.get("count_head_initial", 0)
        hf = stats.get("count_head_final", 0)
        table[deprel] = "initial" if hi >= hf else "final"
    return table


def extract_sop_lookup(grammar_profile: dict) -> dict:
    """Extract SOP templates as a lookup dict.

    Returns dict: frozenset(deprels_without_punct) -> {majority_order: [...], frequency: int}
    For duplicate deprel_sets (different head_upos), keeps the highest frequency.
    """
    templates_list = grammar_profile.get("sibling_order_templates", [])
    lookup = {}
    for tpl in templates_list:
        deprel_set = tpl.get("deprel_set", [])
        # Remove punct from deprel set
        deprels_clean = frozenset(d for d in deprel_set if d != "punct")
        if not deprels_clean:
            continue
        # Remove punct from majority_order
        order_clean = [s for s in tpl.get("majority_order", []) if s != "punct"]
        freq = tpl.get("frequency", 0)
        existing = lookup.get(deprels_clean)
        if existing is None or freq > existing.get("frequency", 0):
            lookup[deprels_clean] = {
                "majority_order": order_clean,
                "frequency": freq,
            }
    return lookup


# ---------------------------------------------------------------------------
# STEP 1: Process One Sentence
# ---------------------------------------------------------------------------

def process_sentence(head_array: list[int], deprel_array: list[str],
                     head_dir_table: dict, sop_lookup: dict,
                     n_perms: int, rng_seed: int) -> dict | None:
    """Process a single sentence: observed + baseline autocorrelations."""
    # 1a. Filter punctuation
    new_heads, new_deprels = filter_punctuation(head_array, deprel_array)
    n_nopunct = len(new_heads)
    if n_nopunct < 20:
        return None

    # 1b. Observed DD (consecutive positions in filtered tree)
    dd_obs = compute_dd_consecutive(new_heads)
    if len(dd_obs) < 4:
        return None
    r1_obs = compute_autocorrelation(dd_obs)
    if np.isnan(r1_obs):
        return None

    # 1c. Build tree structure
    children_map, root_node = build_children_map(new_heads)
    if root_node is None:
        return None

    deprel_map = {i + 1: new_deprels[i] for i in range(n_nopunct)}

    # 1d. Generate baseline permutations
    rng = np.random.RandomState(rng_seed)
    rpl_r1s, fhd_r1s, sop_r1s = [], [], []

    for perm_i in range(n_perms):
        # RPL
        try:
            lin = rpl_linearize(children_map, root_node, rng)
            dd = dd_from_linearization(lin, new_heads)
            if len(dd) >= 4:
                r = compute_autocorrelation(dd)
                if not np.isnan(r):
                    rpl_r1s.append(r)
        except RecursionError:
            pass

        # FHD
        try:
            lin = fhd_linearize(children_map, deprel_map, root_node, head_dir_table, rng)
            dd = dd_from_linearization(lin, new_heads)
            if len(dd) >= 4:
                r = compute_autocorrelation(dd)
                if not np.isnan(r):
                    fhd_r1s.append(r)
        except RecursionError:
            pass

        # SOP
        try:
            lin = sop_linearize(children_map, deprel_map, root_node,
                                head_dir_table, sop_lookup, rng)
            dd = dd_from_linearization(lin, new_heads)
            if len(dd) >= 4:
                r = compute_autocorrelation(dd)
                if not np.isnan(r):
                    sop_r1s.append(r)
        except RecursionError:
            pass

    # 1e. Compute excess measures
    mean_rpl = float(np.mean(rpl_r1s)) if rpl_r1s else np.nan
    mean_fhd = float(np.mean(fhd_r1s)) if fhd_r1s else np.nan
    mean_sop = float(np.mean(sop_r1s)) if sop_r1s else np.nan

    excess_rpl = r1_obs - mean_rpl if not np.isnan(mean_rpl) else np.nan
    excess_fhd = r1_obs - mean_fhd if not np.isnan(mean_fhd) else np.nan
    excess_sop = r1_obs - mean_sop if not np.isnan(mean_sop) else np.nan

    var_rpl = float(np.var(rpl_r1s) / len(rpl_r1s)) if rpl_r1s else np.nan
    var_fhd = float(np.var(fhd_r1s) / len(fhd_r1s)) if fhd_r1s else np.nan
    var_sop = float(np.var(sop_r1s) / len(sop_r1s)) if sop_r1s else np.nan

    return {
        "observed_r1": float(r1_obs),
        "mean_rpl": float(mean_rpl), "mean_fhd": float(mean_fhd), "mean_sop": float(mean_sop),
        "excess_rpl": float(excess_rpl), "excess_fhd": float(excess_fhd), "excess_sop": float(excess_sop),
        "var_excess_rpl": float(var_rpl), "var_excess_fhd": float(var_fhd), "var_excess_sop": float(var_sop),
        "n_tokens_nopunct": n_nopunct,
        "dd_length": len(dd_obs),
        "n_rpl_perms": len(rpl_r1s), "n_fhd_perms": len(fhd_r1s), "n_sop_perms": len(sop_r1s),
    }


def _process_sentence_wrapper(args):
    """Wrapper for multiprocessing."""
    head_array, deprel_array, head_dir_table, sop_lookup, n_perms, rng_seed = args
    try:
        return process_sentence(head_array, deprel_array, head_dir_table, sop_lookup, n_perms, rng_seed)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# STEP 1.5: Treebank Aggregation
# ---------------------------------------------------------------------------

def aggregate_treebank(sent_results: list[dict], typo_info: dict,
                       grammar_profile: dict) -> dict:
    """Aggregate sentence-level results into treebank-level summary."""
    def _safe_list(key):
        return [s[key] for s in sent_results if not np.isnan(s.get(key, np.nan))]

    excess_rpl = _safe_list("excess_rpl")
    excess_fhd = _safe_list("excess_fhd")
    excess_sop = _safe_list("excess_sop")

    def _stats(vals):
        if not vals:
            return {"mean": None, "median": None, "se": None,
                    "prop_negative": None, "iqr": None}
        a = np.array(vals)
        return {
            "mean": float(np.mean(a)),
            "median": float(np.median(a)),
            "se": float(np.std(a, ddof=1) / np.sqrt(len(a))) if len(a) > 1 else None,
            "prop_negative": float(np.mean(a < 0)),
            "iqr": float(np.percentile(a, 75) - np.percentile(a, 25)) if len(a) > 3 else None,
        }

    rpl_stats = _stats(excess_rpl)
    fhd_stats = _stats(excess_fhd)
    sop_stats = _stats(excess_sop)

    struct = grammar_profile.get("structural_stats", {})

    result = {
        "n_sentences": len(sent_results),
        # Excess RPL
        "mean_excess_rpl": rpl_stats["mean"],
        "median_excess_rpl": rpl_stats["median"],
        "se_excess_rpl": rpl_stats["se"],
        "prop_negative_rpl": rpl_stats["prop_negative"],
        "iqr_excess_rpl": rpl_stats["iqr"],
        # Excess FHD
        "mean_excess_fhd": fhd_stats["mean"],
        "median_excess_fhd": fhd_stats["median"],
        "se_excess_fhd": fhd_stats["se"],
        "prop_negative_fhd": fhd_stats["prop_negative"],
        # Excess SOP
        "mean_excess_sop": sop_stats["mean"],
        "median_excess_sop": sop_stats["median"],
        "se_excess_sop": sop_stats["se"],
        "prop_negative_sop": sop_stats["prop_negative"],
        # Observed distribution
        "mean_observed_r1": float(np.mean([s["observed_r1"] for s in sent_results])),
        "mean_dd_length": float(np.mean([s["dd_length"] for s in sent_results])),
        # Typological metadata
        **{k: v for k, v in typo_info.items()},
        # Structural stats from grammar profile
        "mean_tree_depth": struct.get("mean_tree_depth"),
        "mean_branching_factor": struct.get("mean_branching_factor"),
        "projectivity_rate": struct.get("proportion_projective"),
        "mean_sent_length_gp": struct.get("mean_sentence_length"),
    }
    return result


# ---------------------------------------------------------------------------
# STEP 2: Process a Tier of Treebanks
# ---------------------------------------------------------------------------

def process_tier(tb_ids: list[str], grammar_profiles: dict, typology: dict,
                 n_perms: int, max_sents: int | None, tier_name: str,
                 existing_results: dict | None = None) -> dict:
    """Process a tier of treebanks."""
    results = dict(existing_results or {})
    todo = [tb for tb in tb_ids if tb not in results]
    if not todo:
        logger.info(f"  {tier_name}: all {len(tb_ids)} treebanks already done")
        return results

    logger.info(f"  {tier_name}: processing {len(todo)} new treebanks ({n_perms} perms, max_sents={max_sents})")

    # Load sentences for all todo treebanks
    sentences_by_tb = load_sentences_for_treebanks(set(todo), max_per_tb=max_sents)
    logger.info(f"  Loaded sentences for {len(sentences_by_tb)} treebanks")

    for tb_idx, tb_id in enumerate(todo):
        elapsed = time.time() - START_TIME
        if elapsed > MAX_COMPUTE_SECONDS:
            logger.warning(f"  Time limit reached after {len(results)} treebanks in {tier_name}")
            break

        sents = sentences_by_tb.get(tb_id, [])
        if not sents:
            logger.debug(f"    {tb_id}: no sentences loaded, skipping")
            continue

        gp = grammar_profiles.get(tb_id, {})
        head_dir_table = extract_head_dir_table(gp)
        sop_lookup = extract_sop_lookup(gp)

        # Build args for parallel processing
        args_list = [
            (s["head_array"], s["deprel_array"], head_dir_table, sop_lookup,
             n_perms, hash((tb_id, i)) % (2**31))
            for i, s in enumerate(sents)
        ]

        # Process sentences in parallel
        sent_results = []
        n_workers = max(1, NUM_CPUS - 1)
        chunk_size = max(1, len(args_list) // (n_workers * 4))

        try:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                for res in pool.map(_process_sentence_wrapper, args_list, chunksize=chunk_size):
                    if res is not None:
                        sent_results.append(res)
        except Exception as e:
            logger.error(f"    {tb_id}: parallel processing failed ({e}), falling back to sequential")
            for args in args_list:
                res = _process_sentence_wrapper(args)
                if res is not None:
                    sent_results.append(res)

        if sent_results:
            typo = typology.get(tb_id, {})
            results[tb_id] = aggregate_treebank(sent_results, typo, gp)
            ex_val = results[tb_id].get('mean_excess_rpl')
            ex_str = f"{ex_val:.4f}" if ex_val is not None else "None"
            logger.info(f"    [{tb_idx+1}/{len(todo)}] {tb_id}: {len(sent_results)} sents, excess_rpl={ex_str}")
        else:
            logger.debug(f"    {tb_id}: no valid results")

        # Save checkpoint every 10 treebanks
        if (tb_idx + 1) % 10 == 0:
            _save_checkpoint(results, f"checkpoint_{tier_name}.json")

    _save_checkpoint(results, f"checkpoint_{tier_name}.json")
    logger.info(f"  {tier_name} complete: {len(results)} treebanks")
    return results


def _save_checkpoint(data: dict, filename: str):
    """Save checkpoint JSON."""
    path = WORKSPACE / filename
    # Convert any numpy types to native Python
    path.write_text(json.dumps(data, default=_json_default, indent=None))
    logger.debug(f"  Saved checkpoint: {filename}")


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj) if not np.isnan(obj) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return str(obj)


# ---------------------------------------------------------------------------
# Treebank Selection
# ---------------------------------------------------------------------------

def select_diverse_treebanks(qualifying: list[str], typology: dict, n: int = 10) -> list[str]:
    """Select diverse treebanks for mini tier."""
    # Prioritize: coverage of word orders, modalities, families
    spoken = [t for t in qualifying if typology.get(t, {}).get("modality") == "spoken"]
    sov = [t for t in qualifying if typology.get(t, {}).get("wals_word_order_label") == "SOV"]
    svo = [t for t in qualifying if typology.get(t, {}).get("wals_word_order_label") == "SVO"]
    vso = [t for t in qualifying if typology.get(t, {}).get("wals_word_order_label") == "VSO"]

    selected = set()
    # Pick well-known treebanks for diversity
    priority = [
        "en_ewt", "de_gsd", "ja_gsd", "tr_boun", "cs_pdt",
        "ar_padt", "zh_gsd", "hi_hdtb", "fi_tdt", "ko_kaist",
    ]
    for tb in priority:
        if tb in qualifying:
            selected.add(tb)
        if len(selected) >= n:
            break

    # Fill remaining with spoken treebanks
    for tb in spoken:
        if len(selected) >= n:
            break
        selected.add(tb)

    # Fill with diverse word orders
    for group in [vso, sov, svo]:
        for tb in group:
            if len(selected) >= n:
                break
            selected.add(tb)

    # Fill rest from qualifying
    for tb in qualifying:
        if len(selected) >= n:
            break
        selected.add(tb)

    return list(selected)[:n]


def select_medium_treebanks(qualifying: list[str], typology: dict, n: int = 50) -> list[str]:
    """Select medium-tier treebanks: all spoken + diverse written."""
    selected = set()
    # All spoken first
    for tb in qualifying:
        if typology.get(tb, {}).get("modality") == "spoken":
            selected.add(tb)

    # Then with WALS data
    for tb in qualifying:
        if len(selected) >= n:
            break
        if typology.get(tb, {}).get("wals_word_order_label") is not None:
            selected.add(tb)

    # Fill rest
    for tb in qualifying:
        if len(selected) >= n:
            break
        selected.add(tb)

    return list(selected)[:n]


# ---------------------------------------------------------------------------
# STEP 4: Meta-Analysis
# ---------------------------------------------------------------------------

def run_meta_analysis(results: dict) -> dict:
    """Run random-effects meta-analysis for each excess measure."""
    logger.info("Running meta-analysis...")
    meta_output = {}

    for tier_name, excess_key, se_key in [
        ("excess_RPL", "mean_excess_rpl", "se_excess_rpl"),
        ("excess_FHD", "mean_excess_fhd", "se_excess_fhd"),
        ("excess_SOP", "mean_excess_sop", "se_excess_sop"),
    ]:
        tb_ids = [tb for tb in results if results[tb].get(excess_key) is not None
                  and results[tb].get(se_key) is not None
                  and results[tb][se_key] > 0]
        if len(tb_ids) < 3:
            logger.warning(f"  {tier_name}: only {len(tb_ids)} treebanks, skipping")
            meta_output[tier_name] = {"error": "too_few_treebanks", "K": len(tb_ids)}
            continue

        y = np.array([results[tb][excess_key] for tb in tb_ids])
        v = np.array([results[tb][se_key] ** 2 for tb in tb_ids])

        # Ensure valid values
        valid = np.isfinite(y) & np.isfinite(v) & (v > 0)
        y, v = y[valid], v[valid]
        K = len(y)

        if K < 3:
            meta_output[tier_name] = {"error": "too_few_valid", "K": K}
            continue

        # Try PyMARE REML first, fallback to DerSimonian-Laird
        pooled_est, pooled_se, tau2 = _meta_reml(y, v)
        if pooled_est is None:
            pooled_est, pooled_se, tau2 = _meta_dl(y, v)

        if pooled_est is None:
            meta_output[tier_name] = {"error": "convergence_failure", "K": K}
            continue

        # Compute derived statistics
        typical_v = float(np.median(v))
        I2 = float(tau2 / (tau2 + typical_v)) if (tau2 + typical_v) > 0 else 0.0

        z = pooled_est / pooled_se if pooled_se > 0 else 0.0
        p_value = float(2 * stats.norm.sf(abs(z)))

        ci_lo = float(pooled_est - 1.96 * pooled_se)
        ci_hi = float(pooled_est + 1.96 * pooled_se)

        # Prediction interval
        V_pooled = 1.0 / np.sum(1.0 / v) if np.all(v > 0) else 0.0
        t_crit = float(stats.t.ppf(0.975, df=max(K - 2, 1)))
        pi_lo = float(pooled_est - t_crit * np.sqrt(tau2 + V_pooled))
        pi_hi = float(pooled_est + t_crit * np.sqrt(tau2 + V_pooled))

        meta_output[tier_name] = {
            "pooled_estimate": float(pooled_est),
            "pooled_se": float(pooled_se),
            "ci_95": [ci_lo, ci_hi],
            "tau_squared": float(tau2),
            "I_squared": float(I2),
            "prediction_interval": [pi_lo, pi_hi],
            "K_treebanks": int(K),
            "p_value": p_value,
            "prop_negative": float(np.mean(y < 0)),
            "median_effect": float(np.median(y)),
            "mean_effect": float(np.mean(y)),
        }

        logger.info(
            f"  {tier_name}: pooled={pooled_est:.4f}, CI=[{ci_lo:.4f},{ci_hi:.4f}], "
            f"I2={I2:.2f}, p={p_value:.2e}, prop_neg={np.mean(y<0):.2f}, K={K}"
        )

    return meta_output


def _meta_reml(y: np.ndarray, v: np.ndarray):
    """PyMARE REML meta-analysis."""
    try:
        from pymare import Dataset
        from pymare.estimators import VarianceBasedLikelihoodEstimator
        dataset = Dataset(y=y, v=v)
        est = VarianceBasedLikelihoodEstimator(method="REML")
        est.fit_dataset(dataset)
        summary = est.summary()
        pooled = float(summary.fe_params["estimate"].iloc[0])
        se = float(summary.fe_params["se"].iloc[0])
        tau2 = float(est.params_["tau2"])
        return pooled, se, tau2
    except Exception as e:
        logger.warning(f"  PyMARE REML failed: {e}, falling back to DL")
        return None, None, None


def _meta_dl(y: np.ndarray, v: np.ndarray):
    """DerSimonian-Laird random-effects meta-analysis (manual implementation)."""
    try:
        w = 1.0 / v
        w_sum = np.sum(w)
        pooled_fe = np.sum(w * y) / w_sum

        Q = np.sum(w * (y - pooled_fe) ** 2)
        K = len(y)
        C = w_sum - np.sum(w ** 2) / w_sum
        tau2 = max(0.0, (Q - (K - 1)) / C)

        w_re = 1.0 / (v + tau2)
        w_re_sum = np.sum(w_re)
        pooled_re = float(np.sum(w_re * y) / w_re_sum)
        se_re = float(1.0 / np.sqrt(w_re_sum))

        return pooled_re, se_re, float(tau2)
    except Exception as e:
        logger.error(f"  DL meta-analysis failed: {e}")
        return None, None, None


# ---------------------------------------------------------------------------
# STEP 4b: Sensitivity — Language-Level Aggregation
# ---------------------------------------------------------------------------

def run_language_level_meta(results: dict, typology: dict) -> dict:
    """Aggregate same-language treebanks before meta-analysis."""
    logger.info("Running language-level sensitivity meta-analysis...")
    # Group by iso639_3
    lang_groups = defaultdict(list)
    for tb_id, res in results.items():
        iso = typology.get(tb_id, {}).get("iso639_3", tb_id)
        if res.get("mean_excess_rpl") is not None and res.get("se_excess_rpl") is not None:
            lang_groups[iso].append(res)

    # Average within language
    lang_results = {}
    for iso, group in lang_groups.items():
        vals = [r["mean_excess_rpl"] for r in group if r["mean_excess_rpl"] is not None]
        ses = [r["se_excess_rpl"] for r in group if r["se_excess_rpl"] is not None and r["se_excess_rpl"] > 0]
        if vals and ses:
            mean_val = float(np.mean(vals))
            # pooled SE: 1/sqrt(sum(1/se^2))
            combined_se = float(1.0 / np.sqrt(sum(1.0 / (s ** 2) for s in ses)))
            lang_results[iso] = {"mean_excess_rpl": mean_val, "se_excess_rpl": combined_se}

    if len(lang_results) < 3:
        return {"error": "too_few_languages", "K": len(lang_results)}

    y = np.array([r["mean_excess_rpl"] for r in lang_results.values()])
    v = np.array([r["se_excess_rpl"] ** 2 for r in lang_results.values()])

    pooled, se, tau2 = _meta_dl(y, v)
    if pooled is None:
        return {"error": "convergence_failure"}

    return {
        "pooled_estimate": float(pooled),
        "pooled_se": float(se),
        "tau_squared": float(tau2),
        "K_languages": len(lang_results),
        "p_value": float(2 * stats.norm.sf(abs(pooled / se))) if se > 0 else None,
    }


# ---------------------------------------------------------------------------
# STEP 5: Typological Regression
# ---------------------------------------------------------------------------

def run_regression(results: dict, typology: dict) -> dict:
    """Run mixed-effects typological regression."""
    logger.info("Running typological regression...")
    reg_output = {}

    # Build DataFrame
    rows = []
    for tb_id, res in results.items():
        typo = typology.get(tb_id, {})
        if res.get("mean_excess_rpl") is None:
            continue
        rows.append({
            "treebank": tb_id,
            "excess_rpl": res["mean_excess_rpl"],
            "excess_fhd": res.get("mean_excess_fhd"),
            "excess_sop": res.get("mean_excess_sop"),
            "modality": typo.get("modality", "written"),
            "word_order": typo.get("wals_word_order_label"),
            "case_richness_wals": typo.get("wals_case_category"),
            "ud_case_proportion": typo.get("ud_case_proportion", 0.0),
            "language_family": typo.get("language_family", "Unknown"),
            "mean_sent_length": res.get("mean_dd_length"),
            "n_sentences": res["n_sentences"],
        })

    if len(rows) < 10:
        logger.warning(f"Only {len(rows)} treebanks for regression, too few")
        return {"error": "too_few_treebanks", "n": len(rows)}

    df = pd.DataFrame(rows)
    df["is_spoken"] = (df["modality"] == "spoken").astype(int)
    df["case_measure"] = pd.to_numeric(df["ud_case_proportion"], errors="coerce").fillna(0.0)

    # WALS-UD cross-validation
    wals_mask = df["case_richness_wals"].notna()
    if wals_mask.sum() >= 10:
        from scipy.stats import spearmanr
        rho, p = spearmanr(
            df.loc[wals_mask, "case_richness_wals"].astype(float),
            df.loc[wals_mask, "case_measure"]
        )
        reg_output["wals_ud_spearman"] = {"rho": float(rho), "p": float(p)}
        logger.info(f"  WALS-UD case Spearman rho={rho:.3f}, p={p:.4f}")

    # Prepare regression data
    df_reg = df.dropna(subset=["excess_rpl", "language_family", "case_measure", "mean_sent_length"]).copy()
    family_counts = df_reg["language_family"].value_counts()
    valid_families = family_counts[family_counts >= 2].index.tolist()
    df_reg = df_reg[df_reg["language_family"].isin(valid_families)].copy()

    # Center continuous predictors for interpretable intercept
    case_mean = df_reg["case_measure"].mean()
    len_mean = df_reg["mean_sent_length"].mean()
    df_reg["case_c"] = df_reg["case_measure"] - case_mean
    df_reg["len_c"] = df_reg["mean_sent_length"] - len_mean
    reg_output["predictor_centering"] = {"case_mean": float(case_mean), "len_mean": float(len_mean)}

    import warnings
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Always run OLS as primary (robust, no convergence issues)
    if len(df_reg) >= 15:
        try:
            X = df_reg[["is_spoken", "case_c", "len_c"]].copy()
            X = sm.add_constant(X)
            ols = sm.OLS(df_reg["excess_rpl"], X).fit()
            reg_output["model1_ols_summary"] = ols.summary().as_text()
            reg_output["model1_ols_params"] = {
                k: {"coef": float(v), "se": float(ols.bse[k]), "p": float(ols.pvalues[k])}
                for k, v in ols.params.items()
            }
            logger.info(f"  OLS Model 1: {dict(ols.params.items())}")
        except Exception as e:
            logger.error(f"  OLS Model 1 failed: {e}")

    # Try MixedLM (family random intercepts) as secondary
    if len(df_reg) >= 15 and len(valid_families) >= 3:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model1 = smf.mixedlm(
                    "excess_rpl ~ is_spoken + case_c + len_c",
                    data=df_reg, groups=df_reg["language_family"]
                )
                fit1 = model1.fit(reml=True, method="lbfgs")
            reg_output["model1_mixed_summary"] = fit1.summary().as_text()
            reg_output["model1_mixed_converged"] = fit1.converged
            reg_output["model1_mixed_params"] = {
                k: {"coef": float(v), "se": float(fit1.bse[k]), "p": float(fit1.pvalues[k])}
                for k, v in fit1.fe_params.items()
            }
            re_var = float(fit1.cov_re.iloc[0, 0]) if hasattr(fit1.cov_re, 'iloc') else 0.0
            reg_output["model1_mixed_re_variance"] = re_var
            logger.info(f"  MixedLM Model 1 converged: {fit1.converged}, RE var={re_var:.6f}")
            logger.info(f"  MixedLM params: {dict(fit1.fe_params)}")
        except Exception as e:
            logger.error(f"  MixedLM Model 1 failed: {e}")
    else:
        logger.warning(f"  Not enough data for MixedLM: {len(df_reg)} rows, {len(valid_families)} families")

    # Model 2: Add word order (subset with WALS data)
    df_wo = df_reg[df_reg["word_order"].isin(["SOV", "SVO", "VSO"])].copy()
    if len(df_wo) >= 30:
        # OLS with word order
        try:
            X2 = pd.get_dummies(df_wo["word_order"], drop_first=True, dtype=float)
            X2["is_spoken"] = df_wo["is_spoken"].values
            X2["case_c"] = df_wo["case_c"].values
            X2["len_c"] = df_wo["len_c"].values
            X2 = sm.add_constant(X2)
            ols2 = sm.OLS(df_wo["excess_rpl"].values, X2).fit()
            reg_output["model2_ols_summary"] = ols2.summary().as_text()
            reg_output["model2_ols_params"] = {
                k: {"coef": float(v), "se": float(ols2.bse[k]), "p": float(ols2.pvalues[k])}
                for k, v in ols2.params.items()
            }
            logger.info(f"  OLS Model 2: {dict(ols2.params.items())}")
        except Exception as e:
            logger.error(f"  OLS Model 2 failed: {e}")
        # MixedLM with word order
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model2 = smf.mixedlm(
                    "excess_rpl ~ is_spoken + case_c + C(word_order) + len_c",
                    data=df_wo, groups=df_wo["language_family"]
                )
                fit2 = model2.fit(reml=True, method="lbfgs")
            reg_output["model2_mixed_summary"] = fit2.summary().as_text()
            reg_output["model2_mixed_converged"] = fit2.converged
            logger.info(f"  MixedLM Model 2 converged: {fit2.converged}")
        except Exception as e:
            logger.error(f"  MixedLM Model 2 failed: {e}")

    # Spoken vs Written comparison
    spoken = df_reg[df_reg["is_spoken"] == 1]
    written = df_reg[df_reg["is_spoken"] == 0]
    reg_output["n_spoken"] = len(spoken)
    reg_output["n_written"] = len(written)
    if len(spoken) >= 3 and len(written) >= 3:
        sp_mean = float(spoken["excess_rpl"].mean())
        wr_mean = float(written["excess_rpl"].mean())
        pooled_var = (spoken["excess_rpl"].var() + written["excess_rpl"].var()) / 2
        cohens_d = float((sp_mean - wr_mean) / np.sqrt(pooled_var)) if pooled_var > 0 else 0.0
        t_stat, t_p = stats.ttest_ind(spoken["excess_rpl"], written["excess_rpl"], equal_var=False)
        reg_output["spoken_written"] = {
            "spoken_mean": sp_mean, "written_mean": wr_mean,
            "cohens_d": cohens_d, "t_stat": float(t_stat), "p_value": float(t_p),
        }
        logger.info(f"  Spoken({len(spoken)}) mean={sp_mean:.4f} vs Written({len(written)}) mean={wr_mean:.4f}, d={cohens_d:.3f}")

    # Repeat for FHD and SOP (simplified: just descriptive stats by group)
    for measure_key, measure_name in [("excess_fhd", "FHD"), ("excess_sop", "SOP")]:
        col = f"excess_{measure_name.lower()}"
        if col not in df_reg.columns:
            col = measure_key
        subset = df_reg.dropna(subset=[col]) if col in df_reg.columns else pd.DataFrame()
        if len(subset) >= 10:
            reg_output[f"{measure_name}_descriptive"] = {
                "mean": float(subset[col].mean()),
                "se": float(subset[col].std() / np.sqrt(len(subset))),
                "n": len(subset),
            }

    return reg_output


# ---------------------------------------------------------------------------
# STEP 4c: Forest Plot
# ---------------------------------------------------------------------------

def generate_forest_plot(results: dict, measure_key: str, se_key: str,
                         meta_result: dict, filename: str, title: str):
    """Generate a forest plot for the top treebanks."""
    try:
        # Sort by n_sentences descending, take top 50
        tb_ids = sorted(
            [tb for tb in results if results[tb].get(measure_key) is not None
             and results[tb].get(se_key) is not None and results[tb][se_key] > 0],
            key=lambda t: results[t]["n_sentences"], reverse=True
        )[:50]

        if len(tb_ids) < 3:
            logger.warning(f"  Too few treebanks for forest plot: {len(tb_ids)}")
            return

        estimates = [results[tb][measure_key] for tb in tb_ids]
        ses = [results[tb][se_key] for tb in tb_ids]
        ci_lo = [e - 1.96 * s for e, s in zip(estimates, ses)]
        ci_hi = [e + 1.96 * s for e, s in zip(estimates, ses)]

        fig, ax = plt.subplots(figsize=(10, max(8, len(tb_ids) * 0.3)))
        y_pos = list(range(len(tb_ids)))

        ax.errorbar(estimates, y_pos, xerr=[
            [e - lo for e, lo in zip(estimates, ci_lo)],
            [hi - e for e, hi in zip(estimates, ci_hi)]
        ], fmt='o', color='steelblue', ecolor='steelblue', elinewidth=1, capsize=2, markersize=4)

        # Add pooled estimate
        pooled = meta_result.get("pooled_estimate")
        if pooled is not None:
            pooled_ci = meta_result.get("ci_95", [pooled, pooled])
            ax.axvspan(pooled_ci[0], pooled_ci[1], alpha=0.2, color='red')
            ax.axvline(pooled, color='red', linestyle='--', linewidth=1, label=f'Pooled={pooled:.4f}')

        ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(tb_ids, fontsize=6)
        ax.set_xlabel(title)
        ax.set_title(f"Forest Plot: {title} (top {len(tb_ids)} treebanks)")
        ax.legend(fontsize=8)
        ax.invert_yaxis()
        plt.tight_layout()
        fig.savefig(WORKSPACE / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Saved forest plot: {filename}")
    except Exception as e:
        logger.error(f"  Forest plot generation failed: {e}")


# ---------------------------------------------------------------------------
# STEP 6: Compile Final Output
# ---------------------------------------------------------------------------

def compile_output(results: dict, meta_output: dict, lang_meta: dict,
                   reg_output: dict, typology: dict, scale: str) -> dict:
    """Compile final output in exp_gen_sol_out schema format."""
    logger.info("Compiling final output...")

    examples = []
    for tb_id, res in sorted(results.items()):
        typo = typology.get(tb_id, {})
        input_dict = {
            "treebank_id": tb_id,
            "language_name": typo.get("language_name"),
            "iso639_3": typo.get("iso639_3"),
            "language_family": typo.get("language_family"),
            "modality": typo.get("modality"),
            "wals_word_order_label": typo.get("wals_word_order_label"),
            "wals_case_category": typo.get("wals_case_category"),
            "ud_case_proportion": typo.get("ud_case_proportion"),
        }
        output_dict = {k: v for k, v in res.items()
                       if k not in ("language_name", "wals_case_category",
                                    "wals_word_order_label", "language_family",
                                    "modality", "ud_case_proportion", "iso639_3",
                                    "glottocode")}
        example = {
            "input": json.dumps(input_dict, default=_json_default),
            "output": json.dumps(output_dict, default=_json_default),
            "metadata_treebank_id": tb_id,
            "metadata_language_family": typo.get("language_family", "Unknown"),
            "metadata_modality": typo.get("modality", "unknown"),
        }
        # Add predict_ fields
        if res.get("mean_excess_rpl") is not None:
            example["predict_excess_rpl"] = f"{res['mean_excess_rpl']:.6f}"
        if res.get("mean_excess_fhd") is not None:
            example["predict_excess_fhd"] = f"{res['mean_excess_fhd']:.6f}"
        if res.get("mean_excess_sop") is not None:
            example["predict_excess_sop"] = f"{res['mean_excess_sop']:.6f}"

        examples.append(example)

    # Build plots list
    plots = []
    for fname in ["forest_plot_excess_rpl.png", "forest_plot_excess_fhd.png", "forest_plot_excess_sop.png"]:
        if (WORKSPACE / fname).exists():
            plots.append(fname)

    # Success criteria evaluation
    success = {}
    rpl_meta = meta_output.get("excess_RPL", {})
    sop_meta = meta_output.get("excess_SOP", {})
    if "p_value" in rpl_meta:
        success["criterion_1_pooled_negative"] = rpl_meta["p_value"] < 0.001
        success["criterion_1_median_above_threshold"] = (
            abs(rpl_meta.get("median_effect", 0)) >= 0.05
        )
    if "p_value" in sop_meta:
        success["criterion_1_hierarchy_survives"] = sop_meta["p_value"] < 0.05
    sw = reg_output.get("spoken_written", {})
    if "cohens_d" in sw:
        success["criterion_2_spoken_written_d"] = sw["cohens_d"]
    m1_params = reg_output.get("model1_ols_params", reg_output.get("model1_mixed_params", {}))
    if "case_c" in m1_params:
        success["criterion_3_case_p"] = m1_params["case_c"]["p"]

    output = {
        "metadata": {
            "method_name": "DD_Autocorrelation_Excess_Analysis",
            "description": (
                "Bias-corrected lag-1 autocorrelation of dependency-distance sequences, "
                "minus three-tier baselines (RPL/FHD/SOP), aggregated via meta-analysis "
                "and typological mixed-effects regression."
            ),
            "parameters": {
                "n_perms_mini": 50,
                "n_perms_medium": 100,
                "n_perms_full": 100,
                "min_tokens_after_punct": 20,
                "autocorrelation_estimator": "r1_plus",
            },
            "scale_achieved": scale,
            "n_treebanks_processed": len(results),
            "total_sentences_processed": sum(r["n_sentences"] for r in results.values()),
            "meta_analysis": meta_output,
            "language_level_sensitivity": lang_meta,
            "typological_regression": reg_output,
            "success_criteria_evaluation": success,
            "plots_generated": plots,
            "compute_time_seconds": round(time.time() - START_TIME, 1),
        },
        "datasets": [
            {
                "dataset": "dd_autocorrelation_excess",
                "examples": examples,
            }
        ],
    }
    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@logger.catch
def main():
    logger.info("=" * 70)
    logger.info("DD Autocorrelation Excess Pipeline - Starting")
    logger.info("=" * 70)

    # 0. Load reference data
    treebank_summaries = load_treebank_summaries()
    grammar_profiles = load_grammar_profiles()
    typology = load_typology()

    qualifying = [tb for tb, info in treebank_summaries.items() if info.get("qualifies_primary")]
    logger.info(f"Qualifying treebanks: {len(qualifying)}")

    # ===================================================================
    # TIER 1: MINI — 10 treebanks, 50 permutations
    # ===================================================================
    logger.info("=" * 70)
    logger.info("TIER 1: MINI (10 treebanks, 50 perms)")
    logger.info("=" * 70)

    mini_tbs = select_diverse_treebanks(qualifying, typology, n=10)
    logger.info(f"  Selected: {mini_tbs}")

    t0 = time.time()
    mini_results = process_tier(mini_tbs, grammar_profiles, typology,
                                n_perms=50, max_sents=None, tier_name="mini")
    mini_time = time.time() - t0
    logger.info(f"  MINI took {mini_time:.1f}s for {len(mini_results)} treebanks")

    # Signal check
    excess_vals = [v["mean_excess_rpl"] for v in mini_results.values()
                   if v.get("mean_excess_rpl") is not None]
    if excess_vals:
        mean_ex = float(np.mean(excess_vals))
        prop_neg = float(np.mean(np.array(excess_vals) < 0))
        logger.info(f"  MINI signal: mean_excess_rpl={mean_ex:.4f}, prop_negative={prop_neg:.2f}")
    else:
        logger.warning("  MINI: no valid excess values!")

    # Estimate time per treebank
    time_per_tb = mini_time / max(len(mini_results), 1)
    logger.info(f"  Estimated time per treebank: {time_per_tb:.1f}s")

    # ===================================================================
    # TIER 2: MEDIUM — 50 treebanks, 100 permutations, max 500 sent/tb
    # ===================================================================
    elapsed = time.time() - START_TIME
    remaining = MAX_COMPUTE_SECONDS - elapsed
    estimated_medium = 50 * time_per_tb * 2  # 2x for more perms
    logger.info(f"  Elapsed: {elapsed:.0f}s, remaining: {remaining:.0f}s, est medium: {estimated_medium:.0f}s")

    if remaining > estimated_medium * 1.5:
        logger.info("=" * 70)
        logger.info("TIER 2: MEDIUM (50 treebanks, 100 perms, max 500 sent/tb)")
        logger.info("=" * 70)

        medium_tbs = select_medium_treebanks(qualifying, typology, n=50)
        t0 = time.time()
        medium_results = process_tier(medium_tbs, grammar_profiles, typology,
                                      n_perms=100, max_sents=500, tier_name="medium",
                                      existing_results=mini_results)
        medium_time = time.time() - t0
        logger.info(f"  MEDIUM took {medium_time:.1f}s for {len(medium_results)} treebanks")
        time_per_tb = medium_time / max(len(medium_results) - len(mini_results), 1)
    else:
        logger.warning("  Skipping MEDIUM tier: insufficient time")
        medium_results = mini_results

    # ===================================================================
    # TIER 3: FULL — all qualifying treebanks, 100 permutations
    # ===================================================================
    elapsed = time.time() - START_TIME
    remaining = MAX_COMPUTE_SECONDS - elapsed
    n_remaining_tbs = len(qualifying) - len(medium_results)
    estimated_full = n_remaining_tbs * time_per_tb * 2

    if remaining > 300 and n_remaining_tbs > 0:  # at least 5 min remaining
        logger.info("=" * 70)
        logger.info(f"TIER 3: FULL ({len(qualifying)} treebanks, 100 perms)")
        logger.info(f"  Need to process {n_remaining_tbs} more treebanks, est {estimated_full:.0f}s")
        logger.info("=" * 70)

        full_results = process_tier(qualifying, grammar_profiles, typology,
                                    n_perms=100, max_sents=None, tier_name="full",
                                    existing_results=medium_results)
    else:
        logger.warning(f"  Skipping FULL tier: {remaining:.0f}s remaining, need est {estimated_full:.0f}s")
        full_results = medium_results

    # Determine scale achieved
    n_done = len(full_results)
    if n_done >= 200:
        scale = "full"
    elif n_done >= 40:
        scale = "medium"
    else:
        scale = "mini"

    logger.info(f"Scale achieved: {scale} ({n_done} treebanks)")

    # ===================================================================
    # ANALYSIS
    # ===================================================================
    logger.info("=" * 70)
    logger.info("ANALYSIS PHASE")
    logger.info("=" * 70)

    # Meta-analysis
    meta_output = run_meta_analysis(full_results)

    # Language-level sensitivity
    lang_meta = run_language_level_meta(full_results, typology)

    # Typological regression
    reg_output = run_regression(full_results, typology)

    # Forest plots
    for measure, se_key, meta_key, fname, title in [
        ("mean_excess_rpl", "se_excess_rpl", "excess_RPL",
         "forest_plot_excess_rpl.png", "Excess Autocorrelation (RPL baseline)"),
        ("mean_excess_fhd", "se_excess_fhd", "excess_FHD",
         "forest_plot_excess_fhd.png", "Excess Autocorrelation (FHD baseline)"),
        ("mean_excess_sop", "se_excess_sop", "excess_SOP",
         "forest_plot_excess_sop.png", "Excess Autocorrelation (SOP baseline)"),
    ]:
        meta_res = meta_output.get(meta_key, {})
        generate_forest_plot(full_results, measure, se_key, meta_res, fname, title)

    # ===================================================================
    # COMPILE OUTPUT
    # ===================================================================
    output = compile_output(full_results, meta_output, lang_meta, reg_output, typology, scale)

    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, default=_json_default, indent=2))
    logger.info(f"Saved final output: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    elapsed = time.time() - START_TIME
    logger.info(f"Total compute time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()
