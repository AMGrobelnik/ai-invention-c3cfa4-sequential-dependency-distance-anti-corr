#!/usr/bin/env python3
"""Within-Treebank Distribution Diagnostics, Structural Subtype Analysis,
and Publication Figures via Independent RPL Computation.

Computes per-sentence excess-RPL autocorrelation for 30 strategically selected
treebanks (50 RPL permutations each), produces within-treebank distribution
diagnostics, structural subtype characterization, and 6 publication-quality figures.
"""

import gc
import json
import math
import os
import resource
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import scipy.stats
from loguru import logger

# ── Logging ──────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ── Hardware detection (container-aware) ─────────────────────────────────────
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
logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM (container)")

# ── Memory limits (use ~60% of container RAM = ~17 GB) ──────────────────────
RAM_BUDGET_BYTES = int(TOTAL_RAM_GB * 0.60 * 1e9)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET_BYTES * 3, RAM_BUDGET_BYTES * 3))
logger.info(f"RAM budget: {RAM_BUDGET_BYTES / 1e9:.1f} GB")

# ── Paths ────────────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).parent
DEP_DIR3 = Path("/ai-inventor/aii_pipeline/runs/comp-ling-dobrovoljc_ebw/3_invention_loop/iter_1/gen_art/data_id3_it1__opus")
DEP_DIR5 = Path("/ai-inventor/aii_pipeline/runs/comp-ling-dobrovoljc_ebw/3_invention_loop/iter_1/gen_art/data_id5_it1__opus")
DEP_DIR4 = Path("/ai-inventor/aii_pipeline/runs/comp-ling-dobrovoljc_ebw/3_invention_loop/iter_1/gen_art/data_id4_it1__opus")
FIGURES_DIR = WORKSPACE / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# ── Constants ────────────────────────────────────────────────────────────────
N_PERMUTATIONS = 50
RNG_SEED = 42
NUM_WORKERS = max(1, NUM_CPUS)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 0: LOAD TYPOLOGY + GRAMMAR PROFILES, SELECT 30 TREEBANKS
# ══════════════════════════════════════════════════════════════════════════════

def load_json(path: Path) -> dict:
    """Load JSON file with error handling."""
    logger.info(f"Loading {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
    return json.loads(path.read_text())


def load_typology() -> dict:
    """Load typology table and build lookup."""
    typology = load_json(DEP_DIR4 / "full_data_out.json")
    typo_lookup = {}
    for ex in typology["datasets"][0]["examples"]:
        typo_lookup[ex["metadata_treebank_id"]] = {
            "case_cat": ex.get("metadata_wals_case_category"),
            "word_order": ex.get("metadata_wals_word_order_label"),
            "family": ex.get("metadata_language_family"),
            "modality": ex.get("metadata_modality"),
            "ud_case_prop": ex.get("metadata_ud_case_proportion"),
            "language": ex.get("metadata_language_name"),
            "macroarea": ex.get("metadata_macroarea"),
        }
    logger.info(f"Loaded typology for {len(typo_lookup)} treebanks")
    return typo_lookup


def load_grammar_profiles() -> dict:
    """Load grammar profiles and build structural stats lookup."""
    grammar_profiles = load_json(DEP_DIR5 / "full_data_out.json")
    gp_lookup = {}
    for entry in grammar_profiles["datasets"][0]["examples"]:
        inp = json.loads(entry["input"]) if isinstance(entry["input"], str) else entry["input"]
        out = json.loads(entry["output"]) if isinstance(entry["output"], str) else entry["output"]
        tb_id = inp["treebank_id"]
        gp_lookup[tb_id] = out.get("structural_stats", {})
    logger.info(f"Loaded grammar profiles for {len(gp_lookup)} treebanks")
    return gp_lookup


def load_treebank_summaries() -> dict:
    """Load treebank summaries from data_id3 preview."""
    preview3 = load_json(DEP_DIR3 / "preview_data_out.json")
    summaries = preview3["metadata"]["treebank_summaries"]
    logger.info(f"Loaded summaries for {len(summaries)} treebanks")
    return summaries


def select_treebanks(qualifying: dict, typo_lookup: dict) -> tuple[list, dict]:
    """Select 30 treebanks using the three-group strategy.

    Group A (10): Largest by sentences_ge20 count
    Group B (10): Underrepresented families (typological diversity)
    Group C (10): Spoken/mixed modality + high-case-marking written

    Returns: (list of treebank_ids, dict of tb_id -> group_label)
    """
    selected = []
    group_map = {}

    # ── Group A: Largest by sentence count ───────────────────────────────
    sorted_by_size = sorted(
        qualifying.items(), key=lambda x: x[1]["sentences_ge20"], reverse=True
    )
    group_a = []
    for tb_id, info in sorted_by_size:
        if len(group_a) >= 10:
            break
        group_a.append(tb_id)
        group_map[tb_id] = "largest"
    selected.extend(group_a)
    families_in_a = {typo_lookup.get(tb, {}).get("family") for tb in group_a} - {None}
    logger.info(f"Group A (largest): {len(group_a)} treebanks, families: {families_in_a}")

    # ── Group B: Underrepresented families ───────────────────────────────
    # For each family not in Group A, pick the treebank with most sentences
    family_to_tbs = defaultdict(list)
    for tb_id in qualifying:
        if tb_id in selected:
            continue
        fam = typo_lookup.get(tb_id, {}).get("family")
        if fam and fam not in families_in_a:
            family_to_tbs[fam].append(tb_id)

    # Sort families by count (prefer rarer families)
    family_candidates = []
    for fam, tbs in sorted(family_to_tbs.items(), key=lambda x: len(x[1])):
        best = max(tbs, key=lambda t: qualifying[t]["sentences_ge20"])
        family_candidates.append((fam, best))

    group_b = []
    for fam, tb_id in family_candidates:
        if len(group_b) >= 10:
            break
        group_b.append(tb_id)
        group_map[tb_id] = "diverse"
    selected.extend(group_b)
    logger.info(f"Group B (diverse): {len(group_b)} treebanks")

    # ── Group C: Spoken + high-case written ──────────────────────────────
    group_c = []
    # First add spoken treebanks that qualify
    for tb_id in qualifying:
        if tb_id in selected:
            continue
        mod = typo_lookup.get(tb_id, {}).get("modality")
        if mod in ("spoken", "mixed"):
            group_c.append(tb_id)
            group_map[tb_id] = "spoken"
            if len(group_c) >= 10:
                break

    # Fill remainder with high-case-marking written treebanks
    if len(group_c) < 10:
        remaining = [
            (tb_id, typo_lookup.get(tb_id, {}).get("ud_case_prop", 0))
            for tb_id in qualifying
            if tb_id not in selected and tb_id not in [t for t in group_c]
        ]
        remaining.sort(key=lambda x: x[1], reverse=True)
        for tb_id, _ in remaining:
            if len(group_c) >= 10:
                break
            group_c.append(tb_id)
            group_map[tb_id] = "spoken"  # label as spoken group even if written fill
    selected.extend(group_c)
    logger.info(f"Group C (spoken/case): {len(group_c)} treebanks")

    # Final dedup (shouldn't be needed but safety)
    seen = set()
    deduped = []
    for tb in selected:
        if tb not in seen:
            seen.add(tb)
            deduped.append(tb)

    logger.info(f"Selected {len(deduped)} treebanks total")
    return deduped[:30], group_map


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: LOAD SENTENCE DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_sentences_for_treebanks(selected_set: set) -> dict:
    """Load sentences from 16 split files, filtering to selected treebanks + ge20."""
    sentences_by_treebank = defaultdict(list)
    for part_num in range(1, 17):
        filepath = DEP_DIR3 / f"data_out/full_data_out_{part_num}.json"
        if not filepath.exists():
            logger.warning(f"Split file {filepath.name} not found, skipping")
            continue
        logger.info(f"Loading split {part_num}/16...")
        part_data = load_json(filepath)
        for ds_entry in part_data["datasets"]:
            tb_id = ds_entry["dataset"]
            if tb_id not in selected_set:
                continue
            for ex in ds_entry["examples"]:
                if ex.get("metadata_length_bucket") != "ge20":
                    continue
                try:
                    inp = json.loads(ex["input"]) if isinstance(ex["input"], str) else ex["input"]
                    out = json.loads(ex["output"]) if isinstance(ex["output"], str) else ex["output"]
                    sentences_by_treebank[tb_id].append({
                        "head_array": inp["head_array"],
                        "deprel_array": inp["deprel_array"],
                        "dd_sequence": out["dd_sequence"],
                        "token_count": out["token_count"],
                        "sentence_id": ex.get("metadata_sentence_id", f"unk_{len(sentences_by_treebank[tb_id])}"),
                    })
                except (json.JSONDecodeError, KeyError) as e:
                    logger.debug(f"Skipped malformed example in {tb_id}: {e}")
        del part_data
        gc.collect()

    for tb_id, sents in sentences_by_treebank.items():
        logger.info(f"  {tb_id}: {len(sents)} sentences loaded")
    return dict(sentences_by_treebank)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: CORE COMPUTATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def compute_dd_nopunct(head_array: list, deprel_array: list) -> list:
    """Compute DD sequence excluding punctuation tokens (deprel='punct').
    Returns list of DD values for non-punct, non-root tokens in linear order."""
    dd_seq = []
    for i in range(len(head_array)):
        if deprel_array[i] == "punct":
            continue
        h = head_array[i]
        if h == 0:
            continue  # skip root
        position = i + 1  # 1-indexed
        dd = abs(position - h)
        dd_seq.append(dd)
    return dd_seq


def r1_standard(x: list) -> float:
    """Standard lag-1 autocorrelation for sequence x."""
    n = len(x)
    if n < 4:
        return float('nan')
    mean_x = sum(x) / n
    numerator = sum((x[i] - mean_x) * (x[i + 1] - mean_x) for i in range(n - 1))
    denominator = sum((x[i] - mean_x) ** 2 for i in range(n))
    if denominator == 0:
        return 0.0
    return numerator / denominator


def r1_prime(x: list) -> float:
    """Bias-corrected lag-1 autocorrelation (Huitema & McKean 2000).
    r1' = r1 + (1 - r1^2) / (n - 3)"""
    n = len(x)
    if n < 5:
        return float('nan')
    r1 = r1_standard(x)
    if math.isnan(r1):
        return float('nan')
    correction = (1 - r1 ** 2) / (n - 3)
    return r1 + correction


def build_tree(head_array: list) -> tuple:
    """Build children lookup from head_array (1-indexed).
    Returns (dict: parent -> [children], root_position)."""
    children = defaultdict(list)
    root = None
    for i, h in enumerate(head_array):
        pos = i + 1  # 1-indexed position
        if h == 0:
            root = pos
        else:
            children[h].append(pos)
    return dict(children), root


def random_projective_linearization(children: dict, root: int, rng) -> list:
    """Generate one random projective linearization of the tree.
    Returns: list of original token positions in new linear order."""
    def linearize_subtree(node):
        kids = children.get(node, [])
        if not kids:
            return [node]
        child_subtrees = [linearize_subtree(c) for c in kids]
        # Shuffle the child subtrees order
        perm = rng.permutation(len(child_subtrees)).tolist()
        shuffled = [child_subtrees[p] for p in perm]
        # Randomly insert head among shuffled subtrees
        insert_pos = rng.integers(0, len(shuffled) + 1)
        result = []
        for idx, subtree in enumerate(shuffled):
            if idx == insert_pos:
                result.append(node)
            result.extend(subtree)
        if insert_pos == len(shuffled):
            result.append(node)
        return result

    if root is None:
        return []
    return linearize_subtree(root)


def dd_from_linearization(new_order: list, head_array: list, deprel_array: list) -> list:
    """Compute DD sequence from a linearization, excluding punct tokens."""
    new_pos = {}
    for new_idx, orig_pos in enumerate(new_order):
        new_pos[orig_pos] = new_idx + 1

    dd_seq = []
    for new_idx, orig_pos in enumerate(new_order):
        orig_idx = orig_pos - 1
        if orig_idx < 0 or orig_idx >= len(deprel_array):
            continue
        if deprel_array[orig_idx] == "punct":
            continue
        h = head_array[orig_idx]
        if h == 0:
            continue
        if h not in new_pos:
            continue
        dd = abs(new_pos[orig_pos] - new_pos[h])
        dd_seq.append(dd)
    return dd_seq


def compute_tree_depth(children: dict, root: int) -> int:
    """DFS to find max depth from root."""
    if root is None:
        return 0
    stack = [(root, 0)]
    max_depth = 0
    while stack:
        node, depth = stack.pop()
        max_depth = max(max_depth, depth)
        for child in children.get(node, []):
            stack.append((child, depth + 1))
    return max_depth


def compute_branching_factor(children: dict) -> float:
    """Mean number of children for nodes that have children."""
    branch_counts = [len(kids) for kids in children.values() if kids]
    return float(np.mean(branch_counts)) if branch_counts else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: RPL COMPUTATION (PARALLELIZED)
# ══════════════════════════════════════════════════════════════════════════════

def process_one_sentence(args: tuple) -> dict | None:
    """Process a single sentence: compute observed r1' and N RPL r1' values.
    Takes a tuple (sent_data, perm_seed, n_perms) for pickle compatibility."""
    sent_data, perm_seed, n_perms = args
    try:
        head_array = sent_data["head_array"]
        deprel_array = sent_data["deprel_array"]

        # Observed DD sequence (punct excluded)
        dd_obs = compute_dd_nopunct(head_array, deprel_array)
        if len(dd_obs) < 5:
            return None

        observed_r1 = r1_prime(dd_obs)
        if math.isnan(observed_r1):
            return None

        # Build tree
        children, root = build_tree(head_array)
        if root is None:
            return None

        # Generate RPL permutations
        rng = np.random.default_rng(perm_seed)
        rpl_r1_values = []
        for _ in range(n_perms):
            new_order = random_projective_linearization(children, root, rng)
            dd_perm = dd_from_linearization(new_order, head_array, deprel_array)
            if len(dd_perm) >= 5:
                r1_val = r1_prime(dd_perm)
                if not math.isnan(r1_val):
                    rpl_r1_values.append(r1_val)

        if len(rpl_r1_values) < 10:
            return None

        rpl_mean = float(np.mean(rpl_r1_values))
        rpl_std = float(np.std(rpl_r1_values))
        excess_rpl = observed_r1 - rpl_mean

        # Structural properties for subtype analysis
        n_tokens = len(head_array)
        tree_depth = compute_tree_depth(children, root)
        branching_factor = compute_branching_factor(children)
        max_dd = max(dd_obs) if dd_obs else 0
        max_dd_position = dd_obs.index(max_dd) / len(dd_obs) if dd_obs else 0
        has_conj = any("conj" in dr for dr in deprel_array)
        has_relcl = any("acl:relcl" in dr for dr in deprel_array)
        n_punct = sum(1 for dr in deprel_array if dr == "punct")

        return {
            "sentence_id": sent_data["sentence_id"],
            "token_count": n_tokens,
            "dd_length_nopunct": len(dd_obs),
            "observed_r1": round(observed_r1, 6),
            "rpl_mean_r1": round(rpl_mean, 6),
            "rpl_std_r1": round(rpl_std, 6),
            "excess_rpl": round(excess_rpl, 6),
            "n_valid_perms": len(rpl_r1_values),
            "tree_depth": tree_depth,
            "branching_factor": round(branching_factor, 3),
            "max_dd": max_dd,
            "max_dd_relative_pos": round(max_dd_position, 3),
            "has_conj": has_conj,
            "has_relcl": has_relcl,
            "n_punct_tokens": n_punct,
        }
    except Exception as e:
        logger.debug(f"Error processing sentence: {e}")
        return None


def process_treebank(
    tb_id: str,
    sentences: list,
    n_perms: int = N_PERMUTATIONS,
    max_workers: int = NUM_WORKERS,
) -> list:
    """Process all sentences in a treebank with parallel workers."""
    base_seed = abs(hash(tb_id)) % (2 ** 31)
    args_list = [
        (sent, base_seed + idx, n_perms)
        for idx, sent in enumerate(sentences)
    ]

    results = []
    # Use ProcessPoolExecutor for CPU parallelism
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_one_sentence, a): i for i, a in enumerate(args_list)}
        for future in as_completed(futures):
            try:
                result = future.result(timeout=120)
                if result is not None:
                    results.append(result)
            except Exception as e:
                logger.debug(f"Future failed: {e}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: WITHIN-TREEBANK DISTRIBUTION DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_diagnostics(
    tb_id: str,
    results: list,
    typo_lookup: dict,
    gp_lookup: dict,
    group_map: dict,
    treebank_summaries: dict,
) -> dict | None:
    """Compute within-treebank distribution diagnostics."""
    if not results:
        return None

    excess_values = [r["excess_rpl"] for r in results]
    n = len(excess_values)

    prop_negative = sum(1 for e in excess_values if e < 0) / n
    median_excess = float(np.median(excess_values))
    q25, q75 = float(np.percentile(excess_values, 25)), float(np.percentile(excess_values, 75))
    iqr = q75 - q25
    mean_excess = float(np.mean(excess_values))
    std_excess = float(np.std(excess_values))
    skewness = float(scipy.stats.skew(excess_values))
    kurtosis_val = float(scipy.stats.kurtosis(excess_values))
    se_mean = std_excess / math.sqrt(n) if n > 0 else 0.0

    # Pervasiveness classification
    if prop_negative > 0.60 and abs(skewness) < 1.5:
        pervasiveness = "pervasive"
    elif prop_negative > 0.60:
        pervasiveness = "tail_driven"
    elif prop_negative >= 0.40:
        pervasiveness = "mixed"
    else:
        pervasiveness = "absent"

    # One-sample t-test
    t_stat, p_value = scipy.stats.ttest_1samp(excess_values, 0, alternative='less')
    t_stat = float(t_stat)
    p_value = float(p_value)

    typo = typo_lookup.get(tb_id, {})
    gp = gp_lookup.get(tb_id, {})

    return {
        "n_sentences": n,
        "prop_negative_excess": round(prop_negative, 4),
        "mean_excess_rpl": round(mean_excess, 6),
        "se_mean_excess": round(se_mean, 6),
        "median_excess_rpl": round(median_excess, 6),
        "q25_excess": round(q25, 6),
        "q75_excess": round(q75, 6),
        "iqr_excess": round(iqr, 6),
        "std_excess": round(std_excess, 6),
        "skewness": round(skewness, 4),
        "kurtosis": round(kurtosis_val, 4),
        "pervasiveness_class": pervasiveness,
        "t_statistic": round(t_stat, 4),
        "p_value": p_value,
        "sig_p001": p_value < 0.001,
        "sig_p01": p_value < 0.01,
        "modality": typo.get("modality", "unknown"),
        "word_order": typo.get("word_order"),
        "case_category": typo.get("case_cat"),
        "ud_case_proportion": typo.get("ud_case_prop"),
        "language_family": typo.get("family"),
        "language": typo.get("language"),
        "macroarea": typo.get("macroarea"),
        "mean_tree_depth": gp.get("mean_tree_depth"),
        "mean_branching_factor": gp.get("mean_branching_factor"),
        "proportion_projective": gp.get("proportion_projective"),
        "selection_group": group_map.get(tb_id, "unknown"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: STRUCTURAL SUBTYPE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def compute_subtype_analysis(tb_id: str, results: list) -> dict | None:
    """Compare structural properties of bottom-10% vs full population."""
    if len(results) < 20:
        return None

    sorted_results = sorted(results, key=lambda r: r["excess_rpl"])
    n = len(sorted_results)
    bottom_10pct = sorted_results[:max(n // 10, 2)]

    def struct_summary(group):
        return {
            "n": len(group),
            "mean_tree_depth": round(float(np.mean([r["tree_depth"] for r in group])), 4),
            "mean_branching_factor": round(float(np.mean([r["branching_factor"] for r in group])), 4),
            "mean_max_dd": round(float(np.mean([r["max_dd"] for r in group])), 4),
            "mean_max_dd_rel_pos": round(float(np.mean([r["max_dd_relative_pos"] for r in group])), 4),
            "prop_has_conj": round(float(np.mean([r["has_conj"] for r in group])), 4),
            "prop_has_relcl": round(float(np.mean([r["has_relcl"] for r in group])), 4),
            "mean_token_count": round(float(np.mean([r["token_count"] for r in group])), 4),
            "mean_dd_length_nopunct": round(float(np.mean([r["dd_length_nopunct"] for r in group])), 4),
        }

    full_stats = struct_summary(sorted_results)
    bottom_stats = struct_summary(bottom_10pct)

    comparisons = {}
    for prop in ["tree_depth", "branching_factor", "max_dd", "token_count"]:
        full_vals = [r[prop] for r in sorted_results]
        bottom_vals = [r[prop] for r in bottom_10pct]
        if len(set(bottom_vals)) > 1 and len(bottom_vals) >= 2:
            try:
                u_stat, p_val = scipy.stats.mannwhitneyu(
                    bottom_vals, full_vals, alternative='two-sided'
                )
                comparisons[prop] = {
                    "U": float(u_stat),
                    "p": float(p_val),
                    "significant_05": p_val < 0.05,
                }
            except ValueError:
                comparisons[prop] = {"U": None, "p": None, "significant_05": False}
        else:
            comparisons[prop] = {"U": None, "p": None, "significant_05": False}

    for prop in ["has_conj", "has_relcl"]:
        bottom_count = sum(1 for r in bottom_10pct if r[prop])
        full_count = sum(1 for r in sorted_results if r[prop])
        comparisons[prop] = {
            "bottom_10pct_prop": round(bottom_count / len(bottom_10pct), 4) if bottom_10pct else 0,
            "full_prop": round(full_count / len(sorted_results), 4) if sorted_results else 0,
        }

    return {
        "full_population": full_stats,
        "bottom_10pct": bottom_stats,
        "comparisons": comparisons,
    }


def compute_aggregate_subtype(subtype_analysis: dict) -> dict:
    """Aggregate subtype analysis across all treebanks."""
    aggregate = {}
    for prop in ["tree_depth", "branching_factor", "max_dd", "token_count"]:
        n_sig = sum(
            1 for tb in subtype_analysis.values()
            if tb["comparisons"].get(prop, {}).get("significant_05", False)
        )
        n_total = len(subtype_analysis)
        directions = []
        for tb in subtype_analysis.values():
            bkey = f"mean_{prop}" if f"mean_{prop}" in tb["bottom_10pct"] else prop
            fkey = f"mean_{prop}" if f"mean_{prop}" in tb["full_population"] else prop
            bv = tb["bottom_10pct"].get(bkey)
            fv = tb["full_population"].get(fkey)
            if bv is not None and fv is not None:
                directions.append("higher" if bv > fv else "lower")
        aggregate[prop] = {
            "n_significant": n_sig,
            "n_total": n_total,
            "dominant_direction": max(set(directions), key=directions.count) if directions else "unknown",
        }
    return aggregate


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: PUBLICATION FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def generate_figures(
    all_treebank_results: dict,
    treebank_diagnostics: dict,
    treebank_summaries: dict,
) -> list:
    """Generate 6 publication-quality matplotlib figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })

    figure_files = []

    # ── FIGURE 1: Histogram of per-sentence excess-RPL ───────────────────
    try:
        all_excess = []
        for results in all_treebank_results.values():
            all_excess.extend([r["excess_rpl"] for r in results])

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(all_excess, bins=100, density=True, alpha=0.7, color='steelblue', edgecolor='white')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Zero (no excess)')
        med_val = float(np.median(all_excess))
        ax.axvline(x=med_val, color='darkgreen', linestyle='-', linewidth=1.5,
                   label=f'Median = {med_val:.4f}')
        ax.set_xlabel('Excess lag-1 autocorrelation (observed \u2212 RPL baseline)')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of Per-Sentence Excess-RPL Autocorrelation\n'
                      '(30 treebanks, \u226520 tokens, punct excluded)')
        ax.legend()
        fig_path = FIGURES_DIR / "fig1_excess_rpl_histogram.png"
        fig.savefig(fig_path)
        plt.close()
        figure_files.append("fig1_excess_rpl_histogram.png")
        logger.info("Figure 1 saved")
    except Exception:
        logger.error("Figure 1 failed")

    # ── FIGURE 2: Boxplot by word order type ─────────────────────────────
    try:
        word_order_groups = defaultdict(list)
        for tb_id, diag in treebank_diagnostics.items():
            wo = diag.get("word_order") or "Unknown"
            word_order_groups[wo].append(diag["mean_excess_rpl"])

        fig, ax = plt.subplots(figsize=(8, 5))
        order = [k for k in ["SOV", "SVO", "VSO", "VOS", "OVS", "OSV", "Other", "Unknown"]
                 if k in word_order_groups]
        data_boxes = [word_order_groups[k] for k in order]
        labels = [f"{k}\n(n={len(word_order_groups[k])})" for k in order]
        colors_box = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0', '#795548', '#9E9E9E', '#BDBDBD']
        bp = ax.boxplot(data_boxes, tick_labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors_box[:len(order)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax.set_ylabel('Mean excess-RPL autocorrelation')
        ax.set_title('Treebank Mean Excess-RPL by Word Order Type')
        fig_path = FIGURES_DIR / "fig2_excess_by_wordorder.png"
        fig.savefig(fig_path)
        plt.close()
        figure_files.append("fig2_excess_by_wordorder.png")
        logger.info("Figure 2 saved")
    except Exception:
        logger.error("Figure 2 failed")

    # ── FIGURE 3: Scatter — case-marking vs excess ───────────────────────
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        case_vals_plot, excess_vals_plot = [], []
        for tb_id, diag in treebank_diagnostics.items():
            x_val = diag.get("ud_case_proportion")
            if x_val is None:
                x_val = 0
            y_val = diag["mean_excess_rpl"]
            color = 'red' if diag["modality"] == "spoken" else 'steelblue'
            marker = 's' if diag["modality"] == "spoken" else 'o'
            ax.scatter(x_val, y_val, c=color, marker=marker, s=50, alpha=0.7,
                       edgecolors='black', linewidths=0.5)
            if diag.get("ud_case_proportion") is not None:
                case_vals_plot.append(x_val)
                excess_vals_plot.append(y_val)

        if len(case_vals_plot) >= 3:
            rho_val, p_rho_val = scipy.stats.spearmanr(case_vals_plot, excess_vals_plot)
            ax.annotate(f'Spearman \u03c1 = {rho_val:.3f}, p = {p_rho_val:.4f}',
                        xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10,
                        ha='left', va='top')

        ax.set_xlabel('UD Case Feature Proportion')
        ax.set_ylabel('Mean Excess-RPL Autocorrelation')
        ax.set_title('Case-Marking Richness vs. Compensatory Anti-Correlation')
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue',
                   label='Written', markersize=8),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='red',
                   label='Spoken', markersize=8),
        ]
        ax.legend(handles=legend_elements, loc='lower left')
        fig_path = FIGURES_DIR / "fig3_case_vs_excess.png"
        fig.savefig(fig_path)
        plt.close()
        figure_files.append("fig3_case_vs_excess.png")
        logger.info("Figure 3 saved")
    except Exception:
        logger.error("Figure 3 failed")

    # ── FIGURE 4: Bar chart — proportion ge20 by modality ────────────────
    try:
        fig, ax = plt.subplots(figsize=(7, 5))
        spoken_props, written_props = [], []
        for tb_id, diag in treebank_diagnostics.items():
            summ = treebank_summaries.get(tb_id, {})
            total = summ.get("total_sentences", 1)
            ge20 = summ.get("sentences_ge20", 0)
            prop_val = ge20 / max(total, 1)
            if diag["modality"] == "spoken":
                spoken_props.append(prop_val)
            else:
                written_props.append(prop_val)
        positions = [0, 1]
        means = [
            float(np.mean(spoken_props)) if spoken_props else 0,
            float(np.mean(written_props)) if written_props else 0,
        ]
        stds = [
            float(np.std(spoken_props)) if spoken_props else 0,
            float(np.std(written_props)) if written_props else 0,
        ]
        ax.bar(positions, means, yerr=stds, capsize=5, color=['red', 'steelblue'], alpha=0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels([f'Spoken\n(n={len(spoken_props)})', f'Written\n(n={len(written_props)})'])
        ax.set_ylabel('Proportion of sentences \u226520 tokens')
        ax.set_title('Representativeness of Long Sentences by Modality')
        fig_path = FIGURES_DIR / "fig4_length_threshold_by_modality.png"
        fig.savefig(fig_path)
        plt.close()
        figure_files.append("fig4_length_threshold_by_modality.png")
        logger.info("Figure 4 saved")
    except Exception:
        logger.error("Figure 4 failed")

    # ── FIGURE 5: Forest-style plot ──────────────────────────────────────
    try:
        sorted_tbs = sorted(treebank_diagnostics.items(), key=lambda x: x[1]["mean_excess_rpl"])
        fig_height = max(8, len(sorted_tbs) * 0.35)
        fig, ax = plt.subplots(figsize=(10, fig_height))
        for i, (tb_id, diag) in enumerate(sorted_tbs):
            mean_val = diag["mean_excess_rpl"]
            se = diag["se_mean_excess"]
            ci_lo = mean_val - 1.96 * se
            ci_hi = mean_val + 1.96 * se
            color = 'red' if diag["modality"] == "spoken" else 'steelblue'
            ax.plot([ci_lo, ci_hi], [i, i], color=color, linewidth=1)
            ax.plot(mean_val, i, 'o', color=color, markersize=5)

        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
        # Pooled estimate (inverse-variance weighted)
        weights = []
        weighted_vals = []
        for _, d in sorted_tbs:
            if d["se_mean_excess"] > 0:
                w = 1 / d["se_mean_excess"] ** 2
                weights.append(w)
                weighted_vals.append(d["mean_excess_rpl"] * w)
        pooled = sum(weighted_vals) / sum(weights) if weights else 0
        ax.axvline(x=pooled, color='darkgreen', linestyle='-', linewidth=2, alpha=0.7,
                   label=f'Pooled estimate = {pooled:.4f}')
        ax.set_yticks(list(range(len(sorted_tbs))))
        ax.set_yticklabels(
            [f"{tb_id} ({diag.get('language', '?')})" for tb_id, diag in sorted_tbs],
            fontsize=7,
        )
        ax.set_xlabel('Mean Excess-RPL Autocorrelation (95% CI)')
        ax.set_title('Forest Plot: Per-Treebank Excess-RPL Estimates')
        ax.legend(loc='lower right')
        fig_path = FIGURES_DIR / "fig5_forest_plot.png"
        fig.savefig(fig_path)
        plt.close()
        figure_files.append("fig5_forest_plot.png")
        logger.info("Figure 5 saved")
    except Exception:
        logger.error("Figure 5 failed")

    # ── FIGURE 6: Pervasiveness barplot ──────────────────────────────────
    try:
        sorted_tbs_perv = sorted(
            treebank_diagnostics.items(), key=lambda x: x[1]["prop_negative_excess"]
        )
        fig_height = max(6, len(sorted_tbs_perv) * 0.3)
        fig, ax = plt.subplots(figsize=(10, fig_height))
        props_bar = [d["prop_negative_excess"] for _, d in sorted_tbs_perv]
        colors_bar = [
            'darkred' if p > 0.6 else 'orange' if p > 0.5 else 'gray'
            for p in props_bar
        ]
        ax.barh(list(range(len(sorted_tbs_perv))), props_bar, color=colors_bar, alpha=0.8)
        ax.axvline(x=0.5, color='gray', linestyle='--')
        ax.axvline(x=0.6, color='red', linestyle='--', alpha=0.5, label='60% threshold')
        ax.set_yticks(list(range(len(sorted_tbs_perv))))
        ax.set_yticklabels([f"{tb_id}" for tb_id, _ in sorted_tbs_perv], fontsize=7)
        ax.set_xlabel('Proportion of sentences with negative excess-RPL')
        ax.set_title('Pervasiveness of Compensatory Anti-Correlation per Treebank')
        ax.legend()
        fig_path = FIGURES_DIR / "fig6_pervasiveness_barplot.png"
        fig.savefig(fig_path)
        plt.close()
        figure_files.append("fig6_pervasiveness_barplot.png")
        logger.info("Figure 6 saved")
    except Exception:
        logger.error("Figure 6 failed")

    return figure_files


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7: ASSEMBLE method_out.json
# ══════════════════════════════════════════════════════════════════════════════

def assemble_output(
    treebank_diagnostics: dict,
    subtype_analysis: dict,
    aggregate_subtype: dict,
    all_treebank_results: dict,
    selected_30: list,
    group_map: dict,
    figure_files: list,
    pooled: float,
    rho: float,
    p_rho: float,
) -> dict:
    """Assemble method_out.json following exp_gen_sol_out schema."""
    output = {
        "datasets": [
            {
                "dataset": "treebank_distribution_diagnostics",
                "examples": [
                    {
                        "input": json.dumps({"treebank_id": tb_id}),
                        "output": json.dumps(diag_data, default=str),
                        "metadata_treebank_id": tb_id,
                        "metadata_language": diag_data.get("language", "unknown"),
                        "metadata_modality": diag_data.get("modality", "unknown"),
                        "metadata_pervasiveness": diag_data.get("pervasiveness_class", "unknown"),
                        "predict_mean_excess_rpl": str(diag_data["mean_excess_rpl"]),
                        "predict_prop_negative": str(diag_data["prop_negative_excess"]),
                        "predict_significant": str(diag_data["sig_p001"]),
                    }
                    for tb_id, diag_data in treebank_diagnostics.items()
                ],
            },
            {
                "dataset": "structural_subtype_analysis",
                "examples": [
                    {
                        "input": json.dumps({"treebank_id": tb_id}),
                        "output": json.dumps(sub_data, default=str),
                        "metadata_treebank_id": tb_id,
                        "predict_bottom10pct_deeper": str(
                            sub_data["bottom_10pct"].get("mean_tree_depth", 0)
                            > sub_data["full_population"].get("mean_tree_depth", 0)
                        ),
                    }
                    for tb_id, sub_data in subtype_analysis.items()
                ],
            },
            {
                "dataset": "aggregate_summary",
                "examples": [
                    {
                        "input": json.dumps({"analysis": "pooled_30_treebanks"}),
                        "output": json.dumps(
                            {
                                "n_treebanks": len(treebank_diagnostics),
                                "n_pervasive": sum(
                                    1 for d in treebank_diagnostics.values()
                                    if d["pervasiveness_class"] == "pervasive"
                                ),
                                "n_significant_p001": sum(
                                    1 for d in treebank_diagnostics.values() if d["sig_p001"]
                                ),
                                "n_significant_p01": sum(
                                    1 for d in treebank_diagnostics.values() if d["sig_p01"]
                                ),
                                "pooled_estimate": pooled,
                                "median_mean_excess": float(
                                    np.median([d["mean_excess_rpl"] for d in treebank_diagnostics.values()])
                                ),
                                "prop_treebanks_majority_negative": (
                                    sum(
                                        1 for d in treebank_diagnostics.values()
                                        if d["prop_negative_excess"] > 0.5
                                    )
                                    / max(len(treebank_diagnostics), 1)
                                ),
                                "aggregate_subtype_enrichment": aggregate_subtype,
                                "spearman_case_vs_excess": {"rho": rho, "p": p_rho},
                                "figures_generated": figure_files,
                                "selection_groups": {
                                    "largest": [tb for tb in selected_30 if group_map.get(tb) == "largest"],
                                    "diverse": [tb for tb in selected_30 if group_map.get(tb) == "diverse"],
                                    "spoken": [tb for tb in selected_30 if group_map.get(tb) == "spoken"],
                                },
                            },
                            default=str,
                        ),
                        "metadata_analysis_type": "aggregate",
                        "predict_hypothesis_supported": str(pooled < -0.05),
                    }
                ],
            },
            {
                "dataset": "per_sentence_details",
                "examples": [
                    {
                        "input": json.dumps({"treebank_id": tb_id, "sentence_id": r["sentence_id"]}),
                        "output": json.dumps(r, default=str),
                        "metadata_treebank_id": tb_id,
                        "metadata_extremity_rank": str(rank),
                        "predict_excess_rpl": str(r["excess_rpl"]),
                    }
                    for tb_id, results in all_treebank_results.items()
                    for rank, r in enumerate(
                        sorted(results, key=lambda x: x["excess_rpl"])[:5]
                    )
                ],
            },
        ]
    }
    return output


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

@logger.catch
def main():
    import time
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("STARTING: Within-Treebank Distribution Diagnostics")
    logger.info(f"Workspace: {WORKSPACE}")
    logger.info(f"Workers: {NUM_WORKERS}, Permutations: {N_PERMUTATIONS}")
    logger.info("=" * 60)

    # ── Step 0: Load metadata ────────────────────────────────────────────
    typo_lookup = load_typology()
    gp_lookup = load_grammar_profiles()
    treebank_summaries = load_treebank_summaries()

    # Filter qualifying treebanks
    qualifying = {
        tb_id: info
        for tb_id, info in treebank_summaries.items()
        if info.get("qualifies_primary", False)
    }
    logger.info(f"Qualifying treebanks: {len(qualifying)}")

    # Select 30 treebanks
    selected_30, group_map = select_treebanks(qualifying, typo_lookup)
    for tb_id in selected_30:
        typo = typo_lookup.get(tb_id, {})
        summ = qualifying.get(tb_id, {})
        logger.info(
            f"  [{group_map.get(tb_id, '?'):>7}] {tb_id:30s} "
            f"lang={typo.get('language', '?'):15s} "
            f"fam={str(typo.get('family', '?')):20s} "
            f"mod={typo.get('modality', '?'):8s} "
            f"sents={summ.get('sentences_ge20', 0)}"
        )

    gc.collect()

    # ── Step 1: Load sentence data ───────────────────────────────────────
    selected_set = set(selected_30)
    sentences_by_treebank = load_sentences_for_treebanks(selected_set)

    total_sents = sum(len(v) for v in sentences_by_treebank.values())
    logger.info(f"Total sentences loaded: {total_sents} across {len(sentences_by_treebank)} treebanks")
    gc.collect()

    # ── Step 3: RPL computation ──────────────────────────────────────────
    all_treebank_results = {}
    checkpoint_path = WORKSPACE / "checkpoint_results.json"

    for idx, tb_id in enumerate(selected_30):
        if tb_id not in sentences_by_treebank:
            logger.warning(f"No sentences for {tb_id}, skipping")
            continue

        sentences = sentences_by_treebank[tb_id]
        logger.info(f"[{idx + 1}/{len(selected_30)}] Processing {tb_id}: {len(sentences)} sentences")
        t0 = time.time()

        results = process_treebank(tb_id, sentences, N_PERMUTATIONS, NUM_WORKERS)
        elapsed = time.time() - t0

        all_treebank_results[tb_id] = results
        logger.info(
            f"  {tb_id}: {len(results)}/{len(sentences)} valid "
            f"({elapsed:.1f}s, {elapsed / max(len(sentences), 1) * 1000:.1f}ms/sent)"
        )

        # Checkpoint every 5 treebanks
        if (idx + 1) % 5 == 0:
            logger.info(f"Checkpoint at {idx + 1} treebanks ({time.time() - start_time:.0f}s elapsed)")

    logger.info(f"RPL computation complete: {time.time() - start_time:.0f}s total")
    gc.collect()

    # Free sentence data to save memory
    del sentences_by_treebank
    gc.collect()

    # ── Step 4: Distribution diagnostics ─────────────────────────────────
    treebank_diagnostics = {}
    for tb_id, results in all_treebank_results.items():
        diag = compute_diagnostics(
            tb_id, results, typo_lookup, gp_lookup, group_map, treebank_summaries
        )
        if diag is not None:
            treebank_diagnostics[tb_id] = diag
            logger.info(
                f"  {tb_id}: pervasiveness={diag['pervasiveness_class']}, "
                f"prop_neg={diag['prop_negative_excess']:.3f}, "
                f"mean_excess={diag['mean_excess_rpl']:.4f}, "
                f"p={diag['p_value']:.2e}"
            )

    logger.info(f"Diagnostics computed for {len(treebank_diagnostics)} treebanks")

    # ── Step 5: Structural subtype analysis ──────────────────────────────
    subtype_analysis = {}
    for tb_id, results in all_treebank_results.items():
        sub = compute_subtype_analysis(tb_id, results)
        if sub is not None:
            subtype_analysis[tb_id] = sub

    aggregate_subtype = compute_aggregate_subtype(subtype_analysis)
    logger.info(f"Subtype analysis for {len(subtype_analysis)} treebanks")
    for prop, agg in aggregate_subtype.items():
        logger.info(f"  {prop}: {agg['n_significant']}/{agg['n_total']} significant, direction={agg['dominant_direction']}")

    # ── Compute pooled estimate and Spearman correlation ─────────────────
    weights = []
    weighted_vals = []
    for _, d in treebank_diagnostics.items():
        if d["se_mean_excess"] > 0:
            w = 1 / d["se_mean_excess"] ** 2
            weights.append(w)
            weighted_vals.append(d["mean_excess_rpl"] * w)
    pooled = sum(weighted_vals) / sum(weights) if weights else 0.0

    case_vals = [
        d["ud_case_proportion"]
        for d in treebank_diagnostics.values()
        if d["ud_case_proportion"] is not None
    ]
    excess_vals = [
        d["mean_excess_rpl"]
        for d in treebank_diagnostics.values()
        if d["ud_case_proportion"] is not None
    ]
    if len(case_vals) >= 3:
        rho, p_rho = scipy.stats.spearmanr(case_vals, excess_vals)
        rho, p_rho = float(rho), float(p_rho)
    else:
        rho, p_rho = 0.0, 1.0

    logger.info(f"Pooled estimate: {pooled:.6f}")
    logger.info(f"Spearman case vs excess: rho={rho:.4f}, p={p_rho:.4f}")

    # ── Step 6: Publication figures ──────────────────────────────────────
    figure_files = generate_figures(all_treebank_results, treebank_diagnostics, treebank_summaries)
    logger.info(f"Generated {len(figure_files)} figures")

    # ── Step 7: Assemble output ──────────────────────────────────────────
    output = assemble_output(
        treebank_diagnostics=treebank_diagnostics,
        subtype_analysis=subtype_analysis,
        aggregate_subtype=aggregate_subtype,
        all_treebank_results=all_treebank_results,
        selected_30=selected_30,
        group_map=group_map,
        figure_files=figure_files,
        pooled=pooled,
        rho=rho,
        p_rho=p_rho,
    )

    # Save output
    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Saved method_out.json ({out_path.stat().st_size / 1e6:.1f} MB)")

    total_elapsed = time.time() - start_time
    logger.info(f"DONE in {total_elapsed:.0f}s ({total_elapsed / 60:.1f}m)")
    logger.info(f"Treebanks with pervasive effect: "
                f"{sum(1 for d in treebank_diagnostics.values() if d['pervasiveness_class'] == 'pervasive')}"
                f"/{len(treebank_diagnostics)}")
    logger.info(f"Treebanks significant at p<0.001: "
                f"{sum(1 for d in treebank_diagnostics.values() if d['sig_p001'])}"
                f"/{len(treebank_diagnostics)}")


if __name__ == "__main__":
    main()
