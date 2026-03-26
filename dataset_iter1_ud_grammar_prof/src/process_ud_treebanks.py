#!/usr/bin/env python3
"""Extract per-treebank grammar profiles from all Universal Dependencies treebanks.

Produces a single dataset with one row per treebank containing:
1. Head-direction profiles per deprel
2. Sibling-ordering templates
3. Structural statistics (tree depth, branching, projectivity, etc.)
4. Per-deprel frequency counts
5. Typological metadata from WALS 49A and UD-derived morphological features
"""

from loguru import logger
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import sys
import os
import math
import time
import resource
import csv
import gc
import traceback

# ── Logging ─────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ── Hardware detection ──────────────────────────────────────────────────
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
RAM_BUDGET_BYTES = int(TOTAL_RAM_GB * 0.7 * 1e9)  # Use 70% of available RAM

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM")
logger.info(f"RAM budget: {RAM_BUDGET_BYTES / 1e9:.1f} GB")

# Set memory limit
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET_BYTES * 3, RAM_BUDGET_BYTES * 3))

# ── Constants ───────────────────────────────────────────────────────────
WORKSPACE = Path("/ai-inventor/aii_pipeline/runs/comp-ling-dobrovoljc_ebw/3_invention_loop/iter_1/gen_art/data_id5_it1__opus")
DATASETS_DIR = WORKSPACE / "temp" / "datasets"
MIN_DEPREL_COUNT = 10
MIN_SIBLING_COUNT = 5

# UPOS mapping derived from dataset features metadata
UPOS_MAP = ['NOUN', 'PUNCT', 'ADP', 'NUM', 'SYM', 'SCONJ', 'ADJ', 'PART',
            'DET', 'CCONJ', 'PROPN', 'PRON', 'X', '_', 'ADV', 'INTJ', 'VERB', 'AUX']

# Known spoken treebank IDs (from Dobrovoljc 2022)
SPOKEN_TREEBANKS = {
    "sl_sst", "fr_rhapsodie", "fr_parisstories", "en_childes", "en_eslspok",
    "it_kiparlaforest", "pcm_nsc", "es_coser", "qfn_fame", "kpv_ikdp",
    "bej_autogramm"
}
# Sign language treebanks (exclude from spoken, mark separately)
SIGN_TREEBANKS = {"ssp_lse", "swl_sslc"}
# Substrings to heuristically detect spoken treebanks
SPOKEN_SUBSTRINGS = ["spok", "sst", "spoken", "oral", "rhapsodie",
                     "parisstories", "childes", "kiparla", "coser", "nsc", "fame"]

# ── WALS & Typological Metadata ────────────────────────────────────────

def load_wals_data() -> tuple[dict, dict, dict]:
    """Load WALS 49A case-marking data and language metadata."""
    logger.info("Loading WALS data...")

    # Load codes: maps code_id -> description
    codes_map = {}
    codes_path = DATASETS_DIR / "wals_codes.csv"
    with open(codes_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Parameter_ID") == "49A":
                codes_map[row["ID"]] = row["Name"]

    # Load languages: maps WALS ID -> (name, iso639p3, family, genus, ...)
    lang_map = {}  # wals_id -> dict
    iso_to_wals = {}  # iso639p3 -> wals_id
    langs_path = DATASETS_DIR / "wals_languages.csv"
    with open(langs_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            wals_id = row["ID"]
            lang_map[wals_id] = {
                "name": row.get("Name", ""),
                "family": row.get("Family", row.get("Macroarea", "")),
                "iso639p3": row.get("ISO639P3code", ""),
            }
            iso_code = row.get("ISO639P3code", "").strip()
            if iso_code:
                iso_to_wals[iso_code] = wals_id

    # Load values: filter to 49A, maps wals_lang_id -> case_value
    wals_49a = {}  # wals_lang_id -> value string (e.g., "1", "2", ...)
    values_path = DATASETS_DIR / "wals_values.csv"
    with open(values_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Parameter_ID") == "49A":
                wals_49a[row["Language_ID"]] = row["Value"]

    logger.info(f"WALS: {len(wals_49a)} languages with 49A data, "
                f"{len(iso_to_wals)} ISO mappings, {len(codes_map)} codes")
    return wals_49a, iso_to_wals, lang_map, codes_map


def build_ud_to_iso_map() -> dict[str, str]:
    """Map UD language prefixes to ISO 639-3 codes.

    Uses a manually curated mapping for common UD prefixes.
    """
    # Common UD prefix -> ISO 639-3 mappings
    # Many UD prefixes are already ISO 639-1 (2-letter) or ISO 639-3 (3-letter)
    ud_to_iso = {
        # 2-letter ISO 639-1 codes -> ISO 639-3
        "af": "afr", "am": "amh", "ar": "ara", "az": "aze", "be": "bel",
        "bg": "bul", "bn": "ben", "br": "bre", "ca": "cat", "cs": "ces",
        "cu": "chu", "cy": "cym", "da": "dan", "de": "deu", "el": "ell",
        "en": "eng", "eo": "epo", "es": "spa", "et": "est", "eu": "eus",
        "fa": "fas", "fi": "fin", "fo": "fao", "fr": "fra", "ga": "gle",
        "gd": "gla", "gl": "glg", "gu": "guj", "ha": "hau", "he": "heb",
        "hi": "hin", "hr": "hrv", "ht": "hat", "hu": "hun", "hy": "hye",
        "id": "ind", "is": "isl", "it": "ita", "ja": "jpn", "jv": "jav",
        "ka": "kat", "kk": "kaz", "ko": "kor", "ky": "kir", "la": "lat",
        "lb": "ltz", "lt": "lit", "lv": "lav", "mk": "mkd", "ml": "mal",
        "mn": "mon", "mr": "mar", "mt": "mlt", "my": "mya", "nl": "nld",
        "no": "nor", "or": "ori", "pl": "pol", "ps": "pus", "pt": "por",
        "ro": "ron", "ru": "rus", "sa": "san", "sd": "snd", "si": "sin",
        "sk": "slk", "sl": "slv", "sq": "sqi", "sr": "srp", "sv": "swe",
        "sw": "swa", "ta": "tam", "te": "tel", "th": "tha", "tl": "tgl",
        "tn": "tsn", "tr": "tur", "tt": "tat", "ug": "uig", "uk": "ukr",
        "ur": "urd", "uz": "uzb", "vi": "vie", "wo": "wol", "yi": "yid",
        "yo": "yor", "zh": "zho",
        # 3-letter codes already ISO 639-3
        "abq": "abq", "akk": "akk", "aqz": "aqz", "gsw": "gsw", "grc": "grc",
        "hbo": "hbo", "got": "got", "hit": "hit", "lzh": "lzh", "cop": "cop",
        "orv": "orv", "ang": "ang", "fro": "fro", "sga": "sga", "pro": "pro",
        "otk": "otk", "ota": "ota", "gub": "gub", "gn": "grn", "gwi": "gwi",
        "pcm": "pcm", "myv": "myv", "mdf": "mdf", "sme": "sme", "kmr": "kmr",
        "ltg": "ltg", "olo": "olo", "nds": "nds", "yue": "yue", "wuu": "wuu",
        "cpg": "cpg", "nap": "nap", "lij": "lij", "scn": "scn", "oc": "oci",
        "bar": "bar", "hsb": "hsb", "krl": "krl", "vep": "vep", "sms": "sms",
        "kpv": "kpv", "koi": "koi", "bxr": "bxr", "sah": "sah", "sjo": "sjo",
        "yrk": "yrk", "ess": "ess", "ckt": "ckt", "ceb": "ceb", "ckb": "ckb",
        "apu": "apu", "bor": "bor", "myu": "myu", "xav": "xav", "gun": "gun",
        "eme": "eme", "tpn": "tpn", "urb": "urb", "arr": "arr", "mpu": "mpu",
        "pay": "pay", "arh": "arh", "sab": "sab", "ctn": "ctn", "yrl": "yrl",
        "bho": "bho", "xnr": "xnr", "xcl": "xcl", "hyw": "hyw", "ajp": "ajp",
        "sdh": "sdh", "qfn": "qfn", "qpm": "qpm", "quc": "quc", "qaf": "qaf",
        "qte": "qte", "qti": "qti", "qtd": "qtd", "frm": "frm", "gya": "gya",
        "nmf": "nmf", "nyq": "nyq", "naq": "naq", "kfm": "kfm", "soj": "soj",
        "pad": "pad", "jaa": "jaa", "aln": "aln", "aii": "aii", "ab": "abk",
        "bm": "bam", "gv": "glv", "nhi": "nhi", "azz": "azz", "wbp": "wbp",
        "bej": "bej", "xpg": "xpg", "xum": "xum", "egy": "egy",
        "say": "say", "gya": "gya",
    }
    return ud_to_iso


def classify_modality(treebank_id: str) -> str:
    """Classify a treebank as spoken/written/sign/unknown."""
    if treebank_id in SIGN_TREEBANKS:
        return "sign"
    if treebank_id in SPOKEN_TREEBANKS:
        return "spoken"
    for substr in SPOKEN_SUBSTRINGS:
        if substr in treebank_id.lower():
            return "spoken"
    return "written"


# ── Per-Treebank Processing ────────────────────────────────────────────

def compute_tree_depth(heads: list[int], n: int) -> int:
    """Compute max tree depth from root via iterative parent-following."""
    max_depth = 0
    for i in range(1, n + 1):
        depth = 0
        cur = i
        visited = set()
        while cur != 0 and cur not in visited:
            visited.add(cur)
            depth += 1
            if 1 <= cur <= n:
                cur = heads[cur - 1]
            else:
                break
        max_depth = max(max_depth, depth)
    return max_depth


def check_projectivity(heads: list[int], n: int) -> bool:
    """Check if a dependency tree is projective (no crossing arcs)."""
    for i in range(1, n + 1):
        h = heads[i - 1]
        if h == 0:
            continue
        lo, hi = min(i, h), max(i, h)
        for j in range(lo + 1, hi):
            hj = heads[j - 1]
            if hj < lo or hj > hi:
                return False
    return True


def process_single_treebank(args: tuple) -> dict | None:
    """Process a single treebank and return its grammar profile row.

    This function runs in a subprocess, so we import datasets here.
    """
    treebank_id, max_sentences = args

    # Import inside subprocess
    from datasets import load_dataset
    import time as _time

    start = _time.time()

    try:
        ds = load_dataset("commul/universal_dependencies", treebank_id)
    except Exception as e:
        return {"error": f"Failed to load {treebank_id}: {e}", "treebank_id": treebank_id}

    # Concatenate all splits
    all_sentences = []
    for split_name in ds:
        all_sentences.extend(ds[split_name])

    if len(all_sentences) == 0:
        return {"error": f"Empty treebank: {treebank_id}", "treebank_id": treebank_id}

    # Apply max_sentences limit if set
    if max_sentences and max_sentences > 0 and len(all_sentences) > max_sentences:
        all_sentences = all_sentences[:max_sentences]

    total_sentences = len(all_sentences)
    total_tokens = 0

    # Accumulators
    head_final_count = Counter()
    head_initial_count = Counter()
    deprel_freq = Counter()
    sibling_templates = defaultdict(lambda: Counter())
    tree_depths = []
    branching_factors = []
    projective_count = 0
    sentence_lengths = []
    tokens_with_case = 0
    tokens_non_punct = 0

    for sent in all_sentences:
        tokens = sent["tokens"]
        upos_ints = sent["upos"]
        heads_str = sent["head"]
        deprels = sent["deprel"]
        feats_list = sent["feats"]
        n = len(tokens)
        total_tokens += n
        sentence_lengths.append(n)

        # Convert heads to int
        try:
            heads = [int(h) for h in heads_str]
        except (ValueError, TypeError):
            continue  # skip malformed sentences

        # Map upos ints to strings
        upos_strs = []
        for u in upos_ints:
            if 0 <= u < len(UPOS_MAP):
                upos_strs.append(UPOS_MAP[u])
            else:
                upos_strs.append("X")

        # 2b. Head-direction profile
        for i in range(n):
            h = heads[i]
            if h == 0:
                continue  # root token
            dep = deprels[i]
            deprel_freq[dep] += 1
            token_pos = i + 1  # 1-indexed
            if token_pos < h:
                head_final_count[dep] += 1
            elif token_pos > h:
                head_initial_count[dep] += 1

        # 2c. Sibling-ordering templates
        children = defaultdict(list)
        for i in range(n):
            h = heads[i]
            if h > 0:
                children[h].append((i + 1, deprels[i], upos_strs[i]))

        for head_pos, child_list in children.items():
            if len(child_list) < 2:
                continue
            head_upos = upos_strs[head_pos - 1] if 1 <= head_pos <= n else "X"
            child_list_sorted = sorted(child_list, key=lambda x: x[0])
            deprel_set = frozenset(c[1] for c in child_list_sorted)
            # Build ordering template with HEAD position
            template = []
            for c in child_list_sorted:
                if c[0] < head_pos:
                    template.append(c[1])
            template.append("HEAD")
            for c in child_list_sorted:
                if c[0] > head_pos:
                    template.append(c[1])
            key = (head_upos, deprel_set)
            sibling_templates[key][tuple(template)] += 1

        # 2d. Structural statistics
        depth = compute_tree_depth(heads, n)
        tree_depths.append(depth)

        # Branching factor (dependents per non-leaf node)
        non_leaf_branches = []
        for h_pos, child_list in children.items():
            if len(child_list) > 0:
                non_leaf_branches.append(len(child_list))
        if non_leaf_branches:
            branching_factors.append(sum(non_leaf_branches) / len(non_leaf_branches))

        # Projectivity
        if check_projectivity(heads, n):
            projective_count += 1

        # 2e. Case feature proportion
        for i in range(n):
            if upos_strs[i] != "PUNCT":
                tokens_non_punct += 1
                feat = feats_list[i]
                if feat and "Case=" in feat:
                    tokens_with_case += 1

    # Also count root deprels
    for sent in all_sentences:
        for i, h in enumerate(sent["head"]):
            if h == "0":
                deprel_freq[sent["deprel"][i]] += 1

    # ── Aggregate results ───────────────────────────────────────────
    import numpy as np

    # Head-direction profile
    head_direction_profile = {}
    for dep in set(list(head_final_count.keys()) + list(head_initial_count.keys())):
        hf = head_final_count[dep]
        hi = head_initial_count[dep]
        total = hf + hi
        if total >= MIN_DEPREL_COUNT:
            head_direction_profile[dep] = {
                "prop_head_final": round(hf / total, 4),
                "count_head_final": hf,
                "count_head_initial": hi,
            }

    # Sibling-ordering templates
    sibling_order_list = []
    for (head_upos, deprel_set), template_counts in sibling_templates.items():
        total_count = sum(template_counts.values())
        if total_count < MIN_SIBLING_COUNT:
            continue
        majority_template = template_counts.most_common(1)[0]
        sibling_order_list.append({
            "head_upos": head_upos,
            "deprel_set": sorted(deprel_set),
            "majority_order": list(majority_template[0]),
            "frequency": total_count,
            "coverage": round(majority_template[1] / total_count, 4),
            "n_distinct_orders": len(template_counts),
        })
    # Sort by frequency descending, keep top 200 to avoid enormous output
    sibling_order_list.sort(key=lambda x: x["frequency"], reverse=True)
    sibling_order_list = sibling_order_list[:200]

    # Structural stats
    depths_arr = np.array(tree_depths) if tree_depths else np.array([0])
    lengths_arr = np.array(sentence_lengths) if sentence_lengths else np.array([0])

    structural_stats = {
        "mean_tree_depth": round(float(depths_arr.mean()), 2),
        "std_tree_depth": round(float(depths_arr.std()), 2),
        "mean_branching_factor": round(float(np.mean(branching_factors)) if branching_factors else 0.0, 2),
        "proportion_projective": round(projective_count / total_sentences, 4) if total_sentences > 0 else 0.0,
        "total_sentence_count": total_sentences,
        "sentence_count_ge15": int(np.sum(lengths_arr >= 15)),
        "sentence_count_ge20": int(np.sum(lengths_arr >= 20)),
        "mean_sentence_length": round(float(lengths_arr.mean()), 2),
        "std_sentence_length": round(float(lengths_arr.std()), 2),
    }

    # Case feature proportion
    case_proportion = round(tokens_with_case / tokens_non_punct, 4) if tokens_non_punct > 0 else 0.0

    elapsed = _time.time() - start

    return {
        "treebank_id": treebank_id,
        "total_tokens": total_tokens,
        "elapsed_seconds": round(elapsed, 2),
        "head_direction_profile": head_direction_profile,
        "sibling_order_templates": sibling_order_list,
        "structural_stats": structural_stats,
        "deprel_frequency_counts": dict(deprel_freq),
        "case_feature_proportion": case_proportion,
    }


def assemble_output_row(
    result: dict,
    wals_49a: dict,
    iso_to_wals: dict,
    lang_map: dict,
    codes_map: dict,
    ud_to_iso: dict,
) -> dict:
    """Assemble the final output row for a treebank."""
    treebank_id = result["treebank_id"]
    lang_prefix = treebank_id.split("_")[0]
    iso639_3 = ud_to_iso.get(lang_prefix, lang_prefix)

    # WALS 49A lookup
    wals_lang_id = iso_to_wals.get(iso639_3, "")
    wals_value = None
    wals_label = None
    lang_family = ""
    lang_name = ""

    if wals_lang_id:
        wals_value_str = wals_49a.get(wals_lang_id)
        if wals_value_str:
            try:
                wals_value = int(wals_value_str)
            except (ValueError, TypeError):
                wals_value = None
            wals_label = codes_map.get(f"49A-{wals_value_str}", None)
        lang_info = lang_map.get(wals_lang_id, {})
        lang_family = lang_info.get("family", "")
        lang_name = lang_info.get("name", "")

    # If no WALS match, try to get family from lang_map by scanning iso codes
    if not lang_family:
        for wid, linfo in lang_map.items():
            if linfo.get("iso639p3") == iso639_3:
                lang_family = linfo.get("family", "")
                lang_name = linfo.get("name", "")
                break

    modality = classify_modality(treebank_id)

    return {
        "input": {
            "treebank_id": treebank_id,
            "language_prefix": lang_prefix,
            "iso639_3": iso639_3,
        },
        "output": {
            "head_direction_profile": result["head_direction_profile"],
            "sibling_order_templates": result["sibling_order_templates"],
            "structural_stats": result["structural_stats"],
            "deprel_frequency_counts": result["deprel_frequency_counts"],
            "case_feature_proportion": result["case_feature_proportion"],
        },
        "metadata_fold": "all",
        "metadata_modality": modality,
        "metadata_wals_49a_value": wals_value,
        "metadata_wals_49a_label": wals_label,
        "metadata_language_family": lang_family,
        "metadata_language_name": lang_name,
    }


@logger.catch
def main(
    treebank_ids: list[str] | None = None,
    max_sentences: int = 0,
    num_workers: int = 0,
    output_suffix: str = "",
):
    """Main entry point.

    Args:
        treebank_ids: List of treebank config IDs to process (None = all).
        max_sentences: Max sentences per treebank (0 = no limit).
        num_workers: Number of parallel workers (0 = auto).
        output_suffix: Suffix for output filename.
    """
    start_time = time.time()

    # Get all configs if not specified
    if treebank_ids is None:
        from datasets import get_dataset_config_names
        treebank_ids = get_dataset_config_names("commul/universal_dependencies")

    logger.info(f"Processing {len(treebank_ids)} treebanks "
                f"(max_sentences={max_sentences}, workers={num_workers or 'auto'})")

    # Load typological metadata
    wals_49a, iso_to_wals, lang_map, codes_map = load_wals_data()
    ud_to_iso = build_ud_to_iso_map()

    if num_workers == 0:
        num_workers = max(1, NUM_CPUS - 1)  # Leave 1 CPU for the main process

    # Process treebanks
    results = []
    errors = []

    if num_workers <= 1 or len(treebank_ids) <= 3:
        # Sequential for small batches
        for i, tb_id in enumerate(treebank_ids):
            logger.info(f"[{i+1}/{len(treebank_ids)}] Processing {tb_id}...")
            result = process_single_treebank((tb_id, max_sentences))
            if result and "error" not in result:
                results.append(result)
                logger.info(f"  Done: {result['structural_stats']['total_sentence_count']} sentences, "
                           f"{result['total_tokens']} tokens in {result['elapsed_seconds']:.1f}s")
            elif result:
                errors.append(result)
                logger.warning(f"  Error: {result['error']}")
    else:
        # Parallel processing
        logger.info(f"Using {num_workers} parallel workers")
        args_list = [(tb_id, max_sentences) for tb_id in treebank_ids]
        completed = 0
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_id = {
                executor.submit(process_single_treebank, args): args[0]
                for args in args_list
            }
            for future in as_completed(future_to_id):
                tb_id = future_to_id[future]
                completed += 1
                try:
                    result = future.result(timeout=600)
                    if result and "error" not in result:
                        results.append(result)
                        if completed % 20 == 0 or completed == len(treebank_ids):
                            logger.info(f"  Progress: {completed}/{len(treebank_ids)} "
                                       f"({len(results)} ok, {len(errors)} errors)")
                    elif result:
                        errors.append(result)
                        logger.warning(f"  Error [{completed}/{len(treebank_ids)}] {tb_id}: "
                                      f"{result.get('error', 'unknown')[:100]}")
                except Exception as e:
                    errors.append({"treebank_id": tb_id, "error": str(e)})
                    logger.warning(f"  Exception [{completed}/{len(treebank_ids)}] {tb_id}: {str(e)[:100]}")

    logger.info(f"Processing complete: {len(results)} succeeded, {len(errors)} errors")

    # Assemble output rows
    output_rows = []
    for result in results:
        row = assemble_output_row(result, wals_49a, iso_to_wals, lang_map, codes_map, ud_to_iso)
        output_rows.append(row)

    # Sort by treebank_id for consistency
    output_rows.sort(key=lambda x: x["input"]["treebank_id"])

    # Coverage statistics
    wals_matched = sum(1 for r in output_rows if r["metadata_wals_49a_value"] is not None)
    spoken_count = sum(1 for r in output_rows if r["metadata_modality"] == "spoken")
    sign_count = sum(1 for r in output_rows if r["metadata_modality"] == "sign")
    small_count = sum(1 for r in output_rows
                     if r["output"]["structural_stats"]["total_sentence_count"] < 100)

    logger.info(f"\n=== Coverage Statistics ===")
    logger.info(f"Total treebanks processed: {len(output_rows)}")
    logger.info(f"WALS 49A matches: {wals_matched}/{len(output_rows)}")
    logger.info(f"Spoken treebanks: {spoken_count}")
    logger.info(f"Sign language treebanks: {sign_count}")
    logger.info(f"Small treebanks (<100 sentences): {small_count}")

    if errors:
        logger.warning(f"Failed treebanks ({len(errors)}):")
        for err in errors[:10]:
            logger.warning(f"  - {err['treebank_id']}: {err.get('error', 'unknown')[:80]}")

    # Save output
    suffix = f"_{output_suffix}" if output_suffix else ""
    output_path = WORKSPACE / f"data_out{suffix}.json"
    output_path.write_text(json.dumps(output_rows, indent=2, ensure_ascii=False))
    logger.info(f"Saved {len(output_rows)} rows to {output_path}")
    logger.info(f"File size: {output_path.stat().st_size / 1e6:.2f} MB")

    elapsed_total = time.time() - start_time
    logger.info(f"Total elapsed: {elapsed_total:.1f}s ({elapsed_total/60:.1f}min)")

    return output_rows


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--treebanks", nargs="*", default=None,
                       help="Specific treebank IDs to process")
    parser.add_argument("--max-sentences", type=int, default=0,
                       help="Max sentences per treebank (0=all)")
    parser.add_argument("--num-workers", type=int, default=0,
                       help="Number of parallel workers (0=auto)")
    parser.add_argument("--output-suffix", type=str, default="",
                       help="Suffix for output filename")
    args = parser.parse_args()

    main(
        treebank_ids=args.treebanks,
        max_sentences=args.max_sentences,
        num_workers=args.num_workers,
        output_suffix=args.output_suffix,
    )
