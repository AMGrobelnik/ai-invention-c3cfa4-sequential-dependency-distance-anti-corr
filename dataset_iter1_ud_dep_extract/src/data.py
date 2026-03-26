#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["datasets", "loguru", "huggingface_hub"]
# ///
"""Extract dependency structures from all UD treebanks (commul/universal_dependencies).

For each sentence with >=15 tokens: parse head fields to ints, extract deprel arrays,
compute dependency distance sequences. Per treebank: compute Case morphological feature
proportions, detect spoken/written modality, aggregate sentence counts.

Output conforms to exp_sel_data_out.json schema.
"""

import json
import sys
import os
import gc
import re
import time
import math
import resource
from pathlib import Path
from loguru import logger

# === Paths ===
WORKSPACE = Path("/ai-inventor/aii_pipeline/runs/comp-ling-dobrovoljc_ebw/3_invention_loop/iter_1/gen_art/data_id3_it1__opus")
LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR = WORKSPACE / "temp" / "treebank_parts"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# === Logging ===
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOG_DIR / "data_extraction.log"), rotation="30 MB", level="DEBUG")

# === Resource limits ===
# Container: 29GB RAM, 4 CPUs, no GPU
def _container_ram_gb() -> float:
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return 29.0

TOTAL_RAM_GB = _container_ram_gb()
RAM_BUDGET = int(TOTAL_RAM_GB * 0.65 * 1024**3)  # 65% of container RAM
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
logger.info(f"RAM budget: {RAM_BUDGET / 1e9:.1f} GB (container: {TOTAL_RAM_GB:.1f} GB)")

# === Constants ===
MIN_TOKENS = 15
PRIMARY_THRESHOLD = 20
QUALIFICATION_MIN = 50
COMPACT = (",", ":")

SPOKEN_TREEBANKS = {
    "fr_rhapsodie", "fr_parisstories", "en_childes", "en_eslspok",
    "sl_sst", "no_nynorsklia", "bej_nsc", "pcm_nsc", "kpv_lattice",
    "el_cretan", "nhi_itml", "qfn_fame", "qtd_sagt", "swl_sslc", "es_coser",
}

# Optional: limit configs for testing (set to None for full run)
MAX_CONFIGS = None  # Full run


def detect_spoken_from_comments(sentences: list, max_check: int = 20) -> bool:
    """Check sentence comments for spoken genre indicators."""
    for i, sent in enumerate(sentences):
        if i >= max_check:
            break
        comments = sent.get("comments", [])
        if comments:
            for c in comments:
                if c and re.search(r"[Gg]enre.*spoken|[Ss]poken", str(c)):
                    return True
    return False


def detect_genre_from_comments(sentences: list, max_check: int = 20) -> str:
    """Try to extract genre string from sentence comments."""
    for i, sent in enumerate(sentences):
        if i >= max_check:
            break
        comments = sent.get("comments", [])
        if comments:
            for c in comments:
                if c:
                    m = re.search(r"[Gg]enre\s*[:=]\s*(.+)", str(c))
                    if m:
                        return m.group(1).strip()
    return "unknown"


def process_sentence(sentence: dict, config_name: str) -> tuple:
    """Process one UD sentence.

    Returns:
        (example_dict or None, stats_dict)
    """
    tokens = sentence.get("tokens", [])
    token_count = len(tokens)

    stats = {
        "total": 1,
        "token_count_sum": token_count,
        "case_count": 0,
        "total_tokens": token_count,
        "multi_root": 0,
        "skipped_validation": 0,
    }

    # Count Case features across ALL tokens (even in short sentences)
    feats = sentence.get("feats", [])
    if feats:
        for feat_str in feats:
            if feat_str and feat_str != "_" and feat_str is not None:
                if "Case=" in str(feat_str):
                    stats["case_count"] += 1

    # Skip short sentences (but stats already accumulated)
    if token_count < MIN_TOKENS:
        return None, stats

    # Parse head array: strings -> ints
    head_raw = sentence.get("head", [])
    if len(head_raw) != token_count:
        stats["skipped_validation"] = 1
        return None, stats

    try:
        head_array = [int(h) for h in head_raw]
    except (ValueError, TypeError):
        stats["skipped_validation"] = 1
        return None, stats

    # Validate head values
    root_count = 0
    valid = True
    for h in head_array:
        if h < 0 or h > token_count:
            valid = False
            break
        if h == 0:
            root_count += 1

    if not valid or root_count == 0:
        stats["skipped_validation"] = 1
        return None, stats

    if root_count > 1:
        stats["multi_root"] = 1

    # Deprel array
    deprel_array = sentence.get("deprel", [])
    if len(deprel_array) != token_count:
        stats["skipped_validation"] = 1
        return None, stats

    # Compute dependency distance sequence (non-root tokens, linear order)
    dd_sequence = []
    for i in range(token_count):
        h = head_array[i]
        if h == 0:
            continue  # Skip root token
        position = i + 1  # Convert to 1-indexed
        dd = abs(position - h)
        dd_sequence.append(dd)

    # Length bucket
    length_bucket = "ge20" if token_count >= PRIMARY_THRESHOLD else "ge15"
    sent_id = sentence.get("sent_id", "")

    input_data = json.dumps(
        {"head_array": head_array, "deprel_array": list(deprel_array)},
        separators=COMPACT,
    )
    output_data = json.dumps(
        {"dd_sequence": dd_sequence, "token_count": token_count},
        separators=COMPACT,
    )

    example = {
        "input": input_data,
        "output": output_data,
        "metadata_treebank_id": config_name,
        "metadata_sentence_id": str(sent_id),
        "metadata_token_count": str(token_count),
        "metadata_length_bucket": length_bucket,
    }
    return example, stats


def process_treebank(config_name: str) -> tuple:
    """Process one treebank config. Returns (dataset_entry_or_None, summary_dict_or_None)."""
    from datasets import load_dataset

    t0 = time.time()
    logger.info(f"Loading treebank: {config_name}")

    try:
        ds = load_dataset("commul/universal_dependencies", config_name)
    except Exception as e:
        logger.error(f"Failed to load {config_name}: {e}")
        return None, None

    # Combine all splits
    all_sentences = []
    for split_name in ds.keys():
        all_sentences.extend(ds[split_name])

    # Detect genre/spoken from comments
    is_spoken = config_name in SPOKEN_TREEBANKS
    if is_spoken:
        genre = "spoken"
    else:
        is_spoken = detect_spoken_from_comments(all_sentences)
        genre = "spoken" if is_spoken else detect_genre_from_comments(all_sentences)

    examples = []
    total_count = 0
    count_ge15 = 0
    count_ge20 = 0
    total_case = 0
    total_tokens_all = 0
    multi_root_count = 0
    length_sum = 0
    skipped = 0

    for sent in all_sentences:
        example, stats = process_sentence(sent, config_name)
        total_count += stats["total"]
        total_case += stats["case_count"]
        total_tokens_all += stats["total_tokens"]
        multi_root_count += stats["multi_root"]
        length_sum += stats["token_count_sum"]
        skipped += stats["skipped_validation"]

        if example is not None:
            tc = int(example["metadata_token_count"])
            count_ge15 += 1
            if tc >= PRIMARY_THRESHOLD:
                count_ge20 += 1
            examples.append(example)

    elapsed = time.time() - t0
    qualifies = count_ge20 >= QUALIFICATION_MIN
    mean_len = length_sum / max(total_count, 1)
    case_proportion = total_case / max(total_tokens_all, 1)

    summary = {
        "treebank_id": config_name,
        "language": config_name.split("_")[0],
        "total_sentences": total_count,
        "sentences_ge15": count_ge15,
        "sentences_ge20": count_ge20,
        "case_feature_proportion": round(case_proportion, 4),
        "is_spoken": is_spoken,
        "genre": genre,
        "mean_sentence_length": round(mean_len, 2),
        "qualifies_primary": qualifies,
        "multi_root_sentences": multi_root_count,
        "skipped_validation": skipped,
    }

    logger.info(
        f"  {config_name}: {total_count} total, {count_ge15} ge15, {count_ge20} ge20, "
        f"qualifies={qualifies}, time={elapsed:.1f}s"
    )

    # Free memory
    del ds, all_sentences
    gc.collect()

    if not qualifies:
        return None, summary

    dataset_entry = {"dataset": config_name, "examples": examples}
    return dataset_entry, summary


def split_output_files(datasets_list: list, metadata: dict) -> None:
    """Split output into <45MB parts in data_out/ directory."""
    data_out_dir = WORKSPACE / "data_out"
    data_out_dir.mkdir(exist_ok=True)

    MAX_PART_SIZE = 40 * 1024 * 1024  # 40MB target per part
    current_group = []
    current_size = 0
    part_num = 1

    for ds_entry in datasets_list:
        entry_bytes = len(json.dumps(ds_entry, separators=COMPACT).encode("utf-8"))
        if current_size + entry_bytes > MAX_PART_SIZE and current_group:
            _write_part(data_out_dir, part_num, current_group, metadata)
            part_num += 1
            current_group = []
            current_size = 0
        current_group.append(ds_entry)
        current_size += entry_bytes

    if current_group:
        _write_part(data_out_dir, part_num, current_group, metadata)

    logger.info(f"Split into {part_num} parts in {data_out_dir}")


def _write_part(out_dir: Path, part_num: int, datasets: list, metadata: dict) -> None:
    part = {"metadata": metadata, "datasets": datasets}
    path = out_dir / f"full_data_out_{part_num}.json"
    path.write_text(json.dumps(part, separators=COMPACT))
    size_mb = path.stat().st_size / (1024 * 1024)
    logger.info(f"  Part {part_num}: {size_mb:.1f} MB, {len(datasets)} treebanks")


@logger.catch
def main():
    logger.info("=" * 60)
    logger.info("UD Treebank Dependency Structure Extraction")
    logger.info("=" * 60)

    # Step 1: Enumerate all treebank configs
    logger.info("Step 1: Enumerating treebank configurations...")
    try:
        from datasets import get_dataset_config_names
        configs = get_dataset_config_names("commul/universal_dependencies")
    except Exception as e:
        logger.warning(f"get_dataset_config_names failed: {e}, trying HfApi fallback")
        from huggingface_hub import HfApi
        api = HfApi()
        info = api.dataset_info("commul/universal_dependencies")
        configs_set = set()
        for s in info.siblings:
            if s.rfilename.endswith(".parquet"):
                parts = s.rfilename.split("/")
                if len(parts) >= 2:
                    configs_set.add(parts[0])
        configs = sorted(configs_set)

    logger.info(f"Found {len(configs)} treebank configurations")

    if MAX_CONFIGS is not None:
        configs = configs[:MAX_CONFIGS]
        logger.info(f"Limiting to first {MAX_CONFIGS} configs for testing")

    # Step 2-3: Process each treebank
    qualifying_names = []
    all_summaries = {}
    failed_configs = []
    total_qualifying_sentences = 0
    start_time = time.time()

    for i, config_name in enumerate(configs):
        logger.info(f"--- [{i + 1}/{len(configs)}] {config_name} ---")
        try:
            dataset_entry, summary = process_treebank(config_name)
        except MemoryError:
            logger.error(f"MemoryError on {config_name}, skipping")
            failed_configs.append(config_name)
            gc.collect()
            continue
        except Exception as e:
            logger.error(f"Unhandled error on {config_name}: {e}")
            failed_configs.append(config_name)
            continue

        if summary is not None:
            all_summaries[config_name] = summary

        if dataset_entry is not None:
            total_qualifying_sentences += len(dataset_entry["examples"])
            # Write to temp file to conserve memory
            temp_path = TEMP_DIR / f"{config_name}.json"
            temp_path.write_text(json.dumps(dataset_entry, separators=COMPACT))
            qualifying_names.append(config_name)
            del dataset_entry
            gc.collect()

        # Progress report every 25 treebanks
        if (i + 1) % 25 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed * 60  # configs per minute
            remaining = (len(configs) - i - 1) / max(rate, 0.1)
            logger.info(
                f"Progress: {i + 1}/{len(configs)}, "
                f"{len(qualifying_names)} qualifying, "
                f"{total_qualifying_sentences} sentences, "
                f"rate={rate:.1f}/min, ETA={remaining:.0f}min"
            )

    total_elapsed = time.time() - start_time

    # Step 4: Build metadata
    qualifying_count = sum(1 for s in all_summaries.values() if s.get("qualifies_primary"))
    spoken_count = sum(1 for s in all_summaries.values() if s.get("is_spoken"))

    metadata = {
        "source": "commul/universal_dependencies",
        "ud_version": "2.17",
        "extraction_date": "2026-03-26",
        "min_token_threshold": MIN_TOKENS,
        "primary_threshold": PRIMARY_THRESHOLD,
        "total_treebanks_processed": len(configs),
        "failed_treebanks": len(failed_configs),
        "qualifying_treebanks_ge50_at_20tok": qualifying_count,
        "total_qualifying_sentences": total_qualifying_sentences,
        "spoken_treebanks": spoken_count,
        "treebank_summaries": all_summaries,
    }

    logger.info("=" * 60)
    logger.info("=== SUMMARY ===")
    logger.info(f"Processed: {len(configs) - len(failed_configs)}/{len(configs)}")
    logger.info(f"Failed: {len(failed_configs)} — {failed_configs}")
    logger.info(f"Qualifying treebanks: {qualifying_count}")
    logger.info(f"Total qualifying sentences: {total_qualifying_sentences}")
    logger.info(f"Spoken treebanks: {spoken_count}")
    logger.info(f"Total time: {total_elapsed:.0f}s ({total_elapsed / 60:.1f}min)")

    # Step 5: Assemble final output from temp files
    logger.info("Assembling final output from temp files...")
    datasets_list = []
    for cname in qualifying_names:
        temp_path = TEMP_DIR / f"{cname}.json"
        entry = json.loads(temp_path.read_text())
        datasets_list.append(entry)

    output = {"metadata": metadata, "datasets": datasets_list}

    # Write full output
    output_path = WORKSPACE / "data_out.json"
    logger.info(f"Writing to {output_path}...")
    raw = json.dumps(output, separators=COMPACT)
    output_path.write_text(raw)
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Output size: {file_size_mb:.1f} MB")

    # Step 6: Handle file size limits
    if file_size_mb > 45:
        logger.info("File exceeds 45MB limit, splitting...")
        split_output_files(datasets_list, metadata)
        output_path.unlink()
        logger.info("Deleted oversized single file")
    else:
        final_path = WORKSPACE / "full_data_out.json"
        output_path.rename(final_path)
        logger.info(f"Saved as {final_path}")

    # Free memory
    del datasets_list, output, raw
    gc.collect()

    # Step 7: Generate mini and preview versions
    logger.info("Generating mini and preview versions...")
    generate_mini_preview()

    logger.info("=== Extraction complete ===")


def truncate_strings(obj, max_len: int = 200):
    """Recursively truncate all strings in a JSON-like object."""
    if isinstance(obj, str):
        return obj[:max_len] + "..." if len(obj) > max_len else obj
    elif isinstance(obj, dict):
        return {k: truncate_strings(v, max_len) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [truncate_strings(item, max_len) for item in obj]
    return obj


def generate_mini_preview() -> None:
    """Generate mini_data_out.json (3 datasets, 3 examples each) and
    preview_data_out.json (same but with strings truncated to 200 chars).
    Reads from data_out/ split files or full_data_out.json."""
    data_out_dir = WORKSPACE / "data_out"
    full_single = WORKSPACE / "full_data_out.json"

    # Determine source files
    if data_out_dir.exists():
        part_files = sorted(
            data_out_dir.glob("full_data_out_*.json"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )
    elif full_single.exists():
        part_files = [full_single]
    else:
        logger.error("No full_data_out files found!")
        return

    # Collect first 3 datasets with first 3 examples each
    metadata = None
    datasets_mini = []
    for pf in part_files:
        if len(datasets_mini) >= 3:
            break
        pdata = json.loads(pf.read_text())
        if metadata is None:
            metadata = pdata["metadata"]
        for ds in pdata["datasets"]:
            if len(datasets_mini) >= 3:
                break
            datasets_mini.append({
                "dataset": ds["dataset"],
                "examples": ds["examples"][:3],
            })
            logger.info(f"  mini/preview: added {ds['dataset']} ({len(ds['examples'])} -> 3 examples)")

    # Write mini
    mini_output = {"metadata": metadata, "datasets": datasets_mini}
    mini_path = WORKSPACE / "mini_data_out.json"
    mini_path.write_text(json.dumps(mini_output, indent=2))
    logger.info(f"Wrote {mini_path.name} ({mini_path.stat().st_size / 1024:.1f} KB)")

    # Write preview (truncated strings)
    preview_output = truncate_strings(mini_output)
    preview_path = WORKSPACE / "preview_data_out.json"
    preview_path.write_text(json.dumps(preview_output, indent=2))
    logger.info(f"Wrote {preview_path.name} ({preview_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
