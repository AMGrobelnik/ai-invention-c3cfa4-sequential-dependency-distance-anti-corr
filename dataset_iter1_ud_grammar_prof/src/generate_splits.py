#!/usr/bin/env python3
"""Generate full/mini/preview splits from the processed grammar profiles dataset.

Reads data_out.json and produces:
- data_out.json (full, all ~336 rows) — already exists, just validate
- data_out_mini.json (50 diverse rows across language families/modalities)
- data_out_preview.json (5 rows: en_ewt, ja_gsd, tr_imst, fi_tdt + 1 diverse)
"""

from loguru import logger
from pathlib import Path
import json
import sys

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

WORKSPACE = Path("/ai-inventor/aii_pipeline/runs/comp-ling-dobrovoljc_ebw/3_invention_loop/iter_1/gen_art/data_id5_it1__opus")


def select_preview_rows(data: list[dict]) -> list[dict]:
    """Select 5 typologically diverse preview rows."""
    target_ids = ["en_ewt", "ja_gsd", "tr_imst", "fi_tdt"]
    preview = []
    remaining = []

    for row in data:
        if row["input"]["treebank_id"] in target_ids:
            preview.append(row)
        else:
            remaining.append(row)

    # Sort target by the original order
    preview.sort(key=lambda r: target_ids.index(r["input"]["treebank_id"])
                 if r["input"]["treebank_id"] in target_ids else 999)

    # Add 1 more diverse row (Arabic if available, else first non-selected)
    if len(preview) < 5:
        for r in remaining:
            if r["input"]["treebank_id"] == "ar_padt":
                preview.append(r)
                break
        if len(preview) < 5 and remaining:
            # Pick one with WALS data from a different family
            families_seen = {r.get("metadata_language_family", "") for r in preview}
            for r in remaining:
                fam = r.get("metadata_language_family", "")
                if fam and fam not in families_seen:
                    preview.append(r)
                    break
            if len(preview) < 5 and remaining:
                preview.append(remaining[0])

    return preview[:5]


def select_mini_rows(data: list[dict], n: int = 50) -> list[dict]:
    """Select 50 diverse rows sampling across language families, modalities, case-marking."""
    if len(data) <= n:
        return data

    selected = []
    selected_ids = set()

    # Priority 1: ensure preview rows are included
    preview_ids = {"en_ewt", "ja_gsd", "tr_imst", "fi_tdt", "ar_padt"}
    for row in data:
        if row["input"]["treebank_id"] in preview_ids:
            selected.append(row)
            selected_ids.add(row["input"]["treebank_id"])

    # Priority 2: one from each language family
    families = {}
    for row in data:
        fam = row.get("metadata_language_family", "")
        if fam and fam not in families and row["input"]["treebank_id"] not in selected_ids:
            families[fam] = row

    for fam, row in sorted(families.items()):
        if len(selected) >= n:
            break
        selected.append(row)
        selected_ids.add(row["input"]["treebank_id"])

    # Priority 3: spoken treebanks
    for row in data:
        if len(selected) >= n:
            break
        if row["metadata_modality"] == "spoken" and row["input"]["treebank_id"] not in selected_ids:
            selected.append(row)
            selected_ids.add(row["input"]["treebank_id"])

    # Priority 4: different WALS 49A values
    wals_values_seen = {r.get("metadata_wals_49a_value") for r in selected if r.get("metadata_wals_49a_value") is not None}
    for row in data:
        if len(selected) >= n:
            break
        wv = row.get("metadata_wals_49a_value")
        if wv is not None and wv not in wals_values_seen and row["input"]["treebank_id"] not in selected_ids:
            selected.append(row)
            selected_ids.add(row["input"]["treebank_id"])
            wals_values_seen.add(wv)

    # Priority 5: large treebanks with diverse structural stats
    remaining = [r for r in data if r["input"]["treebank_id"] not in selected_ids]
    remaining.sort(key=lambda r: r["output"]["structural_stats"]["total_sentence_count"], reverse=True)
    for row in remaining:
        if len(selected) >= n:
            break
        selected.append(row)
        selected_ids.add(row["input"]["treebank_id"])

    # Sort by treebank_id for consistency
    selected.sort(key=lambda r: r["input"]["treebank_id"])
    return selected[:n]


def validate_row(row: dict, idx: int) -> list[str]:
    """Validate a single output row. Returns list of issues."""
    issues = []

    # Check required fields
    for field in ["input", "output", "metadata_fold", "metadata_modality"]:
        if field not in row:
            issues.append(f"Row {idx}: missing field '{field}'")

    inp = row.get("input", {})
    for f in ["treebank_id", "language_prefix", "iso639_3"]:
        if f not in inp:
            issues.append(f"Row {idx}: missing input.{f}")

    out = row.get("output", {})
    for f in ["head_direction_profile", "sibling_order_templates", "structural_stats",
              "deprel_frequency_counts", "case_feature_proportion"]:
        if f not in out:
            issues.append(f"Row {idx}: missing output.{f}")

    # Check head_direction_profile has at least 1 entry
    hdp = out.get("head_direction_profile", {})
    if not hdp:
        issues.append(f"Row {idx} ({inp.get('treebank_id', '?')}): empty head_direction_profile")

    # Check structural_stats required fields
    stats = out.get("structural_stats", {})
    for sf in ["mean_tree_depth", "std_tree_depth", "mean_branching_factor",
               "proportion_projective", "total_sentence_count",
               "sentence_count_ge15", "sentence_count_ge20",
               "mean_sentence_length", "std_sentence_length"]:
        if sf not in stats:
            issues.append(f"Row {idx}: missing structural_stats.{sf}")

    return issues


@logger.catch
def main():
    input_path = WORKSPACE / "data_out.json"
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    logger.info(f"Loading {input_path}")
    data = json.loads(input_path.read_text())
    logger.info(f"Loaded {len(data)} rows")

    # Validate all rows
    all_issues = []
    for i, row in enumerate(data):
        issues = validate_row(row, i)
        all_issues.extend(issues)

    if all_issues:
        logger.warning(f"Validation found {len(all_issues)} issues:")
        for issue in all_issues[:20]:
            logger.warning(f"  {issue}")
    else:
        logger.info("All rows passed validation")

    # Coverage stats
    wals_matched = sum(1 for r in data if r.get("metadata_wals_49a_value") is not None)
    spoken_count = sum(1 for r in data if r.get("metadata_modality") == "spoken")
    sign_count = sum(1 for r in data if r.get("metadata_modality") == "sign")
    small_count = sum(1 for r in data
                     if r["output"]["structural_stats"]["total_sentence_count"] < 100)
    families = set(r.get("metadata_language_family", "") for r in data if r.get("metadata_language_family"))

    logger.info(f"\n=== Dataset Summary ===")
    logger.info(f"Total treebanks: {len(data)}")
    logger.info(f"Language families: {len(families)}")
    logger.info(f"WALS 49A matches: {wals_matched}/{len(data)} ({wals_matched/len(data)*100:.1f}%)")
    logger.info(f"Spoken: {spoken_count}, Sign: {sign_count}, Written: {len(data)-spoken_count-sign_count}")
    logger.info(f"Small (<100 sentences): {small_count}")
    total_sents = sum(r["output"]["structural_stats"]["total_sentence_count"] for r in data)
    logger.info(f"Total sentences across all treebanks: {total_sents:,}")

    # Generate preview (5 rows)
    preview = select_preview_rows(data)
    preview_path = WORKSPACE / "data_out_preview.json"
    preview_path.write_text(json.dumps(preview, indent=2, ensure_ascii=False))
    logger.info(f"Preview: {len(preview)} rows -> {preview_path} ({preview_path.stat().st_size/1e3:.1f} KB)")
    for r in preview:
        logger.info(f"  - {r['input']['treebank_id']} ({r.get('metadata_language_name', '?')})")

    # Generate mini (50 rows)
    mini = select_mini_rows(data, 50)
    mini_path = WORKSPACE / "data_out_mini.json"
    mini_path.write_text(json.dumps(mini, indent=2, ensure_ascii=False))
    logger.info(f"Mini: {len(mini)} rows -> {mini_path} ({mini_path.stat().st_size/1e6:.2f} MB)")

    # Report file sizes
    logger.info(f"\n=== File Sizes ===")
    logger.info(f"  data_out.json:         {input_path.stat().st_size/1e6:.2f} MB")
    logger.info(f"  data_out_mini.json:    {mini_path.stat().st_size/1e6:.2f} MB")
    logger.info(f"  data_out_preview.json: {preview_path.stat().st_size/1e3:.1f} KB")


if __name__ == "__main__":
    main()
