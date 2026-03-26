#!/usr/bin/env python3
"""Convert per-treebank grammar profiles from data_out.json to exp_sel_data_out schema.

Loads the raw processed UD treebank profiles (336 rows from data_out.json),
converts each row into a schema-compliant example with string input/output
and metadata_* fields, and saves to full_data_out.json.

Dataset: universal_dependencies_grammar_profiles
  - 336 examples (one per UD treebank)
  - input: JSON string of {treebank_id, language_prefix, iso639_3}
  - output: JSON string of {head_direction_profile, sibling_order_templates,
            structural_stats, deprel_frequency_counts, case_feature_proportion}
  - metadata_fold: "all" (all splits combined)
  - metadata_modality: spoken/written/mixed/unknown
  - metadata_wals_49a_value: WALS 49A case-marking value (1-9 or null)
  - metadata_wals_49a_label: human-readable label for 49A value
  - metadata_language_family: language family name
  - metadata_language_name: language name
"""

from loguru import logger
from pathlib import Path
import json
import sys

WORKSPACE = Path("/ai-inventor/aii_pipeline/runs/comp-ling-dobrovoljc_ebw/3_invention_loop/iter_1/gen_art/data_id5_it1__opus")
LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOG_DIR / "data.log"), rotation="30 MB", level="DEBUG")


def load_raw_rows(path: Path) -> list[dict]:
    """Load raw treebank profile rows from data_out.json."""
    logger.info(f"Loading raw data from {path}")
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError(f"Expected list, got {type(raw)}")
    logger.info(f"Loaded {len(raw)} raw rows")
    return raw


def convert_row_to_example(row: dict, idx: int) -> dict:
    """Convert one raw treebank row to a schema-compliant example.

    Schema requires:
      - input: string
      - output: string
      - metadata_*: flat metadata fields
    """
    example = {
        "input": json.dumps(row["input"], ensure_ascii=False),
        "output": json.dumps(row["output"], ensure_ascii=False),
    }
    # Copy all metadata_* fields from the raw row
    for key, value in row.items():
        if key.startswith("metadata_"):
            example[key] = value

    # Add row index metadata
    example["metadata_row_index"] = idx

    return example


def validate_example(example: dict, idx: int) -> bool:
    """Validate a single example has required fields and correct types."""
    if "input" not in example or not isinstance(example["input"], str):
        logger.error(f"Example {idx}: missing or non-string 'input'")
        return False
    if "output" not in example or not isinstance(example["output"], str):
        logger.error(f"Example {idx}: missing or non-string 'output'")
        return False
    # Verify input is valid JSON
    try:
        parsed_input = json.loads(example["input"])
        if "treebank_id" not in parsed_input:
            logger.error(f"Example {idx}: input missing 'treebank_id'")
            return False
    except json.JSONDecodeError:
        logger.error(f"Example {idx}: input is not valid JSON")
        return False
    # Verify output is valid JSON
    try:
        parsed_output = json.loads(example["output"])
        required_output_keys = [
            "head_direction_profile",
            "sibling_order_templates",
            "structural_stats",
            "deprel_frequency_counts",
            "case_feature_proportion",
        ]
        for key in required_output_keys:
            if key not in parsed_output:
                logger.error(f"Example {idx}: output missing '{key}'")
                return False
    except json.JSONDecodeError:
        logger.error(f"Example {idx}: output is not valid JSON")
        return False
    # Check no disallowed top-level keys
    allowed_prefixes = {"input", "output", "metadata_"}
    for key in example:
        if key not in ("input", "output") and not key.startswith("metadata_"):
            logger.error(f"Example {idx}: disallowed key '{key}'")
            return False
    return True


@logger.catch
def main():
    input_path = WORKSPACE / "data_out.json"
    output_path = WORKSPACE / "full_data_out.json"

    # Load raw rows
    raw_rows = load_raw_rows(input_path)

    # Convert each row to a schema-compliant example
    examples = []
    errors = 0
    for idx, row in enumerate(raw_rows):
        try:
            example = convert_row_to_example(row, idx)
            if validate_example(example, idx):
                examples.append(example)
            else:
                errors += 1
                logger.warning(f"Skipping invalid example {idx} (treebank: {row.get('input', {}).get('treebank_id', '?')})")
        except Exception:
            logger.error(f"Failed to convert row {idx}")
            errors += 1

    logger.info(f"Converted {len(examples)} examples ({errors} errors)")

    if not examples:
        logger.error("No valid examples produced — aborting")
        sys.exit(1)

    # Assemble schema-compliant output
    output = {
        "metadata": {
            "source": "commul/universal_dependencies (HuggingFace)",
            "description": "Per-treebank grammar profiles from 336 Universal Dependencies treebanks",
            "version": "1.0",
            "wals_source": "cldf-datasets/wals (GitHub)",
            "ud_version": "2.17",
            "total_treebanks": len(examples),
        },
        "datasets": [
            {
                "dataset": "universal_dependencies_grammar_profiles",
                "examples": examples,
            }
        ],
    }

    # Write output
    output_text = json.dumps(output, indent=2, ensure_ascii=False)
    output_path.write_text(output_text)
    size_mb = output_path.stat().st_size / 1e6
    logger.info(f"Saved {len(examples)} examples to {output_path.name} ({size_mb:.2f} MB)")

    # Summary statistics
    modalities = {}
    families = {}
    wals_matched = 0
    for ex in examples:
        mod = ex.get("metadata_modality", "unknown")
        modalities[mod] = modalities.get(mod, 0) + 1
        fam = ex.get("metadata_language_family", "unknown")
        families[fam] = families.get(fam, 0) + 1
        if ex.get("metadata_wals_49a_value") is not None:
            wals_matched += 1

    logger.info(f"Modalities: {modalities}")
    logger.info(f"WALS 49A matched: {wals_matched}/{len(examples)}")
    logger.info(f"Language families: {len(families)} distinct")
    logger.info("Top 10 families: " + ", ".join(
        f"{k}({v})" for k, v in sorted(families.items(), key=lambda x: -x[1])[:10]
    ))


if __name__ == "__main__":
    main()
