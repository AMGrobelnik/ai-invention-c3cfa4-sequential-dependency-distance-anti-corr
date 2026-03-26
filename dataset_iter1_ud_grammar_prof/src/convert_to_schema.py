#!/usr/bin/env python3
"""Convert data_out.json to exp_sel_data_out schema format.

The schema requires:
- Top-level object with "datasets" array
- Each dataset has "dataset" (str) and "examples" array
- Each example has "input" (str), "output" (str), plus metadata_* fields
"""

from pathlib import Path
import json

WORKSPACE = Path("/ai-inventor/aii_pipeline/runs/comp-ling-dobrovoljc_ebw/3_invention_loop/iter_1/gen_art/data_id5_it1__opus")


def convert_to_schema(rows: list[dict]) -> dict:
    """Convert raw rows to schema-compliant format."""
    examples = []
    for row in rows:
        example = {
            "input": json.dumps(row["input"], ensure_ascii=False),
            "output": json.dumps(row["output"], ensure_ascii=False),
        }
        # Add metadata_ fields
        for k, v in row.items():
            if k.startswith("metadata_"):
                example[k] = v
        examples.append(example)

    return {
        "metadata": {
            "source": "commul/universal_dependencies (HuggingFace)",
            "description": "Per-treebank grammar profiles from 336 Universal Dependencies treebanks",
            "version": "1.0",
            "wals_source": "cldf-datasets/wals (GitHub)",
            "ud_version": "2.17",
        },
        "datasets": [
            {
                "dataset": "universal_dependencies_grammar_profiles",
                "examples": examples,
            }
        ],
    }


def main():
    # Convert full
    for suffix, in_name, out_name in [
        ("full", "data_out.json", "full_data_out.json"),
        ("mini", "data_out_mini.json", "mini_data_out.json"),
        ("preview", "data_out_preview.json", "preview_data_out.json"),
    ]:
        in_path = WORKSPACE / in_name
        out_path = WORKSPACE / out_name
        if not in_path.exists():
            print(f"Skipping {in_name} (not found)")
            continue

        data = json.loads(in_path.read_text())
        converted = convert_to_schema(data)
        out_path.write_text(json.dumps(converted, indent=2, ensure_ascii=False))
        n_examples = len(converted["datasets"][0]["examples"])
        size_mb = out_path.stat().st_size / 1e6
        print(f"{suffix}: {n_examples} examples -> {out_path.name} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
