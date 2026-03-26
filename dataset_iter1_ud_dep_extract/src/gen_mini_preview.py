#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["loguru"]
# ///
"""Generate mini and preview versions of the split full_data_out files.

mini_data_out.json: First 3 datasets, first 3 examples each (full strings)
preview_data_out.json: First 3 datasets, first 3 examples each (strings truncated to 200 chars)
"""
import json
import sys
from pathlib import Path
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")

WORKSPACE = Path("/ai-inventor/aii_pipeline/runs/comp-ling-dobrovoljc_ebw/3_invention_loop/iter_1/gen_art/data_id3_it1__opus")
DATA_DIR = WORKSPACE / "data_out"
COMPACT = (",", ":")


def truncate_strings(obj, max_len: int = 200):
    """Recursively truncate all strings in a JSON-like object."""
    if isinstance(obj, str):
        return obj[:max_len] + "..." if len(obj) > max_len else obj
    elif isinstance(obj, dict):
        return {k: truncate_strings(v, max_len) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [truncate_strings(item, max_len) for item in obj]
    return obj


def main():
    # Load first split file to get metadata and first datasets
    first_part = DATA_DIR / "full_data_out_1.json"
    logger.info(f"Loading {first_part}")
    data = json.loads(first_part.read_text())
    metadata = data["metadata"]

    # Collect first 3 datasets with first 3 examples each
    datasets_mini = []
    datasets_collected = 0

    # May need to look across split files if first part has fewer than 3 datasets
    part_files = sorted(DATA_DIR.glob("full_data_out_*.json"),
                        key=lambda p: int(p.stem.split("_")[-1]))

    for pf in part_files:
        if datasets_collected >= 3:
            break
        pdata = json.loads(pf.read_text())
        for ds in pdata["datasets"]:
            if datasets_collected >= 3:
                break
            mini_ds = {
                "dataset": ds["dataset"],
                "examples": ds["examples"][:3],
            }
            datasets_mini.append(mini_ds)
            datasets_collected += 1
            logger.info(f"  Added {ds['dataset']}: {len(ds['examples'])} total -> 3 examples")

    # Create mini version
    mini_output = {"metadata": metadata, "datasets": datasets_mini}
    mini_path = WORKSPACE / "mini_data_out.json"
    mini_path.write_text(json.dumps(mini_output, indent=2))
    logger.info(f"Wrote mini: {mini_path} ({mini_path.stat().st_size / 1024:.1f} KB)")

    # Create preview version (truncated strings)
    preview_output = truncate_strings(mini_output)
    preview_path = WORKSPACE / "preview_data_out.json"
    preview_path.write_text(json.dumps(preview_output, indent=2))
    logger.info(f"Wrote preview: {preview_path} ({preview_path.stat().st_size / 1024:.1f} KB)")

    # Validate both have correct structure
    for name, path in [("mini", mini_path), ("preview", preview_path)]:
        d = json.loads(path.read_text())
        assert "metadata" in d, f"{name}: missing metadata"
        assert "datasets" in d, f"{name}: missing datasets"
        assert len(d["datasets"]) == 3, f"{name}: expected 3 datasets, got {len(d['datasets'])}"
        for ds in d["datasets"]:
            assert "dataset" in ds, f"{name}: missing dataset name"
            assert "examples" in ds, f"{name}: missing examples"
            assert len(ds["examples"]) <= 3, f"{name}: too many examples"
            for ex in ds["examples"]:
                assert "input" in ex, f"{name}: missing input"
                assert "output" in ex, f"{name}: missing output"
        logger.info(f"  {name} structure validated OK")

    logger.info("Done!")


if __name__ == "__main__":
    main()
