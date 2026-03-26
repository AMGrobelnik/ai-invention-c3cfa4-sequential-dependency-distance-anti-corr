#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["loguru"]
# ///
"""Re-split oversized part files (>45MB) by splitting large treebank examples across parts."""
import json
import sys
from pathlib import Path
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")

WORKSPACE = Path("/ai-inventor/aii_pipeline/runs/comp-ling-dobrovoljc_ebw/3_invention_loop/iter_1/gen_art/data_id3_it1__opus")
DATA_DIR = WORKSPACE / "data_out"
MAX_SIZE = 44 * 1024 * 1024  # 44MB
COMPACT = (",", ":")


def main():
    # Collect all existing parts
    part_files = sorted(DATA_DIR.glob("full_data_out_*.json"), key=lambda p: int(p.stem.split("_")[-1]))
    logger.info(f"Found {len(part_files)} existing parts")

    # Load all datasets from all parts, preserving metadata from part 1
    all_datasets = []
    metadata = None
    for pf in part_files:
        data = json.loads(pf.read_text())
        if metadata is None:
            metadata = data["metadata"]
        all_datasets.extend(data["datasets"])

    logger.info(f"Total datasets (treebanks): {len(all_datasets)}")

    # Remove old parts
    for pf in part_files:
        pf.unlink()

    # Re-split with proper handling of large single treebanks
    part_num = 1
    current_datasets = []
    current_size = 0
    meta_size = len(json.dumps({"metadata": metadata}, separators=COMPACT).encode("utf-8"))

    for ds in all_datasets:
        ds_json = json.dumps(ds, separators=COMPACT)
        ds_size = len(ds_json.encode("utf-8"))

        # If single treebank exceeds limit, split its examples
        if ds_size + meta_size > MAX_SIZE:
            # Flush current group first
            if current_datasets:
                _write_part(part_num, current_datasets, metadata)
                part_num += 1
                current_datasets = []
                current_size = 0

            # Split this large treebank's examples
            examples = ds["examples"]
            chunk_target = MAX_SIZE - meta_size - 500  # overhead for wrapper
            chunk = []
            chunk_size = 0
            sub_idx = 0

            for ex in examples:
                ex_json = json.dumps(ex, separators=COMPACT)
                ex_size = len(ex_json.encode("utf-8")) + 1  # +1 for comma
                if chunk_size + ex_size > chunk_target and chunk:
                    sub_ds = {"dataset": f"{ds['dataset']}", "examples": chunk}
                    _write_part(part_num, [sub_ds], metadata)
                    part_num += 1
                    chunk = []
                    chunk_size = 0
                    sub_idx += 1
                chunk.append(ex)
                chunk_size += ex_size

            if chunk:
                sub_ds = {"dataset": f"{ds['dataset']}", "examples": chunk}
                _write_part(part_num, [sub_ds], metadata)
                part_num += 1
            continue

        # Normal case: add to current group
        if current_size + ds_size > MAX_SIZE - meta_size and current_datasets:
            _write_part(part_num, current_datasets, metadata)
            part_num += 1
            current_datasets = []
            current_size = 0

        current_datasets.append(ds)
        current_size += ds_size

    if current_datasets:
        _write_part(part_num, current_datasets, metadata)

    # Verify
    logger.info("Verification:")
    for pf in sorted(DATA_DIR.glob("full_data_out_*.json"), key=lambda p: int(p.stem.split("_")[-1])):
        sz = pf.stat().st_size / (1024 * 1024)
        flag = " *** OVERSIZED ***" if sz > 45 else ""
        logger.info(f"  {pf.name}: {sz:.1f} MB{flag}")


def _write_part(part_num: int, datasets: list, metadata: dict):
    part = {"metadata": metadata, "datasets": datasets}
    path = DATA_DIR / f"full_data_out_{part_num}.json"
    path.write_text(json.dumps(part, separators=COMPACT))
    sz = path.stat().st_size / (1024 * 1024)
    n_ex = sum(len(d["examples"]) for d in datasets)
    logger.info(f"  Part {part_num}: {sz:.1f} MB, {len(datasets)} treebanks, {n_ex} examples")


if __name__ == "__main__":
    main()
