from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm

from utils.utils import readSWC


def list_swc_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix == ".swc")


def next_part_index(output_dir: Path, prefix: str) -> int:
    max_idx = 0
    for path in output_dir.glob(f"{prefix}_part_*.parquet"):
        stem = path.stem
        try:
            max_idx = max(max_idx, int(stem.rsplit("_part_", 1)[1]))
        except Exception:
            continue
    return max_idx + 1


def load_completed_sources(progress_file: Path) -> set[str]:
    if not progress_file.exists():
        return set()

    completed = set()
    with progress_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                completed.add(line)
    return completed


def append_completed_sources(progress_file: Path, source_ids: Iterable[str]) -> None:
    with progress_file.open("a", encoding="utf-8") as f:
        for source_id in source_ids:
            f.write(f"{source_id}\n")


def build_or_load_global_dendrite_index(
    target_dir: Path,
    cache_path: Path,
    progress_mininterval: float = 5.0,
) -> dict:
    if cache_path.exists():
        with cache_path.open("rb") as f:
            cache = pickle.load(f)
        cache["tree"] = cKDTree(cache["all_den_pts"])
        return cache

    target_files = list_swc_files(target_dir)
    target_stems: list[str] = []
    den_pts_chunks: list[np.ndarray] = []
    den_target_chunks: list[np.ndarray] = []

    next_target_idx = 0
    for target_path in tqdm(
        target_files,
        desc="indexing target dendrites",
        mininterval=progress_mininterval,
        dynamic_ncols=True,
    ):
        df = readSWC(str(target_path), use_bouton=False)
        den = df.loc[~df["type"].isin([1, 2, 5, 0]), ["x", "y", "z"]].copy()
        den = den.apply(pd.to_numeric, errors="coerce").dropna().astype(np.int32)
        if den.empty:
            continue

        den_pts_chunks.append(den.to_numpy(dtype=np.int32, copy=True))
        den_target_chunks.append(np.full(len(den), next_target_idx, dtype=np.int32))
        target_stems.append(target_path.stem)
        next_target_idx += 1

    if not den_pts_chunks:
        raise ValueError(f"No dendrite points found under: {target_dir}")

    all_den_pts = np.vstack(den_pts_chunks)
    all_den_target_idx = np.concatenate(den_target_chunks)
    cache = {
        "all_den_pts": all_den_pts,
        "all_den_target_idx": all_den_target_idx,
        "target_stems": target_stems,
    }
    with cache_path.open("wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    cache["tree"] = cKDTree(all_den_pts)
    return cache


def write_parquet_part(records: list[dict], output_dir: Path, prefix: str, part_idx: int) -> Path:
    part_path = output_dir / f"{prefix}_part_{part_idx:06d}.parquet"
    df = pd.DataFrame.from_records(records)
    df.to_parquet(part_path, index=False)
    return part_path


def load_part_paths(parts_dir: Path, prefix: str) -> list[Path]:
    return sorted(parts_dir.glob(f"{prefix}_part_*.parquet"))
