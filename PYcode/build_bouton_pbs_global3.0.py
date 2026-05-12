#!/usr/bin/env python3
"""
Build bouton summary points directly from source bouton files and a global
dendrite index. This replaces per-pair CSV generation with chunked parquet
part files plus source-level resume support.

Main outputs:
  1. PBs_part_*.parquet
  2. source_progress.txt
  3. optional sparse pair-level connectivity parquet
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm

from utils.utils import readSWC
from utils.network_build_utils import (
    append_completed_sources,
    build_or_load_global_dendrite_index,
    list_swc_files,
    load_completed_sources,
    load_part_paths,
    next_part_index,
    write_parquet_part,
)

# ── per-worker globals (set by _worker_init) ──
_w_tree: cKDTree | None = None
_w_den_pts: np.ndarray | None = None
_w_den_target_idx: np.ndarray | None = None
_w_target_stems: list[str] | None = None


def _worker_init(den_pts: np.ndarray, den_target_idx: np.ndarray, target_stems: list[str]) -> None:
    global _w_tree, _w_den_pts, _w_den_target_idx, _w_target_stems
    _w_den_pts = den_pts
    _w_den_target_idx = den_target_idx
    _w_target_stems = target_stems
    _w_tree = cKDTree(den_pts)


def _worker_process_source(args_tuple: tuple) -> tuple[str, list[dict], int]:
    source_path, radius_euc, radius_sq = args_tuple
    source_path = Path(source_path)
    records, raw_hits = build_pb_records_for_source(
        source_path=source_path,
        tree=_w_tree,
        all_den_pts=_w_den_pts,
        all_den_target_idx=_w_den_target_idx,
        target_stems=_w_target_stems,
        radius_euc=radius_euc,
        radius_sq=radius_sq,
    )
    return source_path.stem, records, raw_hits


RADIUS_EUC = 5
RADIUS_SQ = 25


def _log(message: str, enabled: bool) -> None:
    if enabled:
        print(message)


def _resolve_index_cache_path(index_cache_arg: str, output_dir: Path) -> Path:
    cache_path = Path(index_cache_arg).expanduser()
    if cache_path.is_absolute():
        return cache_path.resolve()
    return (output_dir / cache_path).resolve()


def _min_distance_per_bouton(
    bouton_ids: np.ndarray,
    sqdists: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if len(bouton_ids) == 0:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int32),
        )

    order = np.lexsort((sqdists, bouton_ids))
    bouton_sorted = bouton_ids[order]
    dist_sorted = sqdists[order]
    keep = np.empty(len(bouton_sorted), dtype=bool)
    keep[0] = True
    keep[1:] = bouton_sorted[1:] != bouton_sorted[:-1]
    return bouton_sorted[keep], dist_sorted[keep]


def build_pb_records_for_source(
    source_path: Path,
    tree: cKDTree,
    all_den_pts: np.ndarray,
    all_den_target_idx: np.ndarray,
    target_stems: list[str],
    radius_euc: int,
    radius_sq: int,
) -> tuple[list[dict], int]:
    source_id = source_path.stem
    df = readSWC(str(source_path), use_bouton=True)
    df = df[["x", "y", "z"]].copy()
    df = df.apply(pd.to_numeric, errors="coerce").dropna().astype(np.int32)
    if df.empty:
        return [], 0

    bouton_pts = df.to_numpy(dtype=np.int32, copy=True)
    bouton_local_ids = df.index.to_numpy(dtype=np.int64, copy=False)
    hit_lists = tree.query_ball_point(bouton_pts, r=radius_euc, workers=1)

    bouton_pos_arr: list[np.ndarray] = []
    den_gidx_arr: list[np.ndarray] = []
    for bouton_pos, hit_idx_list in enumerate(hit_lists):
        if not hit_idx_list:
            continue
        n_hit = len(hit_idx_list)
        bouton_pos_arr.append(np.full(n_hit, bouton_pos, dtype=np.int64))
        den_gidx_arr.append(np.asarray(hit_idx_list, dtype=np.int64))

    if not bouton_pos_arr:
        return [], 0

    bouton_pos_arr = np.concatenate(bouton_pos_arr)
    den_gidx_arr = np.concatenate(den_gidx_arr)
    target_arr = all_den_target_idx[den_gidx_arr]

    diff = bouton_pts[bouton_pos_arr].astype(np.int64) - all_den_pts[den_gidx_arr].astype(np.int64)
    sqdists = (diff * diff).sum(axis=1).astype(np.int32)
    valid = sqdists <= radius_sq
    if not np.any(valid):
        return [], 0

    bouton_pos_arr = bouton_pos_arr[valid]
    den_gidx_arr = den_gidx_arr[valid]
    target_arr = target_arr[valid]
    sqdists = sqdists[valid]

    records: list[dict] = []
    raw_hit_count = 0
    for target_idx in np.unique(target_arr):
        target_id = target_stems[int(target_idx)]
        if target_id == source_id:
            continue

        mask = target_arr == target_idx
        raw_hit_count += int(np.count_nonzero(mask))
        target_bouton_ids = bouton_local_ids[bouton_pos_arr[mask]]
        target_sqdists = sqdists[mask]
        uniq_bouton_ids, min_sqdists = _min_distance_per_bouton(
            bouton_ids=target_bouton_ids.astype(np.int64, copy=False),
            sqdists=target_sqdists.astype(np.int32, copy=False),
        )
        if len(uniq_bouton_ids) == 0:
            continue

        coords = df.loc[uniq_bouton_ids, ["x", "y", "z"]].to_numpy(dtype=np.int32, copy=True)
        for bouton_id, coord, dis in zip(uniq_bouton_ids, coords, min_sqdists):
            records.append(
                {
                    "source_cell": source_id,
                    "target_cell": target_id,
                    "bouton_id": int(bouton_id),
                    "x": int(coord[0]),
                    "y": int(coord[1]),
                    "z": int(coord[2]),
                    "dis": int(dis),
                }
            )

    return records, raw_hit_count


def summarize_pb_parts(parts_dir: Path, prefix: str) -> dict:
    part_paths = load_part_paths(parts_dir, prefix)
    total_rows = 0
    unique_pairs = 0
    unique_sources = 0
    targets_per_source: list[int] = []

    for part_path in part_paths:
        df = pd.read_parquet(part_path, columns=["source_cell", "target_cell"])
        if df.empty:
            continue
        total_rows += len(df)
        pair_df = df.drop_duplicates()
        unique_pairs += len(pair_df)
        per_source = pair_df.groupby("source_cell")["target_cell"].nunique()
        unique_sources += len(per_source)
        targets_per_source.extend(per_source.to_list())

    summary = {
        "part_files": len(part_paths),
        "total_pb_rows": int(total_rows),
        "unique_connected_pairs": int(unique_pairs),
        "unique_sources_with_hits": int(unique_sources),
        "mean_targets_per_hit_source": float(np.mean(targets_per_source)) if targets_per_source else 0.0,
        "median_targets_per_hit_source": float(np.median(targets_per_source)) if targets_per_source else 0.0,
    }
    return summary


def cmd_build_pbs(args: argparse.Namespace) -> None:
    source_dir = Path(args.source_dir).expanduser().resolve()
    target_dir = Path(args.target_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    progress_file = output_dir / args.progress_file
    cache_path = _resolve_index_cache_path(args.index_cache, output_dir)
    meta_path = output_dir / "pbs_build_meta.json"

    cache = build_or_load_global_dendrite_index(
        target_dir=target_dir,
        cache_path=cache_path,
        progress_mininterval=args.progress_mininterval,
    )
    all_den_pts = cache["all_den_pts"]
    all_den_target_idx = cache["all_den_target_idx"]
    target_stems = cache["target_stems"]

    source_files = list_swc_files(source_dir)
    completed_sources = load_completed_sources(progress_file)
    pending_sources = [p for p in source_files if p.stem not in completed_sources]

    if args.overwrite_progress:
        progress_file.unlink(missing_ok=True)
        completed_sources = set()
        pending_sources = source_files

    n_workers = min(args.workers, len(pending_sources)) if pending_sources else 1

    meta = {
        "source_dir": str(source_dir),
        "target_dir": str(target_dir),
        "index_cache_path": str(cache_path),
        "radius_euc": args.radius_euc,
        "radius_sq": args.radius_sq,
        "n_targets_indexed": len(target_stems),
        "n_dendrite_points": int(len(all_den_pts)),
        "chunk_sources": args.chunk_sources,
        "part_prefix": args.part_prefix,
        "workers": n_workers,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    _log(f"source files total: {len(source_files)}", enabled=True)
    _log(f"already completed: {len(completed_sources)}", enabled=True)
    _log(f"pending sources: {len(pending_sources)}", enabled=True)
    _log(f"indexed targets: {len(target_stems)}", enabled=True)
    _log(f"indexed dendrite points: {len(all_den_pts):,}", enabled=True)
    _log(f"index cache path: {cache_path}", enabled=True)
    _log(f"parallel workers: {n_workers}", enabled=True)

    part_idx = next_part_index(output_dir, args.part_prefix)
    chunk_records: list[dict] = []
    chunk_source_ids: list[str] = []
    total_raw_hits = 0
    total_new_pb_rows = 0
    processed_source_count = 0

    work_items = [
        (str(sp), args.radius_euc, args.radius_sq)
        for sp in pending_sources
    ]

    with mp.Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(all_den_pts, all_den_target_idx, target_stems),
    ) as pool:
        results_iter = pool.imap_unordered(_worker_process_source, work_items, chunksize=4)
        for source_id, source_records, raw_hit_count in tqdm(
            results_iter,
            total=len(work_items),
            desc="building PB parts",
            mininterval=args.progress_mininterval,
            dynamic_ncols=True,
        ):
            total_raw_hits += raw_hit_count
            total_new_pb_rows += len(source_records)
            processed_source_count += 1
            chunk_records.extend(source_records)
            chunk_source_ids.append(source_id)

            if len(chunk_source_ids) >= args.chunk_sources:
                if chunk_records:
                    part_path = write_parquet_part(
                        records=chunk_records,
                        output_dir=output_dir,
                        prefix=args.part_prefix,
                        part_idx=part_idx,
                    )
                    _log(
                        f"wrote {len(chunk_records):,} PB rows -> {part_path.name}",
                        enabled=args.log_chunks,
                    )
                    part_idx += 1
                append_completed_sources(progress_file, chunk_source_ids)
                chunk_records = []
                chunk_source_ids = []

    if chunk_source_ids:
        if chunk_records:
            part_path = write_parquet_part(
                records=chunk_records,
                output_dir=output_dir,
                prefix=args.part_prefix,
                part_idx=part_idx,
            )
            _log(
                f"wrote {len(chunk_records):,} PB rows -> {part_path.name}",
                enabled=args.log_chunks,
            )
        append_completed_sources(progress_file, chunk_source_ids)

    summary = summarize_pb_parts(output_dir, args.part_prefix)
    summary["current_run_processed_sources"] = int(processed_source_count)
    summary["current_run_raw_hits_before_pb_compression"] = int(total_raw_hits)
    summary["current_run_pb_rows_written"] = int(total_new_pb_rows)
    summary["current_run_pb_compression_ratio_raw_hits_to_pb_rows"] = (
        float(total_raw_hits / total_new_pb_rows)
        if total_new_pb_rows > 0
        else 0.0
    )
    summary_txt_path = output_dir / "pbs_summary.txt"
    summary_txt_lines = [
        "PB build summary",
        f"current_run_processed_sources: {summary['current_run_processed_sources']}",
        f"current_run_raw_hits_before_pb_compression: {summary['current_run_raw_hits_before_pb_compression']:,}",
        f"current_run_pb_rows_written: {summary['current_run_pb_rows_written']:,}",
        f"current_run_pb_compression_ratio_raw_hits_to_pb_rows: {summary['current_run_pb_compression_ratio_raw_hits_to_pb_rows']:.4f}",
        f"part_files: {summary['part_files']}",
        f"total_pb_rows_across_all_parts: {summary['total_pb_rows']:,}",
        f"unique_connected_source_target_pairs: {summary['unique_connected_pairs']:,}",
        f"unique_sources_with_hits: {summary['unique_sources_with_hits']:,}",
        f"mean_targets_per_hit_source: {summary['mean_targets_per_hit_source']:.2f}",
        f"median_targets_per_hit_source: {summary['median_targets_per_hit_source']:.2f}",
    ]
    summary_txt_path.write_text("\n".join(summary_txt_lines) + "\n", encoding="utf-8")

    print("PB build complete.")
    print(f"current run processed sources: {summary['current_run_processed_sources']:,}")
    print(f"current run raw bouton-dendrite hits before compression: {summary['current_run_raw_hits_before_pb_compression']:,}")
    print(f"current run PB rows written: {summary['current_run_pb_rows_written']:,}")
    print(
        "current run PB compression ratio (raw hits / PB rows): "
        f"{summary['current_run_pb_compression_ratio_raw_hits_to_pb_rows']:.4f}"
    )
    print(f"part files: {summary['part_files']}")
    print(f"total PB rows across all parts: {summary['total_pb_rows']:,}")
    print(f"unique connected source-target pairs: {summary['unique_connected_pairs']:,}")
    print(f"unique sources with hits: {summary['unique_sources_with_hits']:,}")
    print(f"mean targets per hit source: {summary['mean_targets_per_hit_source']:.2f}")
    print(f"median targets per hit source: {summary['median_targets_per_hit_source']:.2f}")
    print(f"saved summary -> {summary_txt_path}")

def cmd_build_sparse_matrix(args: argparse.Namespace) -> None:
    parts_dir = Path(args.parts_dir).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    part_paths = load_part_paths(parts_dir, args.part_prefix)
    if not part_paths:
        raise FileNotFoundError(f"No parquet parts found under {parts_dir} with prefix {args.part_prefix}")

    grouped_frames: list[pd.DataFrame] = []
    for part_path in tqdm(
        part_paths,
        desc="aggregating PB parts",
        mininterval=args.progress_mininterval,
        dynamic_ncols=True,
    ):
        df = pd.read_parquet(part_path)
        if df.empty:
            continue
        df["weight"] = np.exp(-(df["dis"].to_numpy(dtype=np.float64)) / float(args.weight_scale))
        grouped = (
            df.groupby(["source_cell", "target_cell"], as_index=False)
            .agg(
                score=("weight", "sum"),
                hit_count=("dis", "size"),
                min_dis=("dis", "min"),
            )
        )
        grouped_frames.append(grouped)

    if not grouped_frames:
        sparse_df = pd.DataFrame(columns=["source_cell", "target_cell", "score", "hit_count", "min_dis"])
    else:
        sparse_df = pd.concat(grouped_frames, axis=0, ignore_index=True)
        sparse_df = (
            sparse_df.groupby(["source_cell", "target_cell"], as_index=False)
            .agg(
                score=("score", "sum"),
                hit_count=("hit_count", "sum"),
                min_dis=("min_dis", "min"),
            )
        )

    sparse_df.to_parquet(output_path, index=False)
    print(f"saved sparse bouton connectivity -> {output_path}")
    print(f"rows: {len(sparse_df):,}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Directly build PB parquet parts with a global dendrite index."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_pbs = subparsers.add_parser("build-pbs", help="Generate PBs_part_*.parquet directly from SWC files.")
    build_pbs.add_argument("--source-dir", required=True, help="Directory containing source bouton .swc files.")
    build_pbs.add_argument("--target-dir", required=True, help="Directory containing target dendrite .swc files.")
    build_pbs.add_argument("--output-dir", required=True, help="Directory for parquet parts and progress files.")
    build_pbs.add_argument("--radius-euc", type=int, default=RADIUS_EUC, help="Euclidean search radius.")
    build_pbs.add_argument("--radius-sq", type=int, default=RADIUS_SQ, help="Squared-distance threshold.")
    build_pbs.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1),
                           help="Number of parallel worker processes (default: min(96, cpu_count)).")
    build_pbs.add_argument("--chunk-sources", type=int, default=32, help="How many sources to buffer per parquet part.")
    build_pbs.add_argument("--part-prefix", default="PBs", help="Prefix for output part files.")
    build_pbs.add_argument("--progress-file", default="source_progress.txt", help="Resume state file name.")
    build_pbs.add_argument(
        "--index-cache",
        default="global_dendrite_index.pkl",
        help=(
            "Cached global dendrite index file. Relative paths are resolved under "
            "--output-dir; absolute paths can be shared across runs."
        ),
    )
    build_pbs.add_argument(
        "--progress-mininterval",
        type=float,
        default=5.0,
        help="Minimum seconds between tqdm refreshes.",
    )
    build_pbs.add_argument(
        "--log-chunks",
        action="store_true",
        help="Print one line whenever a parquet part is written.",
    )
    build_pbs.add_argument(
        "--overwrite-progress",
        action="store_true",
        help="Ignore existing progress and restart from the full source list.",
    )

    build_sparse = subparsers.add_parser(
        "build-sparse-matrix",
        help="Aggregate PB parquet parts into a sparse pair-level connectivity parquet.",
    )
    build_sparse.add_argument("--parts-dir", required=True, help="Directory containing PBs_part_*.parquet.")
    build_sparse.add_argument("--output-path", required=True, help="Output sparse parquet path.")
    build_sparse.add_argument("--part-prefix", default="PBs", help="Prefix used by the parquet parts.")
    build_sparse.add_argument(
        "--progress-mininterval",
        type=float,
        default=5.0,
        help="Minimum seconds between tqdm refreshes.",
    )
    build_sparse.add_argument(
        "--weight-scale",
        type=float,
        default=25.0,
        help="Scale used in exp(-dis / weight_scale).",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "build-pbs":
        cmd_build_pbs(args)
    elif args.command == "build-sparse-matrix":
        cmd_build_sparse_matrix(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
