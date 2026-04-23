from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable

HERE = Path(__file__).resolve().parent

import klb_codec as klb
from benchmark_logging import BenchmarkLogger


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def _default_source_dir() -> Path:
    return HERE.parent.parent / "testdata" / "20220514-153033_Task_1_ovary_sample2_C"


def _default_output_root() -> Path:
    return HERE.parent.parent / "benchmark_outputs"


def _default_log_root() -> Path:
    return HERE.parent.parent / "benchmark_logs"


def _source_summary(source_dir: Path) -> dict[str, int]:
    file_count = 0
    total_bytes = 0
    lux_file_count = 0
    for path in source_dir.rglob("*"):
        if not path.is_file():
            continue
        file_count += 1
        total_bytes += path.stat().st_size
        if path.name.lower().endswith(".lux.h5"):
            lux_file_count += 1
    return {
        "file_count": file_count,
        "total_bytes": total_bytes,
        "luxendo_files": lux_file_count,
    }


def _parse_int_list(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Expected a comma-separated list of integers")
    try:
        return [int(item) for item in items]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _dir_size_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    return total


def _run_one(
    *,
    source_dir: Path,
    output_root: Path,
    log_root: Path,
    compression: str,
    workers: int,
    parallel_files: int,
    block_size: tuple[int, int, int, int, int],
    cleanup_output: bool,
    source_stats: dict[str, int],
) -> dict:
    run_name = f"w{workers}f{parallel_files}"
    logger = BenchmarkLogger(
        log_root,
        run_name=run_name,
        metadata={
            "source_dir": source_dir,
            "source_stats": source_stats,
            "compression": compression,
            "workers": workers,
            "parallel_files": parallel_files,
            "block_size_xyz": list(block_size[:3]),
        },
    )
    output_dir = output_root / logger.run_id / f"{source_dir.name}_klb_bundle"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    logger.record_point(
        "benchmark_sweep.configuration",
        source_dir=source_dir,
        output_dir=output_dir,
        log_dir=logger.run_dir,
        compression=compression,
        workers=workers,
        parallel_files=parallel_files,
        block_size=list(block_size),
        source_stats=source_stats,
    )

    status = "ok"
    output_deleted = False
    output_bytes = 0
    result: klb.DirectoryBatchResult | None = None
    try:
        result = klb.compress_directory_to_klb_bundle(
            source_dir,
            output_dir,
            block_size=block_size,
            compression=compression,
            workers=workers,
            parallel_files=parallel_files,
            archive_other_files=True,
            benchmark_logger=logger,
        )
        output_bytes = _dir_size_bytes(output_dir)
    except Exception as exc:
        status = "error"
        logger.record_point(
            "benchmark_sweep.error",
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        logger.close(
            status=status,
            extra_summary={
                "source_dir": source_dir,
                "output_dir": output_dir,
                "log_dir": logger.run_dir,
            },
        )
        raise
    else:
        if cleanup_output:
            shutil.rmtree(output_dir.parent)
            output_deleted = True
        logger.close(
            status=status,
            extra_summary={
                "source_dir": source_dir,
                "output_dir": output_dir,
                "log_dir": logger.run_dir,
                "luxendo_files": result.luxendo_files if result else None,
                "klb_files": result.klb_files if result else None,
                "archived_files": result.archived_files if result else None,
                "archived_dirs": result.archived_dirs if result else None,
                "archive_path": result.archive_path if result else None,
                "output_bytes": output_bytes,
                "output_deleted": output_deleted,
            },
        )

    summary = json.loads(logger.summary_path.read_text(encoding="utf-8"))
    return {
        "run_id": logger.run_id,
        "workers": workers,
        "parallel_files": parallel_files,
        "total_duration_s": summary["total_duration_s"],
        "summary_path": str(logger.summary_path),
        "log_dir": str(logger.run_dir),
        "output_dir": str(output_dir),
        "output_bytes": output_bytes,
        "output_deleted": output_deleted,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep worker/file-parallelism settings for Luxendo folder compression.")
    parser.add_argument("--source", type=Path, default=_default_source_dir(), help="Source experiment folder to compress.")
    parser.add_argument("--output-root", type=Path, default=_default_output_root(), help="Root folder for per-run outputs.")
    parser.add_argument("--log-root", type=Path, default=_default_log_root(), help="Root folder for per-run logs.")
    parser.add_argument("--compression", choices=sorted(klb.COMPRESSION_CODES), default="bzip2", help="Compression backend.")
    parser.add_argument("--workers-list", type=_parse_int_list, default=[16, 24, 32, 48], help="Comma-separated worker counts.")
    parser.add_argument("--parallel-files-list", type=_parse_int_list, default=[1, 2], help="Comma-separated file parallelism counts.")
    parser.add_argument("--block-x", type=int, default=96, help="KLB block size in x.")
    parser.add_argument("--block-y", type=int, default=96, help="KLB block size in y.")
    parser.add_argument("--block-z", type=int, default=8, help="KLB block size in z.")
    parser.add_argument("--label", default="sweep", help="Short label for the sweep summary folder.")
    parser.add_argument("--keep-outputs", action="store_true", help="Keep output bundles for each run instead of deleting them after success.")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    source_dir = args.source.resolve()
    if not source_dir.is_dir():
        raise SystemExit(f"Source directory does not exist: {source_dir}")

    sweep_dir = HERE.parent.parent / "benchmark_sweeps" / f"{args.label}_{_timestamp()}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    output_root = args.output_root.resolve()
    log_root = args.log_root.resolve()
    block_size = (args.block_x, args.block_y, args.block_z, 1, 1)
    source_stats = _source_summary(source_dir)

    results: list[dict] = []
    combos: list[tuple[int, int]] = []
    for workers in args.workers_list:
        for parallel_files in args.parallel_files_list:
            combos.append((workers, parallel_files))

    print(f"Sweep directory: {sweep_dir}")
    print(f"Testing {len(combos)} configurations")
    for index, (workers, parallel_files) in enumerate(combos, start=1):
        print(f"[{index}/{len(combos)}] workers={workers} parallel_files={parallel_files}")
        result = _run_one(
            source_dir=source_dir,
            output_root=output_root,
            log_root=log_root,
            compression=args.compression,
            workers=workers,
            parallel_files=parallel_files,
            block_size=block_size,
            cleanup_output=not args.keep_outputs,
            source_stats=source_stats,
        )
        results.append(result)
        print(f"  total_duration_s={result['total_duration_s']:.3f}")

    results.sort(key=lambda item: item["total_duration_s"])
    summary = {
        "source_dir": str(source_dir),
        "compression": args.compression,
        "block_size": list(block_size),
        "workers_list": args.workers_list,
        "parallel_files_list": args.parallel_files_list,
        "results": results,
        "best": results[0] if results else None,
    }
    summary_path = sweep_dir / "sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    csv_path = sweep_dir / "sweep_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_id",
                "workers",
                "parallel_files",
                "total_duration_s",
                "output_bytes",
                "output_deleted",
                "summary_path",
                "log_dir",
                "output_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"Summary JSON: {summary_path}")
    print(f"Summary CSV: {csv_path}")
    print("Best configuration:")
    if results:
        best = results[0]
        print(
            f"  workers={best['workers']} parallel_files={best['parallel_files']} "
            f"total_duration_s={best['total_duration_s']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
