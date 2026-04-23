from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import klb_codec as klb
from benchmark_logging import BenchmarkLogger


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark Luxendo folder compression with timestamped JSON logs.")
    parser.add_argument("--source", type=Path, default=_default_source_dir(), help="Source experiment folder to compress.")
    parser.add_argument("--output-root", type=Path, default=_default_output_root(), help="Root folder for unique benchmark outputs.")
    parser.add_argument("--log-root", type=Path, default=_default_log_root(), help="Root folder for unique benchmark logs.")
    parser.add_argument("--compression", choices=sorted(klb.COMPRESSION_CODES), default="bzip2", help="Compression backend.")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1), help="Worker count for block compression.")
    parser.add_argument("--parallel-files", type=int, default=0, help="How many Luxendo files to compress concurrently. Use 0 for auto.")
    parser.add_argument("--block-x", type=int, default=128, help="KLB block size in x.")
    parser.add_argument("--block-y", type=int, default=128, help="KLB block size in y.")
    parser.add_argument("--block-z", type=int, default=16, help="KLB block size in z.")
    parser.add_argument("--label", default="folder_compression", help="Short label used in the log folder name.")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    source_dir = args.source.resolve()
    output_root = args.output_root.resolve()
    log_root = args.log_root.resolve()
    if not source_dir.is_dir():
        raise SystemExit(f"Source directory does not exist: {source_dir}")

    source_stats = _source_summary(source_dir)
    logger = BenchmarkLogger(
        log_root,
        run_name=args.label,
        metadata={
            "source_dir": source_dir,
            "source_stats": source_stats,
            "compression": args.compression,
            "workers": args.workers,
            "parallel_files": args.parallel_files,
            "block_size_xyz": [args.block_x, args.block_y, args.block_z],
        },
    )
    output_dir = output_root / logger.run_id / f"{source_dir.name}_klb_bundle"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    logger.record_point(
        "benchmark_folder_compression.configuration",
        source_dir=source_dir,
        output_dir=output_dir,
        log_dir=logger.run_dir,
        compression=args.compression,
        workers=args.workers,
        parallel_files=args.parallel_files,
        block_size=[args.block_x, args.block_y, args.block_z, 1, 1],
        source_stats=source_stats,
    )

    status = "ok"
    result: klb.DirectoryBatchResult | None = None
    try:
        result = klb.compress_directory_to_klb_bundle(
            source_dir,
            output_dir,
            block_size=(args.block_x, args.block_y, args.block_z, 1, 1),
            compression=args.compression,
            workers=args.workers,
            parallel_files=args.parallel_files,
            archive_other_files=True,
            benchmark_logger=logger,
        )
    except Exception as exc:
        status = "error"
        logger.record_point(
            "benchmark_folder_compression.error",
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
        print(f"Benchmark failed. Logs: {logger.run_dir}", file=sys.stderr)
        raise
    else:
        logger.close(
            status=status,
            extra_summary={
                "source_dir": source_dir,
                "output_dir": output_dir,
                "log_dir": logger.run_dir,
                "luxendo_files": result.luxendo_files,
                "klb_files": result.klb_files,
                "archived_files": result.archived_files,
                "archived_dirs": result.archived_dirs,
                "archive_path": result.archive_path,
            },
        )

    summary = json.loads(logger.summary_path.read_text(encoding="utf-8"))
    print(f"Run ID: {logger.run_id}")
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Logs: {logger.run_dir}")
    print(f"Events: {logger.events_path}")
    print(f"Summary: {logger.summary_path}")
    print(f"Total duration: {summary['total_duration_s']:.3f}s")
    print("Top phases:")
    for item in summary.get("span_totals", [])[:10]:
        print(
            f"  {item['name']}: total={item['total_duration_s']:.3f}s "
            f"count={item['count']} avg={item['avg_duration_s']:.3f}s max={item['max_duration_s']:.3f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
