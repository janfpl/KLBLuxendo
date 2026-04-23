from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import tifffile


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_name(value: str) -> str:
    return value.replace("/", "__").replace("\\", "__")


def _to_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", "replace")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _to_text(value.item())
    return str(value)


def _find_hdf5_layers(path: Path) -> list[str]:
    found: list[str] = []
    with h5py.File(path, "r") as handle:
        def visit(name: str, obj: object) -> None:
            if not isinstance(obj, h5py.Dataset):
                return
            leaf = name.rsplit("/", 1)[-1]
            if leaf == "metadata" or obj.ndim < 3:
                return
            if leaf == "Data" or leaf.startswith("Data_"):
                found.append(name)

        handle.visititems(visit)
    found.sort(key=lambda item: (item != "Data", item))
    return found


def _read_metadata_text(handle: h5py.File) -> str:
    dataset = handle.get("metadata")
    if not isinstance(dataset, h5py.Dataset):
        return ""
    return _to_text(dataset[()])


def _iter_z_ranges(shape: tuple[int, ...], chunk_depth: int) -> list[tuple[int, int]]:
    z_dim = int(shape[0])
    depth = max(1, min(chunk_depth, z_dim))
    return [(z0, min(z0 + depth, z_dim)) for z0 in range(0, z_dim, depth)]


@dataclass
class VolumeMetrics:
    voxel_count: int
    differing_voxels: int
    differing_fraction: float
    max_abs_diff: float
    mean_abs_diff: float
    rmse: float
    original_min: float
    original_max: float
    original_mean: float
    original_std: float
    restored_min: float
    restored_max: float
    restored_mean: float
    restored_std: float
    exact_equal: bool


@dataclass
class TiffMetrics:
    pixel_count: int
    differing_pixels: int
    differing_fraction: float
    max_abs_diff: float
    mean_abs_diff: float
    rmse: float
    exact_equal: bool


def _volume_dtype_for_diff(dtype: np.dtype) -> np.dtype:
    if np.issubdtype(dtype, np.integer):
        return np.int64
    return np.float64


def _compute_tiff_metrics(original: np.ndarray, restored: np.ndarray) -> TiffMetrics:
    diff = restored.astype(np.float64) - original.astype(np.float64)
    abs_diff = np.abs(diff)
    differing = int(np.count_nonzero(abs_diff))
    pixel_count = int(original.size)
    return TiffMetrics(
        pixel_count=pixel_count,
        differing_pixels=differing,
        differing_fraction=(differing / pixel_count) if pixel_count else 0.0,
        max_abs_diff=float(abs_diff.max(initial=0.0)),
        mean_abs_diff=float(abs_diff.mean()) if pixel_count else 0.0,
        rmse=float(np.sqrt(np.mean(diff * diff))) if pixel_count else 0.0,
        exact_equal=bool(np.array_equal(original, restored)),
    )


def _compare_dataset(
    original_path: Path,
    restored_path: Path,
    dataset_path: str,
    dataset_output_dir: Path,
) -> dict[str, Any]:
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(original_path, "r") as original_h5, h5py.File(restored_path, "r") as restored_h5:
        original_ds = original_h5[dataset_path]
        restored_ds = restored_h5[dataset_path]
        if not isinstance(original_ds, h5py.Dataset) or not isinstance(restored_ds, h5py.Dataset):
            raise ValueError(f"{dataset_path} is not a dataset in both files")
        if tuple(original_ds.shape) != tuple(restored_ds.shape):
            raise ValueError(f"Shape mismatch for {dataset_path}: {original_ds.shape} vs {restored_ds.shape}")
        if str(original_ds.dtype) != str(restored_ds.dtype):
            raise ValueError(f"Dtype mismatch for {dataset_path}: {original_ds.dtype} vs {restored_ds.dtype}")
        if original_ds.ndim != 3:
            raise ValueError(f"Expected 3D Luxendo layer for {dataset_path}, found ndim={original_ds.ndim}")

        dtype = np.dtype(original_ds.dtype)
        diff_dtype = _volume_dtype_for_diff(dtype)
        z_ranges = _iter_z_ranges(tuple(int(v) for v in original_ds.shape), int(original_ds.chunks[0]) if original_ds.chunks else 16)

        original_mip: np.ndarray | None = None
        restored_mip: np.ndarray | None = None

        voxel_count = 0
        differing_voxels = 0
        sum_abs_diff = 0.0
        sum_sq_diff = 0.0
        original_sum = 0.0
        original_sum_sq = 0.0
        restored_sum = 0.0
        restored_sum_sq = 0.0
        original_min = math.inf
        original_max = -math.inf
        restored_min = math.inf
        restored_max = -math.inf
        max_abs_diff = 0.0

        for z0, z1 in z_ranges:
            original_chunk = np.asarray(original_ds[z0:z1, :, :])
            restored_chunk = np.asarray(restored_ds[z0:z1, :, :])

            chunk_diff = restored_chunk.astype(diff_dtype) - original_chunk.astype(diff_dtype)
            chunk_abs_diff = np.abs(chunk_diff, dtype=np.float64)

            voxel_count += int(original_chunk.size)
            differing_voxels += int(np.count_nonzero(chunk_abs_diff))
            sum_abs_diff += float(chunk_abs_diff.sum(dtype=np.float64))
            sum_sq_diff += float(np.square(chunk_diff, dtype=np.float64).sum(dtype=np.float64))

            original_f64 = original_chunk.astype(np.float64)
            restored_f64 = restored_chunk.astype(np.float64)
            original_sum += float(original_f64.sum(dtype=np.float64))
            original_sum_sq += float(np.square(original_f64, dtype=np.float64).sum(dtype=np.float64))
            restored_sum += float(restored_f64.sum(dtype=np.float64))
            restored_sum_sq += float(np.square(restored_f64, dtype=np.float64).sum(dtype=np.float64))

            original_min = min(original_min, float(original_chunk.min(initial=0)))
            original_max = max(original_max, float(original_chunk.max(initial=0)))
            restored_min = min(restored_min, float(restored_chunk.min(initial=0)))
            restored_max = max(restored_max, float(restored_chunk.max(initial=0)))
            max_abs_diff = max(max_abs_diff, float(chunk_abs_diff.max(initial=0.0)))

            chunk_original_mip = original_chunk.max(axis=0)
            chunk_restored_mip = restored_chunk.max(axis=0)
            if original_mip is None:
                original_mip = chunk_original_mip.copy()
                restored_mip = chunk_restored_mip.copy()
            else:
                np.maximum(original_mip, chunk_original_mip, out=original_mip)
                np.maximum(restored_mip, chunk_restored_mip, out=restored_mip)

        assert original_mip is not None and restored_mip is not None

        abs_diff_mip = np.abs(restored_mip.astype(np.int64) - original_mip.astype(np.int64)).astype(np.uint16)
        if int(abs_diff_mip.max(initial=0)) > 0:
            scaled_abs_diff_mip = np.round(abs_diff_mip.astype(np.float64) / float(abs_diff_mip.max()) * 65535.0).astype(np.uint16)
        else:
            scaled_abs_diff_mip = np.zeros_like(abs_diff_mip, dtype=np.uint16)

        original_mip_path = dataset_output_dir / f"{_safe_name(dataset_path)}__original_zmax.tiff"
        restored_mip_path = dataset_output_dir / f"{_safe_name(dataset_path)}__restored_zmax.tiff"
        abs_diff_mip_path = dataset_output_dir / f"{_safe_name(dataset_path)}__absdiff_zmax.tiff"
        scaled_abs_diff_mip_path = dataset_output_dir / f"{_safe_name(dataset_path)}__absdiff_zmax_scaled.tiff"

        tifffile.imwrite(original_mip_path, original_mip)
        tifffile.imwrite(restored_mip_path, restored_mip)
        tifffile.imwrite(abs_diff_mip_path, abs_diff_mip)
        tifffile.imwrite(scaled_abs_diff_mip_path, scaled_abs_diff_mip)

        original_mip_disk = tifffile.imread(original_mip_path)
        restored_mip_disk = tifffile.imread(restored_mip_path)
        mip_metrics = _compute_tiff_metrics(original_mip_disk, restored_mip_disk)

        original_mean = original_sum / voxel_count if voxel_count else 0.0
        restored_mean = restored_sum / voxel_count if voxel_count else 0.0
        original_var = max(0.0, (original_sum_sq / voxel_count) - (original_mean * original_mean)) if voxel_count else 0.0
        restored_var = max(0.0, (restored_sum_sq / voxel_count) - (restored_mean * restored_mean)) if voxel_count else 0.0

        volume_metrics = VolumeMetrics(
            voxel_count=voxel_count,
            differing_voxels=differing_voxels,
            differing_fraction=(differing_voxels / voxel_count) if voxel_count else 0.0,
            max_abs_diff=max_abs_diff,
            mean_abs_diff=(sum_abs_diff / voxel_count) if voxel_count else 0.0,
            rmse=(math.sqrt(sum_sq_diff / voxel_count) if voxel_count else 0.0),
            original_min=original_min if original_min is not math.inf else 0.0,
            original_max=original_max if original_max is not -math.inf else 0.0,
            original_mean=original_mean,
            original_std=math.sqrt(original_var),
            restored_min=restored_min if restored_min is not math.inf else 0.0,
            restored_max=restored_max if restored_max is not -math.inf else 0.0,
            restored_mean=restored_mean,
            restored_std=math.sqrt(restored_var),
            exact_equal=(differing_voxels == 0),
        )

        return {
            "dataset_path": dataset_path,
            "shape": [int(v) for v in original_ds.shape],
            "dtype": str(original_ds.dtype),
            "original_chunks": list(original_ds.chunks) if original_ds.chunks else None,
            "restored_chunks": list(restored_ds.chunks) if restored_ds.chunks else None,
            "original_storage_bytes": int(original_ds.id.get_storage_size()),
            "restored_storage_bytes": int(restored_ds.id.get_storage_size()),
            "volume_metrics": asdict(volume_metrics),
            "mip_metrics": asdict(mip_metrics),
            "mip_paths": {
                "original": str(original_mip_path),
                "restored": str(restored_mip_path),
                "abs_diff": str(abs_diff_mip_path),
                "abs_diff_scaled": str(scaled_abs_diff_mip_path),
            },
        }


def _compare_small_files(original_root: Path, restored_root: Path) -> list[dict[str, Any]]:
    original_files = {path.relative_to(original_root).as_posix(): path for path in original_root.rglob("*") if path.is_file() and not path.name.lower().endswith(".lux.h5")}
    restored_files = {path.relative_to(restored_root).as_posix(): path for path in restored_root.rglob("*") if path.is_file() and not path.name.lower().endswith(".lux.h5")}
    rel_paths = sorted(set(original_files) | set(restored_files))
    results = []
    for rel_path in rel_paths:
        original_path = original_files.get(rel_path)
        restored_path = restored_files.get(rel_path)
        same = False
        original_size = original_path.stat().st_size if original_path else None
        restored_size = restored_path.stat().st_size if restored_path else None
        if original_path and restored_path and original_size == restored_size:
            same = original_path.read_bytes() == restored_path.read_bytes()
        results.append(
            {
                "relative_path": rel_path,
                "original_exists": original_path is not None,
                "restored_exists": restored_path is not None,
                "original_size": original_size,
                "restored_size": restored_size,
                "byte_equal": same,
            }
        )
    return results


def _matching_lux_files(original_root: Path, restored_root: Path) -> list[tuple[str, Path, Path]]:
    original_files = {path.relative_to(original_root).as_posix(): path for path in original_root.rglob("*.lux.h5")}
    restored_files = {path.relative_to(restored_root).as_posix(): path for path in restored_root.rglob("*.lux.h5")}
    rel_paths = sorted(set(original_files) & set(restored_files))
    return [(rel_path, original_files[rel_path], restored_files[rel_path]) for rel_path in rel_paths]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare original and restored Luxendo HDF5 files by layer and Z-MIP TIFFs.")
    parser.add_argument("--original-root", type=Path, required=True, help="Directory with the original Luxendo files.")
    parser.add_argument("--restored-root", type=Path, required=True, help="Directory with the restored Luxendo files.")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[2] / "analysis_outputs", help="Root directory for the analysis output.")
    parser.add_argument("--label", default="luxendo_compare", help="Short label for the output folder name.")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    original_root = args.original_root.resolve()
    restored_root = args.restored_root.resolve()
    if not original_root.is_dir():
        raise SystemExit(f"Original root does not exist: {original_root}")
    if not restored_root.is_dir():
        raise SystemExit(f"Restored root does not exist: {restored_root}")

    run_dir = args.output_root.resolve() / f"{_safe_name(args.label)}_{_timestamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    tiff_root = run_dir / "tiffs"
    tiff_root.mkdir(parents=True, exist_ok=True)

    file_results: list[dict[str, Any]] = []
    for relative_path, original_path, restored_path in _matching_lux_files(original_root, restored_root):
        original_layers = _find_hdf5_layers(original_path)
        restored_layers = _find_hdf5_layers(restored_path)
        if original_layers != restored_layers:
            raise ValueError(f"Layer mismatch for {relative_path}: {original_layers} vs {restored_layers}")

        with h5py.File(original_path, "r") as original_h5, h5py.File(restored_path, "r") as restored_h5:
            original_metadata = _read_metadata_text(original_h5)
            restored_metadata = _read_metadata_text(restored_h5)

        layer_results = []
        file_output_dir = tiff_root / _safe_name(relative_path.removesuffix(".lux.h5"))
        for dataset_path in original_layers:
            layer_results.append(_compare_dataset(original_path, restored_path, dataset_path, file_output_dir))

        file_results.append(
            {
                "relative_path": relative_path,
                "original_path": str(original_path),
                "restored_path": str(restored_path),
                "original_file_size": original_path.stat().st_size,
                "restored_file_size": restored_path.stat().st_size,
                "file_size_delta": restored_path.stat().st_size - original_path.stat().st_size,
                "metadata_equal": original_metadata == restored_metadata,
                "metadata_length_original": len(original_metadata),
                "metadata_length_restored": len(restored_metadata),
                "layers": layer_results,
            }
        )

    small_file_results = _compare_small_files(original_root, restored_root)
    summary = {
        "original_root": str(original_root),
        "restored_root": str(restored_root),
        "output_dir": str(run_dir),
        "luxendo_files": file_results,
        "small_files": small_file_results,
    }

    summary_path = run_dir / "comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    csv_path = run_dir / "dataset_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "relative_path",
                "dataset_path",
                "shape",
                "dtype",
                "original_file_size",
                "restored_file_size",
                "file_size_delta",
                "original_chunks",
                "restored_chunks",
                "volume_exact_equal",
                "volume_differing_voxels",
                "volume_differing_fraction",
                "volume_max_abs_diff",
                "volume_mean_abs_diff",
                "volume_rmse",
                "mip_exact_equal",
                "mip_differing_pixels",
                "mip_differing_fraction",
                "mip_max_abs_diff",
                "mip_mean_abs_diff",
                "mip_rmse",
                "original_storage_bytes",
                "restored_storage_bytes",
            ],
        )
        writer.writeheader()
        for file_result in file_results:
            for layer in file_result["layers"]:
                writer.writerow(
                    {
                        "relative_path": file_result["relative_path"],
                        "dataset_path": layer["dataset_path"],
                        "shape": "x".join(str(v) for v in layer["shape"]),
                        "dtype": layer["dtype"],
                        "original_file_size": file_result["original_file_size"],
                        "restored_file_size": file_result["restored_file_size"],
                        "file_size_delta": file_result["file_size_delta"],
                        "original_chunks": layer["original_chunks"],
                        "restored_chunks": layer["restored_chunks"],
                        "volume_exact_equal": layer["volume_metrics"]["exact_equal"],
                        "volume_differing_voxels": layer["volume_metrics"]["differing_voxels"],
                        "volume_differing_fraction": layer["volume_metrics"]["differing_fraction"],
                        "volume_max_abs_diff": layer["volume_metrics"]["max_abs_diff"],
                        "volume_mean_abs_diff": layer["volume_metrics"]["mean_abs_diff"],
                        "volume_rmse": layer["volume_metrics"]["rmse"],
                        "mip_exact_equal": layer["mip_metrics"]["exact_equal"],
                        "mip_differing_pixels": layer["mip_metrics"]["differing_pixels"],
                        "mip_differing_fraction": layer["mip_metrics"]["differing_fraction"],
                        "mip_max_abs_diff": layer["mip_metrics"]["max_abs_diff"],
                        "mip_mean_abs_diff": layer["mip_metrics"]["mean_abs_diff"],
                        "mip_rmse": layer["mip_metrics"]["rmse"],
                        "original_storage_bytes": layer["original_storage_bytes"],
                        "restored_storage_bytes": layer["restored_storage_bytes"],
                    }
                )

    print(f"Analysis written to: {run_dir}")
    print(f"Summary JSON: {summary_path}")
    print(f"Dataset CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
