"""Pure-Python KLB reader/writer helpers for the lightweight GUI."""

from __future__ import annotations

import bz2
import json
import math
import os
import struct
import threading
import time
import zipfile
import zlib
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

import h5py
import numpy as np

from benchmark_logging import BenchmarkLogger

KLB_DATA_DIMS = 5
KLB_METADATA_SIZE = 256
KLB_DEFAULT_HEADER_VERSION = 2
KLB_HEADER_FORMAT = "<B5I5fBB256s5I"
KLB_HEADER_FIXED_SIZE = struct.calcsize(KLB_HEADER_FORMAT)
KLB_BUNDLE_MANIFEST = "klb_bundle_manifest.json"
EXACT_LUXENDO_ARCHIVE = "exact_luxendo_files.zip"

ProgressCallback = Callable[[int, int, str], None]


def _benchmark_span(
    benchmark_logger: BenchmarkLogger | None,
    name: str,
    **fields,
):
    return benchmark_logger.span(name, **fields) if benchmark_logger else nullcontext()


@dataclass(frozen=True)
class KlbDataType:
    code: int
    name: str
    numpy_name: str


KLB_TYPES: tuple[KlbDataType, ...] = (
    KlbDataType(0, "uint8", "uint8"),
    KlbDataType(1, "uint16", "uint16"),
    KlbDataType(2, "uint32", "uint32"),
    KlbDataType(3, "uint64", "uint64"),
    KlbDataType(4, "int8", "int8"),
    KlbDataType(5, "int16", "int16"),
    KlbDataType(6, "int32", "int32"),
    KlbDataType(7, "int64", "int64"),
    KlbDataType(8, "float32", "float32"),
    KlbDataType(9, "float64", "float64"),
)
KLB_TYPES_BY_CODE = {item.code: item for item in KLB_TYPES}
KLB_TYPES_BY_NAME = {item.name: item for item in KLB_TYPES}

COMPRESSION_CODES = {"none": 0, "bzip2": 1, "zlib": 2}
COMPRESSION_NAMES = {value: key for key, value in COMPRESSION_CODES.items()}


@dataclass(frozen=True)
class Hdf5DatasetInfo:
    path: str
    shape: tuple[int, ...]
    dtype: str
    chunks: tuple[int, ...] | None
    compression: str | None

    def label(self) -> str:
        chunks = "contiguous" if self.chunks is None else f"chunks={self.chunks}"
        compression = "uncompressed" if self.compression is None else self.compression
        return f"{self.path}  shape={self.shape}  dtype={self.dtype}  {chunks}  {compression}"


@dataclass(frozen=True)
class KlbHeader:
    header_version: int
    xyzct: tuple[int, int, int, int, int]
    pixel_size: tuple[float, float, float, float, float]
    data_type: int
    compression_type: int
    metadata: bytes
    block_size: tuple[int, int, int, int, int]
    block_offsets: tuple[int, ...]

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(KLB_TYPES_BY_CODE[self.data_type].numpy_name).newbyteorder("<")

    @property
    def dtype_name(self) -> str:
        return KLB_TYPES_BY_CODE[self.data_type].name

    @property
    def compression_name(self) -> str:
        return COMPRESSION_NAMES.get(self.compression_type, f"unknown({self.compression_type})")

    @property
    def bytes_per_pixel(self) -> int:
        return int(self.dtype.itemsize)

    @property
    def nblocks_per_dim(self) -> tuple[int, int, int, int, int]:
        return tuple(_ceil_div(dim, block) for dim, block in zip(self.xyzct, self.block_size))  # type: ignore[return-value]

    @property
    def num_blocks(self) -> int:
        return math.prod(self.nblocks_per_dim)

    @property
    def header_size(self) -> int:
        return KLB_HEADER_FIXED_SIZE + (8 * self.num_blocks)

    @property
    def image_size_pixels(self) -> int:
        return math.prod(self.xyzct)

    @property
    def image_size_bytes(self) -> int:
        return self.image_size_pixels * self.bytes_per_pixel

    @property
    def compressed_payload_size(self) -> int:
        return int(self.block_offsets[-1]) if self.block_offsets else 0

    def metadata_text(self) -> str:
        return self.metadata.split(b"\x00", 1)[0].decode("utf-8", "replace")

    def block_compressed_offset(self, block_id: int) -> int:
        return 0 if block_id == 0 else int(self.block_offsets[block_id - 1])

    def block_compressed_size(self, block_id: int) -> int:
        return int(self.block_offsets[block_id] - self.block_compressed_offset(block_id))


@dataclass(frozen=True)
class DirectoryBatchResult:
    source_dir: Path
    output_dir: Path
    luxendo_files: int
    klb_files: int
    archived_files: int
    archived_dirs: int
    archive_path: Path | None


@dataclass(frozen=True)
class KlbBundleResult:
    source_file: Path
    output_dir: Path
    layer_count: int
    manifest_path: Path


def read_klb_header(path: str | os.PathLike[str]) -> KlbHeader:
    path = Path(path)
    with path.open("rb") as handle:
        fixed = handle.read(KLB_HEADER_FIXED_SIZE)
        if len(fixed) != KLB_HEADER_FIXED_SIZE:
            raise ValueError(f"{path} is too small to contain a KLB header")
        unpacked = struct.unpack(KLB_HEADER_FORMAT, fixed)
        header_version = int(unpacked[0])
        xyzct = tuple(int(v) for v in unpacked[1:6])
        pixel_size = tuple(float(v) for v in unpacked[6:11])
        data_type = int(unpacked[11])
        compression_type = int(unpacked[12])
        metadata = unpacked[13]
        block_size = tuple(int(v) for v in unpacked[14:19])

        _validate_header_parts(xyzct, block_size, data_type, compression_type)
        nblocks = math.prod(_ceil_div(dim, block) for dim, block in zip(xyzct, block_size))
        offsets_bytes = handle.read(8 * nblocks)
        if len(offsets_bytes) != 8 * nblocks:
            raise ValueError(f"{path} ended before the block offset table was complete")
        offsets = struct.unpack(f"<{nblocks}Q", offsets_bytes) if nblocks else ()

    return KlbHeader(
        header_version=header_version,
        xyzct=xyzct,  # type: ignore[arg-type]
        pixel_size=pixel_size,  # type: ignore[arg-type]
        data_type=data_type,
        compression_type=compression_type,
        metadata=metadata,
        block_size=block_size,  # type: ignore[arg-type]
        block_offsets=tuple(int(v) for v in offsets),
    )


def format_klb_header(header: KlbHeader, path: str | os.PathLike[str] | None = None) -> str:
    lines: list[str] = []
    if path is not None:
        lines.extend([f"KLB: {Path(path)}", ""])
    lines.extend(
        [
            f"Header version: {header.header_version}",
            f"Dimensions xyzct: {header.xyzct}",
            f"Pixel size: {tuple(round(v, 8) for v in header.pixel_size)}",
            f"Data type: {header.dtype_name} ({header.bytes_per_pixel} bytes/voxel)",
            f"Compression: {header.compression_name}",
            f"Block size xyzct: {header.block_size}",
            f"Blocks per dimension: {header.nblocks_per_dim}",
            f"Total blocks: {header.num_blocks:,}",
            f"Raw image bytes: {header.image_size_bytes:,}",
            f"Compressed payload bytes: {header.compressed_payload_size:,}",
            f"KLB header bytes: {header.header_size:,}",
            f"Metadata: {header.metadata_text() or '(empty)'}",
        ]
    )
    return "\n".join(lines)


def find_hdf5_image_datasets(
    path: str | os.PathLike[str],
    benchmark_logger: BenchmarkLogger | None = None,
) -> list[Hdf5DatasetInfo]:
    found: list[Hdf5DatasetInfo] = []
    path = Path(path)
    with _benchmark_span(benchmark_logger, "find_hdf5_image_datasets", source_file=path):
        with h5py.File(path, "r") as h5_file:
            def visit(name: str, obj: object) -> None:
                if not isinstance(obj, h5py.Dataset):
                    return
                leaf = name.rsplit("/", 1)[-1]
                if leaf == "metadata" or obj.ndim < 3:
                    return
                if not (leaf == "Data" or leaf.startswith("Data_")):
                    return
                if not np.issubdtype(obj.dtype, np.number):
                    return
                found.append(
                    Hdf5DatasetInfo(
                        path=name,
                        shape=tuple(int(v) for v in obj.shape),
                        dtype=str(obj.dtype),
                        chunks=None if obj.chunks is None else tuple(int(v) for v in obj.chunks),
                        compression=None if obj.compression is None else str(obj.compression),
                    )
                )

            h5_file.visititems(visit)
    found.sort(key=lambda item: (item.path != "Data", item.path))
    if benchmark_logger:
        benchmark_logger.record_point(
            "find_hdf5_image_datasets.result",
            source_file=path,
            dataset_count=len(found),
            dataset_paths=[item.path for item in found],
        )
    return found


def inspect_file(path: str | os.PathLike[str]) -> str:
    path = Path(path)
    suffixes = "".join(path.suffixes).lower()
    if path.suffix.lower() == ".klb":
        return format_klb_header(read_klb_header(path), path)
    if suffixes.endswith(".lux.h5") or path.suffix.lower() in {".h5", ".hdf5"}:
        datasets = find_hdf5_image_datasets(path)
        lines = [f"HDF5: {path}", ""]
        if datasets:
            lines.extend(info.label() for info in datasets)
        else:
            lines.append("No Luxendo-style image datasets were found.")
        return "\n".join(lines)
    raise ValueError(f"Do not know how to inspect {path.name}")


def iter_blocks(
    xyzct: Sequence[int],
    block_size: Sequence[int],
) -> Iterator[tuple[int, tuple[int, int, int, int, int], tuple[int, int, int, int, int]]]:
    xyzct_5d = _normalise_xyzct(xyzct)
    block_size_5d = _normalise_block_size(block_size, xyzct_5d)
    total = math.prod(_ceil_div(dim, block) for dim, block in zip(xyzct_5d, block_size_5d))
    for block_id in range(total):
        yield block_by_id(xyzct_5d, block_size_5d, block_id)


def block_by_id(
    xyzct: Sequence[int],
    block_size: Sequence[int],
    block_id: int,
) -> tuple[int, tuple[int, int, int, int, int], tuple[int, int, int, int, int]]:
    xyzct_5d = _normalise_xyzct(xyzct)
    block_size_5d = _normalise_block_size(block_size, xyzct_5d)
    counts = tuple(_ceil_div(dim, block) for dim, block in zip(xyzct_5d, block_size_5d))
    if block_id < 0 or block_id >= math.prod(counts):
        raise IndexError(block_id)
    remaining = int(block_id)
    start: list[int] = []
    size: list[int] = []
    for dim, block, count in zip(xyzct_5d, block_size_5d, counts):
        coord = remaining % count
        remaining //= count
        first = coord * block
        last = min(first + block, dim)
        start.append(first)
        size.append(last - first)
    return block_id, tuple(start), tuple(size)  # type: ignore[return-value]


def klb_data_type_for_numpy_dtype(dtype: np.dtype | str) -> int:
    dtype = np.dtype(dtype)
    mapping = {
        ("u", 1): 0,
        ("u", 2): 1,
        ("u", 4): 2,
        ("u", 8): 3,
        ("i", 1): 4,
        ("i", 2): 5,
        ("i", 4): 6,
        ("i", 8): 7,
        ("f", 4): 8,
        ("f", 8): 9,
    }
    try:
        return mapping[(dtype.kind, dtype.itemsize)]
    except KeyError as exc:
        raise ValueError(f"Unsupported KLB dtype: {dtype}") from exc


def _process_ordered(
    items: Iterable[tuple[int, bytes]],
    process: Callable[[int, bytes], bytes],
    consume: Callable[[int, bytes], None],
    total: int,
    workers: int,
    verb: str,
    progress: ProgressCallback | None,
    cancel_check: Callable[[], bool] | None,
) -> None:
    max_pending = max(1, workers * 3)
    pending = {}
    next_to_consume = 0
    completed = 0
    iterator = iter(items)
    exhausted = False
    started = time.monotonic()

    if progress:
        progress(0, total, f"{verb} 0/{total}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        while pending or not exhausted:
            _raise_if_cancelled(cancel_check)
            while not exhausted and len(pending) < max_pending:
                try:
                    block_id, payload = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                pending[block_id] = pool.submit(process, block_id, payload)

            advanced = False
            while next_to_consume in pending and pending[next_to_consume].done():
                result = pending.pop(next_to_consume).result()
                consume(next_to_consume, result)
                completed += 1
                next_to_consume += 1
                advanced = True
                if progress:
                    elapsed = max(time.monotonic() - started, 0.001)
                    progress(completed, total, f"{verb} {completed}/{total} blocks ({completed / elapsed:.1f} blocks/s)")

            if pending and not advanced:
                wait(tuple(pending.values()), return_when=FIRST_COMPLETED)


def compress_hdf5_to_klb(
    h5_path: str | os.PathLike[str],
    dataset_path: str,
    output_path: str | os.PathLike[str],
    block_size: Sequence[int] = (128, 128, 16, 1, 1),
    compression: str | int = "bzip2",
    workers: int | None = None,
    write_metadata_sidecar: bool = True,
    progress: ProgressCallback | None = None,
    cancel_check: Callable[[], bool] | None = None,
    benchmark_logger: BenchmarkLogger | None = None,
) -> KlbHeader:
    h5_path = Path(h5_path)
    output_path = Path(output_path)
    compression_code = _compression_code(compression)
    workers = _normalise_workers(workers)
    with _benchmark_span(
        benchmark_logger,
        "compress_hdf5_to_klb.total",
        source_file=h5_path,
        dataset_path=dataset_path,
        output_path=output_path,
        compression=COMPRESSION_NAMES[compression_code],
        workers=workers,
    ):
        with h5py.File(h5_path, "r") as h5_file:
            with _benchmark_span(
                benchmark_logger,
                "compress_hdf5_to_klb.prepare_dataset",
                source_file=h5_path,
                dataset_path=dataset_path,
            ):
                dataset = h5_file[dataset_path]
                if not isinstance(dataset, h5py.Dataset):
                    raise ValueError(f"{dataset_path!r} is not an HDF5 dataset")
                if not np.issubdtype(dataset.dtype, np.number):
                    raise ValueError(f"{dataset_path!r} is not numeric")

                xyzct = _xyzct_from_hdf5_shape(dataset.shape)
                pixel_size, metadata_text = _metadata_from_hdf5_dataset(dataset)
                data_type = klb_data_type_for_numpy_dtype(dataset.dtype)
                block_size_5d = _normalise_block_size(block_size, xyzct)
                header = KlbHeader(
                    header_version=KLB_DEFAULT_HEADER_VERSION,
                    xyzct=xyzct,
                    pixel_size=pixel_size,
                    data_type=data_type,
                    compression_type=compression_code,
                    metadata=_encode_metadata(
                        f"source={h5_path.name}; dataset={dataset_path}; "
                        f"shape_xyzct={xyzct}; voxel_um={pixel_size[:3]}"
                    ),
                    block_size=block_size_5d,
                    block_offsets=(),
                )
                offsets: list[int] = []
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with output_path.open("wb") as out_handle:
                with _benchmark_span(
                    benchmark_logger,
                    "compress_hdf5_to_klb.write_header_placeholder",
                    source_file=h5_path,
                    dataset_path=dataset_path,
                    output_path=output_path,
                    num_blocks=header.num_blocks,
                ):
                    out_handle.write(_pack_header(header, [0] * header.num_blocks))

                cumulative = 0
                dtype = header.dtype
                max_block_bytes = _max_block_bytes(header)
                if benchmark_logger:
                    benchmark_logger.record_point(
                        "compress_hdf5_to_klb.dataset_info",
                        source_file=h5_path,
                        dataset_path=dataset_path,
                        output_path=output_path,
                        shape=dataset.shape,
                        dtype=str(dataset.dtype),
                        chunks=dataset.chunks,
                        num_blocks=header.num_blocks,
                        block_size=header.block_size,
                        image_size_bytes=header.image_size_bytes,
                        max_block_bytes=max_block_bytes,
                    )

                def blocks() -> Iterator[tuple[int, bytes]]:
                    for block_id, start, size in iter_blocks(header.xyzct, header.block_size):
                        _raise_if_cancelled(cancel_check)
                        started_perf = time.perf_counter()
                        started_wall = time.time()
                        block = _read_hdf5_block(dataset, start, size, dtype)
                        finished_perf = time.perf_counter()
                        if benchmark_logger:
                            benchmark_logger.record_span(
                                "compress_hdf5_to_klb.read_block",
                                started_perf=started_perf,
                                finished_perf=finished_perf,
                                started_wall=started_wall,
                                source_file=h5_path,
                                dataset_path=dataset_path,
                                block_id=block_id,
                                start=start,
                                size=size,
                                raw_bytes=block.nbytes,
                            )
                        yield block_id, block.tobytes(order="C")

                def process(block_id: int, raw: bytes) -> bytes:
                    started_perf = time.perf_counter()
                    started_wall = time.time()
                    compressed = _compress_bytes(raw, compression_code, max_block_bytes)
                    finished_perf = time.perf_counter()
                    if benchmark_logger:
                        benchmark_logger.record_span(
                            "compress_hdf5_to_klb.compress_block",
                            started_perf=started_perf,
                            finished_perf=finished_perf,
                            started_wall=started_wall,
                            source_file=h5_path,
                            dataset_path=dataset_path,
                            block_id=block_id,
                            raw_bytes=len(raw),
                            compressed_bytes=len(compressed),
                            compression=COMPRESSION_NAMES[compression_code],
                        )
                    return compressed

                def consume(block_id: int, compressed: bytes) -> None:
                    nonlocal cumulative
                    started_perf = time.perf_counter()
                    started_wall = time.time()
                    out_handle.write(compressed)
                    cumulative += len(compressed)
                    offsets.append(cumulative)
                    finished_perf = time.perf_counter()
                    if benchmark_logger:
                        benchmark_logger.record_span(
                            "compress_hdf5_to_klb.write_block",
                            started_perf=started_perf,
                            finished_perf=finished_perf,
                            started_wall=started_wall,
                            source_file=h5_path,
                            dataset_path=dataset_path,
                            block_id=block_id,
                            compressed_bytes=len(compressed),
                            cumulative_bytes=cumulative,
                        )

                _process_ordered(blocks(), process, consume, header.num_blocks, workers, "Compressed", progress, cancel_check)
                header = _replace_offsets(header, offsets)
                with _benchmark_span(
                    benchmark_logger,
                    "compress_hdf5_to_klb.rewrite_header",
                    source_file=h5_path,
                    dataset_path=dataset_path,
                    output_path=output_path,
                    compressed_payload_size=header.compressed_payload_size,
                ):
                    out_handle.seek(0)
                    out_handle.write(_pack_header(header, offsets))

            if write_metadata_sidecar and metadata_text:
                sidecar_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
                with _benchmark_span(
                    benchmark_logger,
                    "compress_hdf5_to_klb.write_metadata_sidecar",
                    source_file=h5_path,
                    dataset_path=dataset_path,
                    output_path=sidecar_path,
                ):
                    sidecar_path.write_text(metadata_text, encoding="utf-8")

        return header


def compress_hdf5_all_layers_to_klb_bundle(
    h5_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    block_size: Sequence[int] = (128, 128, 16, 1, 1),
    compression: str | int = "bzip2",
    workers: int | None = None,
    write_metadata_sidecar: bool = True,
    progress: ProgressCallback | None = None,
    cancel_check: Callable[[], bool] | None = None,
    benchmark_logger: BenchmarkLogger | None = None,
) -> KlbBundleResult:
    """Compress every Luxendo Data* dataset into a top-level KLB bundle."""

    h5_path = Path(h5_path).resolve()
    output_dir = Path(output_dir).resolve()
    with _benchmark_span(
        benchmark_logger,
        "compress_hdf5_all_layers_to_klb_bundle.total",
        source_file=h5_path,
        output_dir=output_dir,
        workers=workers,
    ):
        output_dir.mkdir(parents=True, exist_ok=True)
        entry, layer_count = _compress_luxendo_file_entry(
            h5_path=h5_path,
            source_root=h5_path.parent,
            relative_path=Path(h5_path.name),
            output_dir=output_dir,
            block_size=block_size,
            compression=compression,
            workers=workers,
            progress=progress,
            cancel_check=cancel_check,
            benchmark_logger=benchmark_logger,
        )
        manifest = {
            "format": "klb-luxendo-folder-bundle",
            "version": 2,
            "source_root": h5_path.parent.name,
            "non_luxendo_archive": None,
            "exact_luxendo_archive": None,
            "luxendo_files": [entry],
        }
        manifest_path = output_dir / KLB_BUNDLE_MANIFEST
        with _benchmark_span(
            benchmark_logger,
            "compress_hdf5_all_layers_to_klb_bundle.write_manifest",
            output_path=manifest_path,
        ):
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return KlbBundleResult(
            source_file=h5_path,
            output_dir=output_dir,
            layer_count=layer_count,
            manifest_path=manifest_path,
        )


def _compress_luxendo_file_entry(
    h5_path: Path,
    source_root: Path,
    relative_path: Path,
    output_dir: Path,
    block_size: Sequence[int],
    compression: str | int,
    workers: int | None,
    progress: ProgressCallback | None,
    cancel_check: Callable[[], bool] | None,
    benchmark_logger: BenchmarkLogger | None = None,
) -> tuple[dict, int]:
    datasets = find_hdf5_image_datasets(h5_path, benchmark_logger=benchmark_logger)
    if not datasets:
        raise ValueError(f"No Luxendo-style Data datasets were found in {h5_path}")

    safe_stem = _lux_relative_stem(relative_path)
    layer_dir = output_dir / "klb_layers" / safe_stem
    layer_dir.mkdir(parents=True, exist_ok=True)

    metadata_text = ""
    with _benchmark_span(
        benchmark_logger,
        "compress_luxendo_file.read_metadata",
        source_file=h5_path,
        relative_path=relative_path,
        dataset_path=datasets[0].path,
    ):
        with h5py.File(h5_path, "r") as h5_file:
            metadata_text = _metadata_from_hdf5_dataset(h5_file[datasets[0].path])[1]

    metadata_file = None
    if metadata_text:
        metadata_path = output_dir / "metadata" / safe_stem / "metadata.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with _benchmark_span(
            benchmark_logger,
            "compress_luxendo_file.write_metadata_file",
            source_file=h5_path,
            relative_path=relative_path,
            output_path=metadata_path,
        ):
            metadata_path.write_text(metadata_text, encoding="utf-8")
        metadata_file = metadata_path.relative_to(output_dir).as_posix()

    layers = []
    total = len(datasets)
    for index, dataset in enumerate(datasets, start=1):
        _raise_if_cancelled(cancel_check)
        klb_path = layer_dir / _dataset_path_to_klb_name(dataset.path)

        def layer_progress(_done: int, _total: int, message: str, dataset_path: str = dataset.path) -> None:
            if progress:
                progress(index - 1, total, f"{relative_path} / {dataset_path}: {message}")

        with _benchmark_span(
            benchmark_logger,
            "compress_luxendo_file.dataset_total",
            source_file=h5_path,
            relative_path=relative_path,
            dataset_path=dataset.path,
            output_path=klb_path,
        ):
            header = compress_hdf5_to_klb(
                h5_path,
                dataset.path,
                klb_path,
                block_size=block_size,
                compression=compression,
                workers=workers,
                write_metadata_sidecar=False,
                progress=layer_progress,
                cancel_check=cancel_check,
                benchmark_logger=benchmark_logger,
            )
        layers.append(
            {
                "dataset_path": dataset.path,
                "klb_file": klb_path.relative_to(output_dir).as_posix(),
                "shape": list(dataset.shape),
                "dtype": dataset.dtype,
                "chunks": None if dataset.chunks is None else list(dataset.chunks),
                "compression": dataset.compression,
                "xyzct": list(header.xyzct),
                "pixel_size": list(header.pixel_size),
            }
        )
        if progress:
            progress(index, total, f"Compressed {relative_path} / {dataset.path}")

    return (
        {
            "source_relative_path": relative_path.as_posix(),
            "source_size": h5_path.stat().st_size,
            "metadata_file": metadata_file,
            "layers": layers,
        },
        len(layers),
    )


def compress_raw_to_klb(
    raw_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    xyzct: Sequence[int],
    data_type: str | int = "uint16",
    pixel_size: Sequence[float] = (1.0, 1.0, 1.0, 1.0, 1.0),
    block_size: Sequence[int] = (128, 128, 16, 1, 1),
    compression: str | int = "bzip2",
    metadata: str | bytes = "",
    workers: int | None = None,
    progress: ProgressCallback | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> KlbHeader:
    raw_path = Path(raw_path)
    output_path = Path(output_path)
    xyzct_5d = _normalise_xyzct(xyzct)
    pixel_size_5d = _normalise_pixel_size(pixel_size)
    data_type_code = _data_type_code(data_type)
    compression_code = _compression_code(compression)
    dtype = np.dtype(KLB_TYPES_BY_CODE[data_type_code].numpy_name).newbyteorder("<")
    block_size_5d = _normalise_block_size(block_size, xyzct_5d)
    workers = _normalise_workers(workers)

    expected_size = math.prod(xyzct_5d) * dtype.itemsize
    actual_size = raw_path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(f"Raw file size mismatch: expected {expected_size:,} bytes, found {actual_size:,} bytes")

    header = KlbHeader(
        header_version=KLB_DEFAULT_HEADER_VERSION,
        xyzct=xyzct_5d,
        pixel_size=pixel_size_5d,
        data_type=data_type_code,
        compression_type=compression_code,
        metadata=_encode_metadata(metadata),
        block_size=block_size_5d,
        block_offsets=(),
    )
    offsets: list[int] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with raw_path.open("rb") as raw_handle, output_path.open("wb") as out_handle:
        out_handle.write(_pack_header(header, [0] * header.num_blocks))
        cumulative = 0
        max_block_bytes = _max_block_bytes(header)

        def blocks() -> Iterator[tuple[int, bytes]]:
            for block_id, start, size in iter_blocks(header.xyzct, header.block_size):
                _raise_if_cancelled(cancel_check)
                yield block_id, _read_raw_block(raw_handle, header.xyzct, dtype.itemsize, start, size)

        def process(_block_id: int, raw: bytes) -> bytes:
            return _compress_bytes(raw, compression_code, max_block_bytes)

        def consume(_block_id: int, compressed: bytes) -> None:
            nonlocal cumulative
            out_handle.write(compressed)
            cumulative += len(compressed)
            offsets.append(cumulative)

        _process_ordered(blocks(), process, consume, header.num_blocks, workers, "Compressed", progress, cancel_check)
        header = _replace_offsets(header, offsets)
        out_handle.seek(0)
        out_handle.write(_pack_header(header, offsets))

    return header


def compress_directory_to_klb_bundle(
    source_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    block_size: Sequence[int] = (128, 128, 16, 1, 1),
    compression: str | int = "bzip2",
    workers: int | None = None,
    parallel_files: int | None = None,
    write_metadata_sidecar: bool = True,
    archive_other_files: bool = True,
    progress: ProgressCallback | None = None,
    cancel_check: Callable[[], bool] | None = None,
    benchmark_logger: BenchmarkLogger | None = None,
) -> DirectoryBatchResult:
    """Convert every .lux.h5 image in a folder and ZIP the non-image files."""

    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir).resolve()
    workers = _normalise_workers(workers)
    if not source_dir.is_dir():
        raise ValueError(f"{source_dir} is not a directory")
    if source_dir == output_dir:
        raise ValueError("Choose an output directory outside the source directory")

    with _benchmark_span(
        benchmark_logger,
        "compress_directory_to_klb_bundle.total",
        source_dir=source_dir,
        output_dir=output_dir,
        workers=workers,
        parallel_files=parallel_files,
        archive_other_files=archive_other_files,
    ):
        with _benchmark_span(
            benchmark_logger,
            "compress_directory_to_klb_bundle.scan_inputs",
            source_dir=source_dir,
            output_dir=output_dir,
        ):
            lux_files, other_files = scan_directory_inputs(source_dir, output_dir)
        with _benchmark_span(
            benchmark_logger,
            "compress_directory_to_klb_bundle.scan_dirs",
            source_dir=source_dir,
            output_dir=output_dir,
        ):
            other_dirs = scan_directory_dirs(source_dir, output_dir)
        if not lux_files and not other_files:
            raise ValueError("No files were found in the selected directory")

        output_dir.mkdir(parents=True, exist_ok=True)
        total_steps = len(lux_files) + (len(other_files) if archive_other_files else 0)
        completed = 0
        klb_count = 0
        lux_entries: list[dict | None] = [None] * len(lux_files)
        effective_parallel_files = _normalise_parallel_file_count(parallel_files, len(lux_files), workers)
        per_file_workers = max(1, workers // effective_parallel_files)
        progress_lock = threading.Lock()

        if benchmark_logger:
            benchmark_logger.record_point(
                "compress_directory_to_klb_bundle.scan_result",
                source_dir=source_dir,
                output_dir=output_dir,
                luxendo_files=len(lux_files),
                other_files=len(other_files),
                other_dirs=len(other_dirs),
                requested_parallel_files=parallel_files,
                effective_parallel_files=effective_parallel_files,
                per_file_workers=per_file_workers,
            )
        if progress:
            progress(0, max(total_steps, 1), f"Found {len(lux_files)} Luxendo file(s) and {len(other_files)} other file(s)")

        def compress_one(index: int, lux_file: Path) -> tuple[int, dict, int, Path]:
            _raise_if_cancelled(cancel_check)
            rel = lux_file.relative_to(source_dir)

            def inner_progress(_done: int, _total: int, message: str, rel_path: Path = rel) -> None:
                if not progress:
                    return
                with progress_lock:
                    completed_snapshot = completed
                partial = completed_snapshot + (_done / max(_total, 1))
                progress(partial, max(total_steps, 1), f"{rel_path}: {message}")

            with _benchmark_span(
                benchmark_logger,
                "compress_directory_to_klb_bundle.compress_lux_file",
                source_dir=source_dir,
                source_file=lux_file,
                relative_path=rel,
                parallel_files=effective_parallel_files,
                file_workers=per_file_workers,
            ):
                entry, layer_count = _compress_luxendo_file_entry(
                    h5_path=lux_file,
                    source_root=source_dir,
                    relative_path=rel,
                    output_dir=output_dir,
                    block_size=block_size,
                    compression=compression,
                    workers=per_file_workers,
                    progress=inner_progress,
                    cancel_check=cancel_check,
                    benchmark_logger=benchmark_logger,
                )
            return index, entry, layer_count, rel

        if effective_parallel_files == 1:
            for index, lux_file in enumerate(lux_files):
                index, entry, layer_count, rel = compress_one(index, lux_file)
                lux_entries[index] = entry
                klb_count += layer_count
                completed += 1
                if progress:
                    progress(completed, max(total_steps, 1), f"Converted {rel}")
        else:
            pending = set()
            next_index = 0
            with ThreadPoolExecutor(max_workers=effective_parallel_files) as pool:
                while next_index < len(lux_files) and len(pending) < effective_parallel_files:
                    pending.add(pool.submit(compress_one, next_index, lux_files[next_index]))
                    next_index += 1

                while pending:
                    _raise_if_cancelled(cancel_check)
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        index, entry, layer_count, rel = future.result()
                        lux_entries[index] = entry
                        klb_count += layer_count
                        with progress_lock:
                            completed += 1
                            completed_snapshot = completed
                        if progress:
                            progress(completed_snapshot, max(total_steps, 1), f"Converted {rel}")
                        if next_index < len(lux_files):
                            pending.add(pool.submit(compress_one, next_index, lux_files[next_index]))
                            next_index += 1

        archive_path: Path | None = None
        archived_count = 0
        archived_dir_count = 0
        if archive_other_files and (other_files or other_dirs):
            archive_path = output_dir / "non_luxendo_files.zip"
            with _benchmark_span(
                benchmark_logger,
                "compress_directory_to_klb_bundle.archive_other_files",
                archive_path=archive_path,
                file_count=len(other_files),
                dir_count=len(other_dirs),
            ):
                with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
                    for other_dir in other_dirs:
                        _raise_if_cancelled(cancel_check)
                        rel_dir = other_dir.relative_to(source_dir).as_posix().rstrip("/") + "/"
                        with _benchmark_span(
                            benchmark_logger,
                            "compress_directory_to_klb_bundle.archive_directory_entry",
                            archive_path=archive_path,
                            relative_path=rel_dir,
                        ):
                            archive.write(other_dir, rel_dir)
                        archived_dir_count += 1
                    for other_file in other_files:
                        _raise_if_cancelled(cancel_check)
                        rel_file = other_file.relative_to(source_dir)
                        with _benchmark_span(
                            benchmark_logger,
                            "compress_directory_to_klb_bundle.archive_other_file",
                            archive_path=archive_path,
                            source_file=other_file,
                            relative_path=rel_file,
                            file_size=other_file.stat().st_size,
                        ):
                            archive.write(other_file, rel_file.as_posix())
                        archived_count += 1
                        completed += 1
                        if progress:
                            progress(completed, max(total_steps, 1), f"Archived {rel_file}")

        manifest = {
            "format": "klb-luxendo-folder-bundle",
            "version": 2,
            "source_root": source_dir.name,
            "non_luxendo_archive": archive_path.relative_to(output_dir).as_posix() if archive_path else None,
            "exact_luxendo_archive": None,
            "luxendo_files": [entry for entry in lux_entries if entry is not None],
        }
        manifest_path = output_dir / KLB_BUNDLE_MANIFEST
        with _benchmark_span(
            benchmark_logger,
            "compress_directory_to_klb_bundle.write_manifest",
            output_path=manifest_path,
        ):
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        return DirectoryBatchResult(
            source_dir=source_dir,
            output_dir=output_dir,
            luxendo_files=len(lux_files),
            klb_files=klb_count,
            archived_files=archived_count,
            archived_dirs=archived_dir_count,
            archive_path=archive_path,
        )


def scan_directory_inputs(
    source_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str] | None = None,
) -> tuple[list[Path], list[Path]]:
    source_dir = Path(source_dir).resolve()
    output_resolved = Path(output_dir).resolve() if output_dir else None
    lux_files: list[Path] = []
    other_files: list[Path] = []
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue
        resolved = path.resolve()
        if output_resolved is not None and _path_is_relative_to(resolved, output_resolved):
            continue
        if path.name.lower().endswith(".lux.h5"):
            lux_files.append(path)
        else:
            other_files.append(path)
    return lux_files, other_files


def scan_directory_dirs(
    source_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str] | None = None,
) -> list[Path]:
    source_dir = Path(source_dir).resolve()
    output_resolved = Path(output_dir).resolve() if output_dir else None
    dirs: list[Path] = []
    for path in sorted(source_dir.rglob("*")):
        if not path.is_dir():
            continue
        resolved = path.resolve()
        if output_resolved is not None and _path_is_relative_to(resolved, output_resolved):
            continue
        dirs.append(path)
    return dirs


def repair_bundle_manifest(
    bundle_dir: str | os.PathLike[str],
    source_dir: str | os.PathLike[str] | None = None,
) -> Path:
    bundle_dir = Path(bundle_dir).resolve()
    source_root = Path(source_dir).resolve() if source_dir else None
    layers_root = bundle_dir / "klb_layers"
    if not layers_root.is_dir():
        raise ValueError(f"Cannot repair manifest because {layers_root} does not exist")

    lux_entries = []
    for layer_dir in sorted(path for path in layers_root.rglob("*") if path.is_dir()):
        klb_files = sorted(layer_dir.glob("*.klb"))
        if not klb_files:
            continue
        rel_stem = layer_dir.relative_to(layers_root)
        source_relative = rel_stem.with_name(rel_stem.name + ".lux.h5")
        source_file = source_root / source_relative if source_root else None
        source_datasets = {}
        if source_file is not None and source_file.exists():
            source_datasets = {item.path: item for item in find_hdf5_image_datasets(source_file)}

        metadata_file = None
        candidate_metadata = bundle_dir / "metadata" / rel_stem / "metadata.json"
        if candidate_metadata.exists():
            metadata_file = candidate_metadata.relative_to(bundle_dir).as_posix()

        entry_layers = []
        for klb_file in klb_files:
            dataset_path = klb_file.stem.replace("__", "/")
            header = read_klb_header(klb_file)
            source_info = source_datasets.get(dataset_path)
            entry_layers.append(
                {
                    "dataset_path": dataset_path,
                    "klb_file": klb_file.relative_to(bundle_dir).as_posix(),
                    "shape": list(source_info.shape) if source_info else list(_hdf5_shape_from_xyzct(header.xyzct)),
                    "dtype": source_info.dtype if source_info else header.dtype_name,
                    "chunks": list(source_info.chunks) if source_info and source_info.chunks else None,
                    "compression": source_info.compression if source_info else None,
                    "xyzct": list(header.xyzct),
                    "pixel_size": list(header.pixel_size),
                }
            )

        lux_entries.append(
            {
                "source_relative_path": source_relative.as_posix(),
                "source_size": source_file.stat().st_size if source_file is not None and source_file.exists() else None,
                "metadata_file": metadata_file,
                "layers": entry_layers,
            }
        )

    non_archive = bundle_dir / "non_luxendo_files.zip"
    manifest = {
        "format": "klb-luxendo-folder-bundle",
        "version": 2,
        "source_root": source_root.name if source_root else None,
        "non_luxendo_archive": non_archive.name if non_archive.exists() else None,
        "exact_luxendo_archive": None,
        "luxendo_files": lux_entries,
    }
    manifest_path = bundle_dir / KLB_BUNDLE_MANIFEST
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return manifest_path


def decompress_klb_to_raw(
    klb_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    workers: int | None = None,
    progress: ProgressCallback | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> KlbHeader:
    klb_path = Path(klb_path)
    output_path = Path(output_path)
    header = read_klb_header(klb_path)
    workers = _normalise_workers(workers)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with klb_path.open("rb") as in_handle, output_path.open("w+b") as out_handle:
        out_handle.truncate(header.image_size_bytes)

        def blocks() -> Iterator[tuple[int, bytes]]:
            for block_id, _start, _size in iter_blocks(header.xyzct, header.block_size):
                _raise_if_cancelled(cancel_check)
                in_handle.seek(header.header_size + header.block_compressed_offset(block_id))
                yield block_id, in_handle.read(header.block_compressed_size(block_id))

        def process(block_id: int, compressed: bytes) -> bytes:
            _, _start, size = block_by_id(header.xyzct, header.block_size, block_id)
            raw = _decompress_bytes(compressed, header.compression_type)
            _check_block_size(block_id, raw, size, header.bytes_per_pixel)
            return raw

        def consume(block_id: int, raw: bytes) -> None:
            _, start, size = block_by_id(header.xyzct, header.block_size, block_id)
            _write_raw_block(out_handle, raw, header.xyzct, header.bytes_per_pixel, start, size)

        _process_ordered(blocks(), process, consume, header.num_blocks, workers, "Decompressed", progress, cancel_check)

    return header


def decompress_klb_to_lux_h5(
    klb_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    workers: int | None = None,
    progress: ProgressCallback | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> KlbHeader:
    klb_path = Path(klb_path)
    output_path = Path(output_path)
    header = read_klb_header(klb_path)
    workers = _normalise_workers(workers)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with klb_path.open("rb") as in_handle, h5py.File(output_path, "w") as h5_file:
        shape = _hdf5_shape_from_xyzct(header.xyzct)
        chunks = _hdf5_chunks_from_block(header.block_size, header.xyzct)
        dataset = h5_file.create_dataset("Data", shape=shape, dtype=header.dtype, chunks=chunks)
        h5_file.create_dataset("metadata", data=_luxendo_metadata_from_klb(header, klb_path), dtype=h5py.string_dtype("utf-8"))

        def blocks() -> Iterator[tuple[int, bytes]]:
            for block_id, _start, _size in iter_blocks(header.xyzct, header.block_size):
                _raise_if_cancelled(cancel_check)
                in_handle.seek(header.header_size + header.block_compressed_offset(block_id))
                yield block_id, in_handle.read(header.block_compressed_size(block_id))

        def process(block_id: int, compressed: bytes) -> bytes:
            _, _start, size = block_by_id(header.xyzct, header.block_size, block_id)
            raw = _decompress_bytes(compressed, header.compression_type)
            _check_block_size(block_id, raw, size, header.bytes_per_pixel)
            return raw

        def consume(block_id: int, raw: bytes) -> None:
            _, start, size = block_by_id(header.xyzct, header.block_size, block_id)
            dataset[_hdf5_selection_from_klb_block(start, size, len(shape))] = _array_from_klb_block(raw, header.dtype, size, header.xyzct)

        _process_ordered(blocks(), process, consume, header.num_blocks, workers, "Decompressed", progress, cancel_check)

    return header


def decompress_klb_bundle_to_lux_h5(
    bundle_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    workers: int | None = None,
    progress: ProgressCallback | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> list[KlbHeader]:
    """Decompress a single KLB or a pyramid bundle into one Luxendo-style HDF5 file."""

    bundle_path = Path(bundle_path)
    output_path = Path(output_path)
    if bundle_path.is_file() and bundle_path.suffix.lower() == ".klb":
        return [decompress_klb_to_lux_h5(bundle_path, output_path, workers, progress, cancel_check)]

    manifest_path = _resolve_manifest_path(bundle_path)
    bundle_dir = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    layers = manifest.get("layers", [])
    if not layers:
        raise ValueError(f"No layers were listed in {manifest_path}")

    workers = _normalise_workers(workers)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers: list[KlbHeader] = []
    total = len(layers)

    with h5py.File(output_path, "w") as h5_file:
        metadata_file = manifest.get("metadata_file")
        metadata_text = ""
        if metadata_file and (bundle_dir / metadata_file).exists():
            metadata_text = (bundle_dir / metadata_file).read_text(encoding="utf-8")

        for index, layer in enumerate(layers, start=1):
            _raise_if_cancelled(cancel_check)
            dataset_path = layer["dataset_path"]
            klb_path = bundle_dir / layer["klb_file"]
            header = read_klb_header(klb_path)
            headers.append(header)
            shape = _hdf5_shape_from_xyzct(header.xyzct)
            chunks = _layer_chunks(layer, header)
            parent = _ensure_hdf5_parent_group(h5_file, dataset_path)
            name = dataset_path.rsplit("/", 1)[-1]
            dataset = parent.create_dataset(name, shape=shape, dtype=header.dtype, chunks=chunks)

            with klb_path.open("rb") as in_handle:
                def blocks() -> Iterator[tuple[int, bytes]]:
                    for block_id, _start, _size in iter_blocks(header.xyzct, header.block_size):
                        _raise_if_cancelled(cancel_check)
                        in_handle.seek(header.header_size + header.block_compressed_offset(block_id))
                        yield block_id, in_handle.read(header.block_compressed_size(block_id))

                def process(block_id: int, compressed: bytes) -> bytes:
                    _, _start, size = block_by_id(header.xyzct, header.block_size, block_id)
                    raw = _decompress_bytes(compressed, header.compression_type)
                    _check_block_size(block_id, raw, size, header.bytes_per_pixel)
                    return raw

                def consume(block_id: int, raw: bytes) -> None:
                    _, start, size = block_by_id(header.xyzct, header.block_size, block_id)
                    dataset[_hdf5_selection_from_klb_block(start, size, len(shape))] = _array_from_klb_block(raw, header.dtype, size, header.xyzct)

                def layer_progress(_done: int, _total: int, message: str, dataset_path: str = dataset_path) -> None:
                    if progress:
                        progress(index - 1, total, f"{dataset_path}: {message}")

                _process_ordered(blocks(), process, consume, header.num_blocks, workers, "Decompressed", layer_progress, cancel_check)

            if progress:
                progress(index, total, f"Restored {dataset_path}")

        if metadata_text:
            if "metadata" not in h5_file:
                h5_file.create_dataset("metadata", data=metadata_text, dtype=h5py.string_dtype("utf-8"))

    return headers


def decompress_klb_bundle_to_lux_folder(
    bundle_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    workers: int | None = None,
    progress: ProgressCallback | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> list[Path]:
    """Restore a compressed operation bundle into an output directory tree."""

    bundle_path = Path(bundle_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if bundle_path.is_file() and bundle_path.suffix.lower() == ".klb":
        output_file = output_dir / f"{bundle_path.stem}.lux.h5"
        decompress_klb_to_lux_h5(bundle_path, output_file, workers, progress, cancel_check)
        return [output_file]

    manifest_path = _resolve_manifest_path(bundle_path)
    bundle_dir = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    restored: list[Path] = []

    exact_archive = manifest.get("exact_luxendo_archive")
    if exact_archive and (bundle_dir / exact_archive).exists():
        extracted = _extract_zip_safe(bundle_dir / exact_archive, output_dir, cancel_check)
        restored.extend(path for path in extracted if path.name.lower().endswith(".lux.h5"))
        archive_name = manifest.get("non_luxendo_archive")
        if archive_name and (bundle_dir / archive_name).exists():
            _extract_zip_safe(bundle_dir / archive_name, output_dir, cancel_check)
        if progress:
            progress(1, 1, f"Restored exact Luxendo files from {exact_archive}")
        return restored

    luxendo_files = manifest.get("luxendo_files")
    if not luxendo_files:
        # Backward compatibility with early single-file bundle manifests.
        output_file = output_dir / manifest.get("source_file", "restored.lux.h5")
        decompress_klb_bundle_to_lux_h5(manifest_path, output_file, workers, progress, cancel_check)
        return [output_file]

    total = len(luxendo_files) + (1 if manifest.get("non_luxendo_archive") else 0)
    for index, file_entry in enumerate(luxendo_files, start=1):
        _raise_if_cancelled(cancel_check)
        rel_path = Path(file_entry["source_relative_path"])
        output_file = output_dir / rel_path
        output_file.parent.mkdir(parents=True, exist_ok=True)
        metadata_text = ""
        metadata_file = file_entry.get("metadata_file")
        if metadata_file and (bundle_dir / metadata_file).exists():
            metadata_text = (bundle_dir / metadata_file).read_text(encoding="utf-8")

        def file_progress(_done: int, _total: int, message: str, rel_path: Path = rel_path) -> None:
            if progress:
                progress(index - 1, total, f"{rel_path}: {message}")

        _restore_luxendo_file_from_layers(
            bundle_dir=bundle_dir,
            file_entry=file_entry,
            output_file=output_file,
            metadata_text=metadata_text,
            workers=workers,
            progress=file_progress,
            cancel_check=cancel_check,
        )
        restored.append(output_file)
        if progress:
            progress(index, total, f"Restored {rel_path}")

    archive_name = manifest.get("non_luxendo_archive")
    if archive_name:
        archive_path = bundle_dir / archive_name
        if archive_path.exists():
            _extract_zip_safe(archive_path, output_dir, cancel_check)
            if progress:
                progress(total, total, f"Unpacked {archive_name}")

    return restored


def _restore_luxendo_file_from_layers(
    bundle_dir: Path,
    file_entry: dict,
    output_file: Path,
    metadata_text: str,
    workers: int | None,
    progress: ProgressCallback | None,
    cancel_check: Callable[[], bool] | None,
) -> list[KlbHeader]:
    workers = _normalise_workers(workers)
    headers: list[KlbHeader] = []
    layers = file_entry.get("layers", [])
    if not layers:
        raise ValueError(f"No layers found for {file_entry.get('source_relative_path')}")

    with h5py.File(output_file, "w") as h5_file:
        for index, layer in enumerate(layers, start=1):
            _raise_if_cancelled(cancel_check)
            dataset_path = layer["dataset_path"]
            klb_path = bundle_dir / layer["klb_file"]
            header = read_klb_header(klb_path)
            headers.append(header)
            shape = _hdf5_shape_from_xyzct(header.xyzct)
            chunks = _layer_chunks(layer, header)
            parent = _ensure_hdf5_parent_group(h5_file, dataset_path)
            name = dataset_path.rsplit("/", 1)[-1]
            dataset = parent.create_dataset(name, shape=shape, dtype=header.dtype, chunks=chunks)

            with klb_path.open("rb") as in_handle:
                def blocks() -> Iterator[tuple[int, bytes]]:
                    for block_id, _start, _size in iter_blocks(header.xyzct, header.block_size):
                        _raise_if_cancelled(cancel_check)
                        in_handle.seek(header.header_size + header.block_compressed_offset(block_id))
                        yield block_id, in_handle.read(header.block_compressed_size(block_id))

                def process(block_id: int, compressed: bytes) -> bytes:
                    _, _start, size = block_by_id(header.xyzct, header.block_size, block_id)
                    raw = _decompress_bytes(compressed, header.compression_type)
                    _check_block_size(block_id, raw, size, header.bytes_per_pixel)
                    return raw

                def consume(block_id: int, raw: bytes) -> None:
                    _, start, size = block_by_id(header.xyzct, header.block_size, block_id)
                    dataset[_hdf5_selection_from_klb_block(start, size, len(shape))] = _array_from_klb_block(raw, header.dtype, size, header.xyzct)

                def layer_progress(_done: int, _total: int, message: str, dataset_path: str = dataset_path) -> None:
                    if progress:
                        progress(index - 1, len(layers), f"{dataset_path}: {message}")

                _process_ordered(blocks(), process, consume, header.num_blocks, workers, "Decompressed", layer_progress, cancel_check)

        if metadata_text:
            if "metadata" not in h5_file:
                h5_file.create_dataset("metadata", data=metadata_text, dtype=h5py.string_dtype("utf-8"))

    return headers


def _read_hdf5_block(dataset: h5py.Dataset, start: Sequence[int], size: Sequence[int], dtype: np.dtype) -> np.ndarray:
    block = np.asarray(dataset[_hdf5_selection_from_klb_block(start, size, dataset.ndim)])
    if block.dtype != dtype:
        block = block.astype(dtype, copy=False)
    return np.ascontiguousarray(block)


def _read_raw_block(handle, xyzct: Sequence[int], bytes_per_pixel: int, start: Sequence[int], size: Sequence[int]) -> bytes:
    x_dim, y_dim, z_dim, c_dim, _t_dim = xyzct
    x0, y0, z0, c0, t0 = start
    nx, ny, nz, nc, nt = size
    row_bytes = nx * bytes_per_pixel
    out = bytearray(math.prod(size) * bytes_per_pixel)
    out_offset = 0
    for lt in range(nt):
        for lc in range(nc):
            for lz in range(nz):
                for ly in range(ny):
                    linear = x0 + x_dim * (
                        (y0 + ly) + y_dim * ((z0 + lz) + z_dim * ((c0 + lc) + c_dim * (t0 + lt)))
                    )
                    handle.seek(linear * bytes_per_pixel)
                    row = handle.read(row_bytes)
                    if len(row) != row_bytes:
                        raise ValueError("Raw input ended while reading a block")
                    out[out_offset : out_offset + row_bytes] = row
                    out_offset += row_bytes
    return bytes(out)


def _write_raw_block(
    handle,
    raw: bytes,
    xyzct: Sequence[int],
    bytes_per_pixel: int,
    start: Sequence[int],
    size: Sequence[int],
) -> None:
    x_dim, y_dim, z_dim, c_dim, _t_dim = xyzct
    x0, y0, z0, c0, t0 = start
    nx, ny, nz, nc, nt = size
    row_bytes = nx * bytes_per_pixel
    for lt in range(nt):
        for lc in range(nc):
            for lz in range(nz):
                for ly in range(ny):
                    block_linear = nx * (ly + ny * (lz + nz * (lc + nc * lt)))
                    block_offset = block_linear * bytes_per_pixel
                    file_linear = x0 + x_dim * (
                        (y0 + ly) + y_dim * ((z0 + lz) + z_dim * ((c0 + lc) + c_dim * (t0 + lt)))
                    )
                    handle.seek(file_linear * bytes_per_pixel)
                    handle.write(raw[block_offset : block_offset + row_bytes])


def _array_from_klb_block(raw: bytes, dtype: np.dtype, size: Sequence[int], xyzct: Sequence[int]) -> np.ndarray:
    nx, ny, nz, nc, nt = size
    _x_dim, _y_dim, _z_dim, c_dim, t_dim = xyzct
    array = np.frombuffer(raw, dtype=dtype)
    if c_dim == 1 and t_dim == 1:
        return array.reshape((nz, ny, nx))
    if t_dim == 1:
        return array.reshape((nc, nz, ny, nx))
    return array.reshape((nt, nc, nz, ny, nx))


def _hdf5_selection_from_klb_block(start: Sequence[int], size: Sequence[int], ndim: int) -> tuple[slice, ...]:
    x0, y0, z0, c0, t0 = start
    nx, ny, nz, nc, nt = size
    if ndim == 3:
        return (slice(z0, z0 + nz), slice(y0, y0 + ny), slice(x0, x0 + nx))
    if ndim == 4:
        return (slice(c0, c0 + nc), slice(z0, z0 + nz), slice(y0, y0 + ny), slice(x0, x0 + nx))
    if ndim == 5:
        return (
            slice(t0, t0 + nt),
            slice(c0, c0 + nc),
            slice(z0, z0 + nz),
            slice(y0, y0 + ny),
            slice(x0, x0 + nx),
        )
    raise ValueError("Only 3D, 4D, and 5D HDF5 image datasets are supported")


def _xyzct_from_hdf5_shape(shape: Sequence[int]) -> tuple[int, int, int, int, int]:
    shape = tuple(int(v) for v in shape)
    if len(shape) == 3:
        z_dim, y_dim, x_dim = shape
        return (x_dim, y_dim, z_dim, 1, 1)
    if len(shape) == 4:
        c_dim, z_dim, y_dim, x_dim = shape
        return (x_dim, y_dim, z_dim, c_dim, 1)
    if len(shape) == 5:
        t_dim, c_dim, z_dim, y_dim, x_dim = shape
        return (x_dim, y_dim, z_dim, c_dim, t_dim)
    raise ValueError(f"Expected a 3D, 4D, or 5D HDF5 image dataset, found shape {shape}")


def _hdf5_shape_from_xyzct(xyzct: Sequence[int]) -> tuple[int, ...]:
    x_dim, y_dim, z_dim, c_dim, t_dim = _normalise_xyzct(xyzct)
    if c_dim == 1 and t_dim == 1:
        return (z_dim, y_dim, x_dim)
    if t_dim == 1:
        return (c_dim, z_dim, y_dim, x_dim)
    return (t_dim, c_dim, z_dim, y_dim, x_dim)


def _hdf5_chunks_from_block(block_size: Sequence[int], xyzct: Sequence[int]) -> tuple[int, ...]:
    bx, by, bz, bc, bt = _normalise_block_size(block_size, xyzct)
    x_dim, y_dim, z_dim, c_dim, t_dim = _normalise_xyzct(xyzct)
    if c_dim == 1 and t_dim == 1:
        return (min(bz, z_dim), min(by, y_dim), min(bx, x_dim))
    if t_dim == 1:
        return (min(bc, c_dim), min(bz, z_dim), min(by, y_dim), min(bx, x_dim))
    return (min(bt, t_dim), min(bc, c_dim), min(bz, z_dim), min(by, y_dim), min(bx, x_dim))


def _layer_chunks(layer: dict, header: KlbHeader) -> tuple[int, ...]:
    chunks = layer.get("chunks")
    if chunks:
        return tuple(int(v) for v in chunks)
    return _hdf5_chunks_from_block(header.block_size, header.xyzct)


def _metadata_from_hdf5_dataset(dataset: h5py.Dataset) -> tuple[tuple[float, float, float, float, float], str]:
    metadata_dataset = dataset.parent.get("metadata")
    if not isinstance(metadata_dataset, h5py.Dataset):
        return (1.0, 1.0, 1.0, 1.0, 1.0), ""
    text = _hdf5_value_to_text(metadata_dataset[()])
    try:
        payload = json.loads(text)
        processing = payload.get("processingInformation", payload)
        voxel = processing.get("voxel_size_um", {})
        return (
            float(voxel.get("width", 1.0)),
            float(voxel.get("height", 1.0)),
            float(voxel.get("depth", 1.0)),
            1.0,
            1.0,
        ), text
    except Exception:
        return (1.0, 1.0, 1.0, 1.0, 1.0), text


def _hdf5_value_to_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", "replace")
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"S", "a"}:
            return value.tobytes().decode("utf-8", "replace")
        if value.shape == ():
            return _hdf5_value_to_text(value.item())
    return str(value)


def _luxendo_metadata_from_klb(header: KlbHeader, klb_path: Path) -> str:
    payload = {
        "processingInformation": {
            "version": "1.0.0",
            "sources": [f"KLB GUI conversion from {klb_path.name}"],
            "voxel_size_um": {
                "width": header.pixel_size[0],
                "height": header.pixel_size[1],
                "depth": header.pixel_size[2],
            },
            "image_size_vx": {"width": header.xyzct[0], "height": header.xyzct[1], "depth": header.xyzct[2]},
            "klb": {
                "header_version": header.header_version,
                "xyzct": header.xyzct,
                "block_size": header.block_size,
                "data_type": header.dtype_name,
                "compression": header.compression_name,
                "metadata": header.metadata_text(),
            },
        }
    }
    return json.dumps(payload, separators=(",", ":"))


def _compress_bytes(data: bytes, compression_type: int, max_block_bytes: int) -> bytes:
    if compression_type == 0:
        return data
    if compression_type == 1:
        level = max(1, min(9, _ceil_div(max_block_bytes, 100_000)))
        return bz2.compress(data, compresslevel=level)
    if compression_type == 2:
        return zlib.compress(data)
    raise ValueError(f"Unsupported compression type: {compression_type}")


def _decompress_bytes(data: bytes, compression_type: int) -> bytes:
    if compression_type == 0:
        return data
    if compression_type == 1:
        return bz2.decompress(data)
    if compression_type == 2:
        return zlib.decompress(data)
    raise ValueError(f"Unsupported compression type: {compression_type}")


def _check_block_size(block_id: int, raw: bytes, size: Sequence[int], bytes_per_pixel: int) -> None:
    expected = math.prod(size) * bytes_per_pixel
    if len(raw) != expected:
        raise ValueError(f"Block {block_id} decompressed to {len(raw):,} bytes; expected {expected:,}")


def _pack_header(header: KlbHeader, offsets: Sequence[int]) -> bytes:
    if len(offsets) != header.num_blocks:
        raise ValueError(f"Expected {header.num_blocks} offsets, received {len(offsets)}")
    fixed = struct.pack(
        KLB_HEADER_FORMAT,
        header.header_version,
        *header.xyzct,
        *header.pixel_size,
        header.data_type,
        header.compression_type,
        _encode_metadata(header.metadata),
        *header.block_size,
    )
    return fixed + struct.pack(f"<{len(offsets)}Q", *offsets)


def _replace_offsets(header: KlbHeader, offsets: Sequence[int]) -> KlbHeader:
    return KlbHeader(
        header_version=header.header_version,
        xyzct=header.xyzct,
        pixel_size=header.pixel_size,
        data_type=header.data_type,
        compression_type=header.compression_type,
        metadata=header.metadata,
        block_size=header.block_size,
        block_offsets=tuple(int(v) for v in offsets),
    )


def _encode_metadata(metadata: str | bytes) -> bytes:
    raw = metadata.encode("utf-8", "replace") if isinstance(metadata, str) else bytes(metadata)
    return raw[:KLB_METADATA_SIZE].ljust(KLB_METADATA_SIZE, b"\x00")


def _normalise_xyzct(values: Sequence[int]) -> tuple[int, int, int, int, int]:
    if len(values) == 3:
        values = (*values, 1, 1)
    if len(values) != KLB_DATA_DIMS:
        raise ValueError("Expected xyzct as 3 or 5 integer values")
    parsed = tuple(int(v) for v in values)
    if any(v <= 0 for v in parsed):
        raise ValueError(f"Dimensions must be positive: {parsed}")
    return parsed  # type: ignore[return-value]


def _normalise_pixel_size(values: Sequence[float]) -> tuple[float, float, float, float, float]:
    if len(values) == 3:
        values = (*values, 1.0, 1.0)
    if len(values) != KLB_DATA_DIMS:
        raise ValueError("Expected pixel size as 3 or 5 numeric values")
    parsed = tuple(float(v) for v in values)
    if any(v <= 0 for v in parsed):
        raise ValueError(f"Pixel sizes must be positive: {parsed}")
    return parsed  # type: ignore[return-value]


def _normalise_block_size(values: Sequence[int], xyzct: Sequence[int]) -> tuple[int, int, int, int, int]:
    if len(values) == 3:
        values = (*values, 1, 1)
    if len(values) != KLB_DATA_DIMS:
        raise ValueError("Expected block size as 3 or 5 integer values")
    dims = _normalise_xyzct(xyzct)
    return tuple(max(1, min(int(v), dim)) for v, dim in zip(values, dims))  # type: ignore[return-value]


def _data_type_code(value: str | int) -> int:
    if isinstance(value, int):
        if value not in KLB_TYPES_BY_CODE:
            raise ValueError(f"Unsupported KLB data type code: {value}")
        return value
    key = str(value).strip().lower()
    if key not in KLB_TYPES_BY_NAME:
        raise ValueError(f"Unsupported KLB data type: {value}")
    return KLB_TYPES_BY_NAME[key].code


def _compression_code(value: str | int) -> int:
    if isinstance(value, int):
        if value not in COMPRESSION_NAMES:
            raise ValueError(f"Unsupported KLB compression type code: {value}")
        return value
    key = str(value).strip().lower()
    if key not in COMPRESSION_CODES:
        raise ValueError(f"Unsupported KLB compression type: {value}")
    return COMPRESSION_CODES[key]


def _dataset_path_to_klb_name(dataset_path: str) -> str:
    safe = dataset_path.strip("/").replace("/", "__")
    return f"{safe}.klb"


def _lux_relative_stem(relative_path: Path) -> Path:
    name = relative_path.name
    if name.lower().endswith(".lux.h5"):
        name = name[:-7]
    else:
        name = relative_path.stem
    return relative_path.parent / name


def _bundle_relative_path(relative_path: Path) -> Path:
    name = relative_path.name
    if name.lower().endswith(".lux.h5"):
        name = name[:-7] + "_klb_bundle"
    else:
        name = relative_path.stem + "_klb_bundle"
    return relative_path.parent / name


def _klb_relative_path(relative_path: Path) -> Path:
    name = relative_path.name
    if name.lower().endswith(".lux.h5"):
        name = name[:-7] + ".klb"
    else:
        name = relative_path.stem + ".klb"
    return relative_path.parent / name


def _resolve_manifest_path(path: Path) -> Path:
    if path.is_dir():
        manifest = path / KLB_BUNDLE_MANIFEST
    else:
        manifest = path
    if not manifest.exists():
        raise ValueError(f"Bundle manifest was not found at {manifest}")
    return manifest


def _ensure_hdf5_parent_group(h5_file: h5py.File, dataset_path: str) -> h5py.Group | h5py.File:
    parts = [part for part in dataset_path.strip("/").split("/") if part]
    if len(parts) <= 1:
        return h5_file
    group = h5_file
    for part in parts[:-1]:
        group = group.require_group(part)
    return group


def _write_exact_luxendo_archive(
    archive_path: Path,
    files: Sequence[tuple[Path, Path]],
    cancel_check: Callable[[], bool] | None,
    benchmark_logger: BenchmarkLogger | None = None,
) -> None:
    if not files:
        return
    temp_path = archive_path.with_name(archive_path.name + ".tmp")
    with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for file_path, arcname in files:
            _raise_if_cancelled(cancel_check)
            with _benchmark_span(
                benchmark_logger,
                "write_exact_luxendo_archive.file",
                archive_path=archive_path,
                source_file=file_path,
                relative_path=arcname,
                file_size=file_path.stat().st_size,
            ):
                archive.write(file_path, arcname.as_posix())
    temp_path.replace(archive_path)


def _extract_zip_safe(
    archive_path: Path,
    output_dir: Path,
    cancel_check: Callable[[], bool] | None,
) -> list[Path]:
    output_dir = output_dir.resolve()
    extracted: list[Path] = []
    with zipfile.ZipFile(archive_path, "r") as archive:
        for member in archive.infolist():
            _raise_if_cancelled(cancel_check)
            target = (output_dir / member.filename).resolve()
            if not _path_is_relative_to(target, output_dir):
                raise ValueError(f"Refusing to extract unsafe archive member: {member.filename}")
            archive.extract(member, output_dir)
            extracted.append(target)
    return extracted


def _path_is_relative_to(path: Path, possible_parent: Path) -> bool:
    try:
        path.relative_to(possible_parent)
        return True
    except ValueError:
        return False


def _max_block_bytes(header: KlbHeader) -> int:
    return math.prod(header.block_size) * header.bytes_per_pixel


def _ceil_div(a: int, b: int) -> int:
    return (int(a) + int(b) - 1) // int(b)


def _normalise_workers(workers: int | None) -> int:
    if workers is None or workers <= 0:
        return max(1, os.cpu_count() or 1)
    return max(1, int(workers))


def _normalise_parallel_file_count(parallel_files: int | None, lux_file_count: int, workers: int) -> int:
    if lux_file_count <= 1:
        return 1
    if parallel_files is None or parallel_files <= 0:
        if workers >= 24:
            return min(lux_file_count, 2)
        return 1
    return max(1, min(int(parallel_files), lux_file_count, workers))


def _validate_header_parts(xyzct: Sequence[int], block_size: Sequence[int], data_type: int, compression_type: int) -> None:
    _normalise_xyzct(xyzct)
    _normalise_block_size(block_size, xyzct)
    if data_type not in KLB_TYPES_BY_CODE:
        raise ValueError(f"Unsupported KLB data type code: {data_type}")
    if compression_type not in COMPRESSION_NAMES:
        raise ValueError(f"Unsupported KLB compression type code: {compression_type}")


def _raise_if_cancelled(cancel_check: Callable[[], bool] | None) -> None:
    if cancel_check is not None and cancel_check():
        raise RuntimeError("Operation cancelled")
