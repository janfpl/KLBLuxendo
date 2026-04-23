# KLB Luxendo GUI

This folder contains a lightweight Python GUI for the Keller Lab Block (`.klb`) compression format.

The repository's native implementation is C++ and is the best path for maximum throughput after it is built. On this workstation, Python 3.13, `tkinter`, `numpy`, and `h5py` are available, while CMake/MSVC/Java are not on `PATH`, so this GUI uses a pure-Python KLB codec to avoid a native build step.

## What It Does

- Compare two files by size and MD5 checksum.
- Compress all Luxendo HDF5 resolution layers such as `Data`, `Data_2_2_2`, and `Data_4_4_4` into a KLB bundle.
- Select a whole experiment folder, convert every `.lux.h5` file into one operation-level KLB bundle, and ZIP non-Luxendo files while preserving relative paths.
- Decompress a KLB bundle into an output directory that recreates the original `.lux.h5` filenames and unpacks non-Luxendo files.

## Format Notes

KLB stores data as a five-dimensional image with dimensions `x, y, z, channel, time`. The header is:

- fixed 319-byte portion: version, `xyzct`, pixel size, data type, compression type, 256-byte metadata field, and block size
- variable block-offset table: one `uint64` cumulative compressed byte offset per block
- block payloads: x-fastest blocks, ordered x, y, z, channel, time

Luxendo `.lux.h5` files store 3D datasets as `z, y, x`. The converter maps each available `Data*` resolution layer to KLB `x, y, z, 1, 1` and keeps the byte order compatible with the C++ KLB reader.

KLB stores one image array per file. To preserve Luxendo resolution pyramids and whole folders, the GUI writes one top-level `klb_bundle_manifest.json` per compression operation plus a `klb_layers/` tree containing one `.klb` per `Data*` layer. Decompression rebuilds each `.lux.h5` from those KLB layers and restores the non-image sidecar files into the original directory structure. Older bundles that already contain `exact_luxendo_files.zip` are still supported during decompression, but new runs do not create that large passthrough archive.

The KLB header only has 256 metadata bytes. When compressing from Luxendo HDF5, the GUI can write the full original metadata to sidecar files and to the bundle manifest folder.

For folder batches, KLB is used only for Luxendo image volumes. Arbitrary files such as JSON, XML, logs, IMS/BDV files, notes, and registration folders such as `reg_obj` and `reg_sti` are written to `non_luxendo_files.zip` in the compressed output folder, because KLB cannot represent general-purpose file contents. Directory entries are included too, so empty folders are recreated during decompression. During decompression, that ZIP is automatically unpacked into the selected output directory.

## Run

From this repository folder:

```powershell
python python_gui\luxendo_klb_gui.py
```

## Validate

The included sample test checks KLB compatibility, all-layer Luxendo bundle restore, and folder-batch packaging:

```powershell
python python_gui\test_klb_codec.py
```

## Benchmark

To benchmark folder compression with timestamped per-step logs and keep every run:

```powershell
python python_gui\benchmark_folder_compression.py
```

Each run creates a unique folder under:

- `..\benchmark_logs\<run-id>\events.jsonl`: raw timestamped events for scans, per-file work, per-layer work, block read/compress/write timings, manifest writes, and archive steps
- `..\benchmark_logs\<run-id>\summary.json`: grouped timing totals plus the slowest individual spans
- `..\benchmark_outputs\<run-id>\...`: the corresponding compressed output bundle

You can override the dataset, worker count, compression, or block size with `--source`, `--workers`, `--compression`, `--block-x`, `--block-y`, and `--block-z`.

## Suggested Settings

For Luxendo `uint16` volumes, start with:

- compression: `bzip2`
- block size: `128 x 128 x 16`
- workers: number of CPU cores

The test dataset at `..\testdata\20220514-165524_Task_3_Ovary_sample3_4_C` contains two large Luxendo files around 6.3 GiB each, with `Data` shaped `704 x 2048 x 2048`, dtype `uint16`, and HDF5 chunks `64 x 64 x 64`.
