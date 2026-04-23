"""Tkinter GUI for compressing Luxendo HDF5/raw data to KLB and back."""

from __future__ import annotations

import os
import queue
import threading
import time
import traceback
import hashlib
import json
from pathlib import Path
import tkinter as tk
from tkinter import BooleanVar, IntVar, StringVar, filedialog, messagebox, ttk

import klb_codec as klb


class KlbLuxendoApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("KLB Luxendo Tool")
        self.geometry("1040x760")
        self.minsize(900, 650)
        self.message_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.cancel_event = threading.Event()
        self.current_thread: threading.Thread | None = None
        self.hdf5_datasets: list[klb.Hdf5DatasetInfo] = []
        self._configure_style()
        self._build_ui()
        self.after(100, self._drain_queue)

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Title.TLabel", font=("Segoe UI", 15, "bold"))
        style.configure("Subtle.TLabel", foreground="#4b5563")
        style.configure("Action.TButton", padding=(14, 7))

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=14)
        root.pack(fill="both", expand=True)
        ttk.Label(root, text="KLB Luxendo Tool", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            root,
            text="Compress experiment folders, restore Luxendo data, and verify file checksums.",
            style="Subtle.TLabel",
        ).pack(anchor="w", pady=(2, 12))

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True)
        # Single-file compression is preserved for future use, but hidden from
        # the main workflow while folder-level bundles are the preferred path.
        # self._build_luxendo_tab()
        self._build_folder_tab()
        self._build_decompress_tab()
        self._build_inspect_tab()

        bottom = ttk.Frame(root)
        bottom.pack(fill="x", pady=(12, 0))
        self.progress = ttk.Progressbar(bottom, orient="horizontal", mode="determinate", maximum=100)
        self.progress.pack(side="left", fill="x", expand=True)
        self.status_var = StringVar(value="Ready")
        ttk.Label(bottom, textvariable=self.status_var, width=45).pack(side="left", padx=(10, 0))
        self.cancel_button = ttk.Button(bottom, text="Cancel", command=self._cancel_current, state="disabled")
        self.cancel_button.pack(side="left", padx=(10, 0))

        log_frame = ttk.LabelFrame(root, text="Log", padding=8)
        log_frame.pack(fill="both", expand=False, pady=(10, 0))
        self.log_text = tk.Text(log_frame, height=8, wrap="word")
        self.log_text.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scroll.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=scroll.set)
        self._log("Ready. Choose a tab above to start.")

    def _build_luxendo_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(frame, text="Luxendo to KLB")

        self.lux_source = StringVar()
        self.lux_output = StringVar()
        self.lux_block_x = IntVar(value=128)
        self.lux_block_y = IntVar(value=128)
        self.lux_block_z = IntVar(value=16)
        self.lux_workers = IntVar(value=max(1, os.cpu_count() or 1))
        self.lux_compression = StringVar(value="bzip2")
        self.lux_info = StringVar(value="Select a .lux.h5 file and scan for datasets.")

        self._path_row(
            frame,
            0,
            "Source .lux.h5",
            self.lux_source,
            lambda: self._browse_open(self.lux_source, [("Luxendo/HDF5", "*.lux.h5 *.h5 *.hdf5"), ("All files", "*.*")]),
        )
        ttk.Button(frame, text="Scan Datasets", command=self._scan_luxendo).grid(row=0, column=3, padx=(8, 0), sticky="ew")
        self._path_row(
            frame,
            1,
            "Output bundle directory",
            self.lux_output,
            lambda: self._browse_directory(self.lux_output),
        )
        self._block_options(frame, 2, self.lux_block_x, self.lux_block_y, self.lux_block_z, self.lux_workers, self.lux_compression)
        ttk.Label(frame, textvariable=self.lux_info, wraplength=790, style="Subtle.TLabel").grid(row=3, column=0, columnspan=4, sticky="ew", pady=(8, 10))
        ttk.Button(frame, text="Compress All Layers", style="Action.TButton", command=self._compress_luxendo).grid(row=4, column=1, sticky="w", pady=8)
        frame.columnconfigure(1, weight=1)

    def _build_folder_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(frame, text="Folder Batch")

        self.folder_source = StringVar()
        self.folder_output = StringVar()
        self.folder_block_x = IntVar(value=128)
        self.folder_block_y = IntVar(value=128)
        self.folder_block_z = IntVar(value=16)
        self.folder_workers = IntVar(value=max(1, os.cpu_count() or 1))
        self.folder_compression = StringVar(value="bzip2")
        self.folder_archive_other = BooleanVar(value=True)
        self.folder_info = StringVar(
            value="Select an experiment folder. .lux.h5 files become .klb; other files can be archived as ZIP."
        )
        self._folder_scan_after_id = None

        self._path_row(frame, 0, "Source directory", self.folder_source, lambda: self._browse_directory(self.folder_source))
        self._path_row(frame, 1, "Output directory", self.folder_output, lambda: self._browse_directory(self.folder_output))
        self._block_options(frame, 2, self.folder_block_x, self.folder_block_y, self.folder_block_z, self.folder_workers, self.folder_compression)
        ttk.Checkbutton(frame, text="Archive non-Luxendo files as non_luxendo_files.zip", variable=self.folder_archive_other).grid(row=3, column=1, sticky="w", pady=6)
        ttk.Label(frame, textvariable=self.folder_info, wraplength=850, justify="left", style="Subtle.TLabel").grid(row=4, column=0, columnspan=5, sticky="ew", pady=(8, 10))
        ttk.Button(frame, text="Compress Folder", style="Action.TButton", command=self._compress_folder).grid(row=5, column=1, sticky="w", pady=8)
        self.folder_source.trace_add("write", lambda *_: self._schedule_folder_scan())
        self.folder_output.trace_add("write", lambda *_: self._schedule_folder_scan())
        frame.columnconfigure(1, weight=1)

    def _build_decompress_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(frame, text="Decompress")

        self.dec_source = StringVar()
        self.dec_output = StringVar()
        self.dec_workers = IntVar(value=max(1, os.cpu_count() or 1))
        self.dec_header_text = StringVar(value="Select a single .klb file, bundle manifest, or bundle directory.")
        self._decompress_load_after_id = None

        self._path_row(frame, 0, "Source .klb or bundle", self.dec_source, lambda: self._browse_open(self.dec_source, [("KLB/bundle", "*.klb *.json"), ("All files", "*.*")]))
        ttk.Button(frame, text="Load Header", command=self._load_decompress_header).grid(row=0, column=4, sticky="ew", padx=(8, 0))
        ttk.Button(frame, text="Browse Folder", command=lambda: self._browse_directory(self.dec_source)).grid(row=0, column=5, sticky="ew", padx=(8, 0))

        ttk.Label(frame, text="Output").grid(row=1, column=0, sticky="w", pady=6)
        opts = ttk.Frame(frame)
        opts.grid(row=1, column=1, sticky="w", pady=6)
        ttk.Label(opts, text="restore folder").pack(side="left")
        ttk.Label(opts, text="workers").pack(side="left", padx=(16, 4))
        ttk.Spinbox(opts, from_=1, to=128, textvariable=self.dec_workers, width=6).pack(side="left")

        self._path_row(frame, 2, "Output directory", self.dec_output, self._browse_decompress_output)
        ttk.Label(frame, textvariable=self.dec_header_text, wraplength=850, justify="left", style="Subtle.TLabel").grid(row=3, column=0, columnspan=4, sticky="ew", pady=(8, 10))
        ttk.Button(frame, text="Decompress KLB", style="Action.TButton", command=self._decompress).grid(row=4, column=1, sticky="w", pady=8)
        self.dec_source.trace_add("write", lambda *_: self._schedule_decompress_header_load())
        frame.columnconfigure(1, weight=1)

    def _build_inspect_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(frame, text="Verify")

        self.inspect_source_a = StringVar()
        self.inspect_source_b = StringVar()
        self._path_row(frame, 0, "File A", self.inspect_source_a, lambda: self._browse_open(self.inspect_source_a, [("All files", "*.*")]))
        self._path_row(frame, 1, "File B", self.inspect_source_b, lambda: self._browse_open(self.inspect_source_b, [("All files", "*.*")]))
        ttk.Button(frame, text="Compare MD5", style="Action.TButton", command=self._compare_checksums).grid(row=2, column=1, sticky="w", pady=(8, 4))
        result_frame = ttk.Frame(frame)
        result_frame.grid(row=3, column=0, columnspan=4, sticky="nsew", pady=(10, 0))
        self.inspect_text = tk.Text(result_frame, wrap="none")
        self.inspect_text.pack(side="left", fill="both", expand=True)
        scroll_y = ttk.Scrollbar(result_frame, command=self.inspect_text.yview)
        scroll_y.pack(side="right", fill="y")
        self.inspect_text.configure(yscrollcommand=scroll_y.set)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(1, weight=1)

    def _path_row(self, frame: ttk.Frame, row: int, label: str, var: StringVar, browse_command) -> None:
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=6)
        ttk.Entry(frame, textvariable=var).grid(row=row, column=1, columnspan=2, sticky="ew", pady=6)
        ttk.Button(frame, text="Browse", command=browse_command).grid(row=row, column=3, padx=(8, 0), sticky="ew", pady=6)

    def _block_options(
        self,
        frame: ttk.Frame,
        row: int,
        block_x: IntVar,
        block_y: IntVar,
        block_z: IntVar,
        workers: IntVar,
        compression: StringVar,
    ) -> None:
        ttk.Label(frame, text="Block size xyz").grid(row=row, column=0, sticky="w", pady=6)
        options = ttk.Frame(frame)
        options.grid(row=row, column=1, columnspan=3, sticky="w", pady=6)
        for label, var in (("x", block_x), ("y", block_y), ("z", block_z)):
            ttk.Label(options, text=label).pack(side="left", padx=(0, 3))
            ttk.Spinbox(options, from_=1, to=10_000, textvariable=var, width=7).pack(side="left", padx=(0, 8))
        ttk.Label(options, text="compression").pack(side="left", padx=(12, 4))
        ttk.Combobox(options, textvariable=compression, values=list(klb.COMPRESSION_CODES), state="readonly", width=9).pack(side="left")
        ttk.Label(options, text="workers").pack(side="left", padx=(12, 4))
        ttk.Spinbox(options, from_=1, to=128, textvariable=workers, width=6).pack(side="left")

    def _scan_luxendo(self) -> None:
        source = self._require_path(self.lux_source.get(), "Choose a Luxendo/HDF5 file first")
        if source is None:
            return
        try:
            datasets = klb.find_hdf5_image_datasets(source)
            if not datasets:
                raise ValueError("No Luxendo-style Data datasets were found")
            self.hdf5_datasets = datasets
            self.lux_info.set("All available resolution layers will be compressed:\n" + "\n".join(item.label() for item in datasets))
            if not self.lux_output.get().strip():
                self.lux_output.set(str(source.parent / f"{self._lux_stem(source)}_klb_bundle"))
            self._log(f"Found {len(datasets)} resolution layer(s) in {source.name}.")
        except Exception as exc:
            messagebox.showerror("Scan failed", str(exc))
            self._log_error(exc)

    def _scan_folder(self) -> None:
        source_text = self.folder_source.get().strip()
        if not source_text:
            self.folder_info.set("Select an experiment folder. .lux.h5 files become .klb; other files can be archived as ZIP.")
            return
        source = Path(source_text)
        if not source.exists():
            self.folder_info.set(f"Waiting for a valid source directory: {source}")
            return
        if not source.is_dir():
            self.folder_info.set(f"Source is not a directory: {source}")
            return
        output = self._folder_output_path(source)
        if not self.folder_output.get().strip():
            self.folder_output.set(str(output))
        try:
            lux_files, other_files = klb.scan_directory_inputs(source, output)
            other_dirs = klb.scan_directory_dirs(source, output)
            self.folder_info.set(
                f"Found {len(lux_files)} .lux.h5 file(s), {len(other_files)} other file(s), "
                f"and {len(other_dirs)} directorie(s). "
                f"Output will go to {output}. Other files are packaged in non_luxendo_files.zip."
            )
            self._log(
                f"Scanned {source}: {len(lux_files)} Luxendo file(s), "
                f"{len(other_files)} other file(s), {len(other_dirs)} directorie(s)."
            )
        except Exception as exc:
            self._log_error(exc)
            self.folder_info.set(f"Folder scan failed: {exc}")

    def _schedule_folder_scan(self) -> None:
        if self._folder_scan_after_id is not None:
            self.after_cancel(self._folder_scan_after_id)
        self._folder_scan_after_id = self.after(250, self._run_scheduled_folder_scan)

    def _run_scheduled_folder_scan(self) -> None:
        self._folder_scan_after_id = None
        self._scan_folder()

    def _compress_luxendo(self) -> None:
        source = self._require_path(self.lux_source.get(), "Choose a Luxendo/HDF5 source file")
        if source is None:
            return
        output = self._output_path(self.lux_output.get(), source.parent / f"{self._lux_stem(source)}_klb_bundle")
        block = (self.lux_block_x.get(), self.lux_block_y.get(), self.lux_block_z.get(), 1, 1)

        def work(progress):
            return klb.compress_hdf5_all_layers_to_klb_bundle(
                source,
                output,
                block_size=block,
                compression=self.lux_compression.get(),
                workers=self.lux_workers.get(),
                write_metadata_sidecar=True,
                progress=progress,
                cancel_check=self.cancel_event.is_set,
            )

        self._run_job(
            "Compress Luxendo",
            work,
            lambda result: self._log(
                f"Compressed {result.layer_count} layer(s) into {result.output_dir}. Manifest: {result.manifest_path}"
            ),
        )

    def _compress_folder(self) -> None:
        source = self._require_directory(self.folder_source.get(), "Choose a source directory")
        if source is None:
            return
        output = self._folder_output_path(source)
        if not self.folder_output.get().strip():
            self.folder_output.set(str(output))
        block = (self.folder_block_x.get(), self.folder_block_y.get(), self.folder_block_z.get(), 1, 1)

        def work(progress):
            return klb.compress_directory_to_klb_bundle(
                source,
                output,
                block_size=block,
                compression=self.folder_compression.get(),
                workers=self.folder_workers.get(),
                write_metadata_sidecar=True,
                archive_other_files=self.folder_archive_other.get(),
                progress=progress,
                cancel_check=self.cancel_event.is_set,
            )

        def done(result: klb.DirectoryBatchResult) -> None:
            archive = result.archive_path if result.archive_path else "(not requested)"
            self._log(
                f"Folder batch complete: {result.klb_files}/{result.luxendo_files} Luxendo file(s) converted; "
                f"{result.archived_files} other file(s) and {result.archived_dirs} directorie(s) archived at {archive}."
            )

        self._run_job("Compress folder", work, done)

    def _schedule_decompress_header_load(self) -> None:
        if self._decompress_load_after_id is not None:
            self.after_cancel(self._decompress_load_after_id)
        self._decompress_load_after_id = self.after(250, self._run_scheduled_decompress_header_load)

    def _run_scheduled_decompress_header_load(self) -> None:
        self._decompress_load_after_id = None
        self._load_decompress_header(quiet=True)

    def _load_decompress_header(self, quiet: bool = False) -> None:
        source_text = self.dec_source.get().strip()
        if not source_text:
            self.dec_header_text.set("Select a single .klb file, bundle manifest, or bundle directory.")
            if not quiet:
                messagebox.showwarning("Missing path", "Choose a KLB file first")
            return
        source = Path(source_text)
        if not source.exists():
            self.dec_header_text.set(f"Waiting for a valid KLB source: {source}")
            if not quiet:
                messagebox.showwarning("Missing path", f"{source} does not exist")
            return
        try:
            if source.is_file() and source.suffix.lower() == ".klb":
                self.dec_header_text.set(klb.format_klb_header(klb.read_klb_header(source), source))
            else:
                manifest = source / klb.KLB_BUNDLE_MANIFEST if source.is_dir() else source
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                files = payload.get("luxendo_files", [])
                self.dec_header_text.set(
                    f"KLB bundle: {manifest}\n\n"
                    f"Luxendo files: {len(files)}\n"
                    + "\n".join(
                        f"- {entry.get('source_relative_path')} ({len(entry.get('layers', []))} layer(s))"
                        for entry in files
                    )
                )
            if not self.dec_output.get().strip():
                self.dec_output.set(str(self._default_decompress_dir(source)))
            self._log(f"Loaded KLB header from {source.name}.")
        except Exception as exc:
            if quiet:
                self.dec_header_text.set(f"Header load failed: {exc}")
            else:
                messagebox.showerror("Header load failed", str(exc))
            self._log_error(exc)

    def _decompress(self) -> None:
        source = self._require_path(self.dec_source.get(), "Choose a KLB source file")
        if source is None:
            return
        output = self._output_path(self.dec_output.get(), self._default_decompress_dir(source))

        def work(progress):
            return klb.decompress_klb_bundle_to_lux_folder(
                source,
                output,
                workers=self.dec_workers.get(),
                progress=progress,
                cancel_check=self.cancel_event.is_set,
            )

        self._run_job("Decompress KLB", work, lambda files: self._log(f"Restored {len(files)} .lux.h5 file(s) into {output}"))

    def _compare_checksums(self) -> None:
        source_a = self._require_path(self.inspect_source_a.get(), "Choose File A")
        source_b = self._require_path(self.inspect_source_b.get(), "Choose File B")
        if source_a is None or source_b is None:
            return

        def work(progress):
            total = source_a.stat().st_size + source_b.stat().st_size
            done = 0

            def checksum_progress(path: Path, bytes_done: int, file_size: int) -> None:
                progress(done + bytes_done, total, f"Checksumming {path.name} ({bytes_done:,}/{file_size:,} bytes)")

            md5_a = self._md5_file(source_a, checksum_progress)
            done += source_a.stat().st_size
            md5_b = self._md5_file(source_b, checksum_progress)
            return {
                "file_a": source_a,
                "file_b": source_b,
                "size_a": source_a.stat().st_size,
                "size_b": source_b.stat().st_size,
                "md5_a": md5_a,
                "md5_b": md5_b,
                "same": source_a.stat().st_size == source_b.stat().st_size and md5_a == md5_b,
            }

        def done(result: dict) -> None:
            text = "\n".join(
                [
                    f"File A: {result['file_a']}",
                    f"Size A: {result['size_a']:,} bytes",
                    f"MD5 A:  {result['md5_a']}",
                    "",
                    f"File B: {result['file_b']}",
                    f"Size B: {result['size_b']:,} bytes",
                    f"MD5 B:  {result['md5_b']}",
                    "",
                    f"Identical: {'YES' if result['same'] else 'NO'}",
                ]
            )
            self.inspect_text.delete("1.0", "end")
            self.inspect_text.insert("1.0", text)
            self._log("MD5 comparison complete: " + ("files match." if result["same"] else "files differ."))

        self._run_job("Compare MD5", work, done)

    def _md5_file(self, path: Path, progress) -> str:
        digest = hashlib.md5()
        file_size = path.stat().st_size
        bytes_done = 0
        with path.open("rb") as handle:
            while True:
                if self.cancel_event.is_set():
                    raise RuntimeError("Operation cancelled")
                chunk = handle.read(8 * 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                bytes_done += len(chunk)
                progress(path, bytes_done, file_size)
        return digest.hexdigest()

    def _browse_open(self, var: StringVar, filetypes) -> None:
        selected = filedialog.askopenfilename(filetypes=filetypes)
        if selected:
            var.set(selected)

    def _browse_save(self, var: StringVar, default_extension: str, filetypes) -> None:
        selected = filedialog.asksaveasfilename(defaultextension=default_extension, filetypes=filetypes)
        if selected:
            var.set(selected)

    def _browse_directory(self, var: StringVar) -> None:
        selected = filedialog.askdirectory()
        if selected:
            var.set(selected)

    def _browse_decompress_output(self) -> None:
        self._browse_directory(self.dec_output)

    def _require_path(self, value: str, message: str) -> Path | None:
        if not value.strip():
            messagebox.showwarning("Missing path", message)
            return None
        path = Path(value)
        if not path.exists():
            messagebox.showwarning("Missing path", f"{path} does not exist")
            return None
        return path

    def _require_directory(self, value: str, message: str) -> Path | None:
        path = self._require_path(value, message)
        if path is None:
            return None
        if not path.is_dir():
            messagebox.showwarning("Missing directory", f"{path} is not a directory")
            return None
        return path

    def _output_path(self, value: str, default: Path) -> Path:
        return Path(value.strip()) if value.strip() else default

    def _folder_output_path(self, source: Path) -> Path:
        if self.folder_output.get().strip():
            return Path(self.folder_output.get().strip())
        return source.parent / f"{source.name}_klb_bundle"

    def _default_decompress_dir(self, source: Path) -> Path:
        if source.is_dir():
            return source.with_name(source.name.removesuffix("_klb_bundle") + "_restored")
        if source.name == klb.KLB_BUNDLE_MANIFEST:
            bundle = source.parent
            return bundle.with_name(bundle.name.removesuffix("_klb_bundle") + "_restored")
        return source.with_name(source.stem + "_restored")

    def _lux_stem(self, source: Path) -> str:
        return source.name[:-7] if source.name.lower().endswith(".lux.h5") else source.stem

    def _run_job(self, name: str, work, on_done=None) -> None:
        if self.current_thread is not None and self.current_thread.is_alive():
            messagebox.showinfo("Busy", "A conversion is already running.")
            return
        self.cancel_event.clear()
        self.cancel_button.configure(state="normal")
        self.progress.configure(value=0)
        self.status_var.set(f"{name} started")
        self._log(f"{name} started.")

        def progress(done: int, total: int, message: str) -> None:
            self.message_queue.put(("progress", (done, total, message)))

        def target() -> None:
            started = time.monotonic()
            try:
                result = work(progress)
                elapsed = time.monotonic() - started
                self.message_queue.put(("done", (name, result, elapsed, on_done)))
            except Exception as exc:
                self.message_queue.put(("error", (name, exc, traceback.format_exc())))

        self.current_thread = threading.Thread(target=target, daemon=True)
        self.current_thread.start()

    def _cancel_current(self) -> None:
        self.cancel_event.set()
        self.status_var.set("Cancelling...")
        self._log("Cancel requested.")

    def _drain_queue(self) -> None:
        try:
            while True:
                kind, payload = self.message_queue.get_nowait()
                if kind == "progress":
                    done, total, message = payload
                    percent = 0 if total == 0 else int((done / total) * 100)
                    self.progress.configure(value=percent)
                    self.status_var.set(message)
                elif kind == "done":
                    name, result, elapsed, on_done = payload
                    self.progress.configure(value=100)
                    self.status_var.set(f"{name} completed in {elapsed:.1f}s")
                    self.cancel_button.configure(state="disabled")
                    self._log(f"{name} completed in {elapsed:.1f}s.")
                    if on_done:
                        on_done(result)
                elif kind == "error":
                    name, exc, details = payload
                    self.cancel_button.configure(state="disabled")
                    self.status_var.set(f"{name} failed")
                    self._log(f"{name} failed: {exc}")
                    self._log(details)
                    messagebox.showerror(f"{name} failed", str(exc))
        except queue.Empty:
            pass
        self.after(100, self._drain_queue)

    def _log(self, text: str) -> None:
        self.log_text.insert("end", text.rstrip() + "\n")
        self.log_text.see("end")

    def _log_error(self, exc: Exception) -> None:
        self._log(f"Error: {exc}")


if __name__ == "__main__":
    KlbLuxendoApp().mainloop()
