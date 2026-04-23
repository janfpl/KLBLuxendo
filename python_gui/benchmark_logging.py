from __future__ import annotations

import heapq
import json
import os
import platform
import socket
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping


def _utc_iso(timestamp: float | None = None) -> str:
    value = time.time() if timestamp is None else float(timestamp)
    return datetime.fromtimestamp(value, timezone.utc).isoformat(timespec="milliseconds")


def _safe_name(value: str) -> str:
    cleaned = []
    for char in value:
        if char.isalnum() or char in {"-", "_"}:
            cleaned.append(char)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("_") or "run"


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except Exception:
            pass
    return str(value)


class BenchmarkLogger:
    """Thread-safe JSONL benchmark logger with a per-run summary."""

    def __init__(
        self,
        log_root: str | os.PathLike[str],
        run_name: str = "benchmark",
        metadata: Mapping[str, Any] | None = None,
        top_spans_limit: int = 100,
    ) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        self.run_name = _safe_name(run_name)
        self.run_id = f"{self.run_name}_{stamp}_pid{os.getpid()}"
        self.log_root = Path(log_root).resolve()
        self.run_dir = self.log_root / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "events.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self.run_path = self.run_dir / "run.json"
        self._lock = threading.Lock()
        self._sequence = 0
        self._closed = False
        self._span_totals: dict[str, dict[str, float]] = {}
        self._top_spans_limit = max(1, int(top_spans_limit))
        self._top_spans_heap: list[tuple[float, int, dict[str, Any]]] = []
        self._started_perf = time.perf_counter()
        self._started_wall = time.time()

        run_metadata = {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "started_at_utc": _utc_iso(self._started_wall),
            "log_root": str(self.log_root),
            "run_dir": str(self.run_dir),
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python_version": sys.version,
            "cpu_count": os.cpu_count(),
            "metadata": _jsonable(metadata or {}),
        }
        self.run_path.write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
        self.record_point("run_started", run_id=self.run_id, run_name=self.run_name)

    @contextmanager
    def span(self, name: str, **fields: Any) -> Iterator[None]:
        started_perf = time.perf_counter()
        started_wall = time.time()
        error: dict[str, Any] | None = None
        try:
            yield
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            finished_perf = time.perf_counter()
            finished_wall = time.time()
            self.record_span(
                name,
                started_perf=started_perf,
                finished_perf=finished_perf,
                started_wall=started_wall,
                finished_wall=finished_wall,
                status="error" if error else "ok",
                error=error,
                **fields,
            )

    def record_point(self, name: str, **fields: Any) -> None:
        payload = {
            "kind": "point",
            "name": name,
            "timestamp_utc": _utc_iso(),
            "thread": threading.current_thread().name,
            **{key: _jsonable(value) for key, value in fields.items()},
        }
        with self._lock:
            self._write_event_locked(payload)

    def record_span(
        self,
        name: str,
        *,
        started_perf: float,
        finished_perf: float,
        started_wall: float | None = None,
        finished_wall: float | None = None,
        **fields: Any,
    ) -> None:
        started_wall = time.time() if started_wall is None else float(started_wall)
        finished_wall = time.time() if finished_wall is None else float(finished_wall)
        duration_s = max(0.0, float(finished_perf) - float(started_perf))
        payload = {
            "kind": "span",
            "name": name,
            "started_at_utc": _utc_iso(started_wall),
            "finished_at_utc": _utc_iso(finished_wall),
            "duration_s": duration_s,
            "thread": threading.current_thread().name,
            **{key: _jsonable(value) for key, value in fields.items()},
        }
        with self._lock:
            self._update_summary_locked(name, duration_s, payload)
            self._write_event_locked(payload)

    def close(self, status: str = "ok", extra_summary: Mapping[str, Any] | None = None) -> None:
        if self._closed:
            return
        finished_perf = time.perf_counter()
        finished_wall = time.time()
        total_duration_s = max(0.0, finished_perf - self._started_perf)
        self.record_point(
            "run_finished",
            run_id=self.run_id,
            status=status,
            total_duration_s=total_duration_s,
        )
        summary = {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "status": status,
            "started_at_utc": _utc_iso(self._started_wall),
            "finished_at_utc": _utc_iso(finished_wall),
            "total_duration_s": total_duration_s,
            "events_path": str(self.events_path),
            "top_spans": [
                item[2]
                for item in sorted(
                    self._top_spans_heap,
                    key=lambda value: (value[0], value[1]),
                    reverse=True,
                )
            ],
            "span_totals": sorted(
                (
                    {
                        "name": name,
                        "count": int(values["count"]),
                        "total_duration_s": values["total_duration_s"],
                        "avg_duration_s": values["total_duration_s"] / values["count"],
                        "min_duration_s": values["min_duration_s"],
                        "max_duration_s": values["max_duration_s"],
                    }
                    for name, values in self._span_totals.items()
                ),
                key=lambda item: item["total_duration_s"],
                reverse=True,
            ),
            "extra_summary": _jsonable(extra_summary or {}),
        }
        self.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self._closed = True

    def _update_summary_locked(self, name: str, duration_s: float, payload: Mapping[str, Any]) -> None:
        current = self._span_totals.get(name)
        if current is None:
            self._span_totals[name] = {
                "count": 1.0,
                "total_duration_s": duration_s,
                "min_duration_s": duration_s,
                "max_duration_s": duration_s,
            }
        else:
            current["count"] += 1.0
            current["total_duration_s"] += duration_s
            current["min_duration_s"] = min(current["min_duration_s"], duration_s)
            current["max_duration_s"] = max(current["max_duration_s"], duration_s)

        top_payload = {
            "name": name,
            "duration_s": duration_s,
            "started_at_utc": payload.get("started_at_utc"),
            "finished_at_utc": payload.get("finished_at_utc"),
        }
        for key in (
            "source_dir",
            "source_file",
            "relative_path",
            "dataset_path",
            "output_path",
            "archive_path",
            "block_id",
            "raw_bytes",
            "compressed_bytes",
            "status",
        ):
            if key in payload:
                top_payload[key] = payload[key]
        heapq.heappush(self._top_spans_heap, (duration_s, self._sequence, top_payload))
        if len(self._top_spans_heap) > self._top_spans_limit:
            heapq.heappop(self._top_spans_heap)

    def _write_event_locked(self, payload: Mapping[str, Any]) -> None:
        event = {
            "sequence": self._sequence,
            "run_id": self.run_id,
            **payload,
        }
        self._sequence += 1
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
