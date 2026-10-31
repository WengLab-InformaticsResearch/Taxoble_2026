from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
import threading
import time
from typing import Any, Iterator
from uuid import uuid4

from .llm_client import invoke_chat_model


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def atomic_write_json(value: Any, path: str) -> None:
    """Write JSON atomically without leaving a partially written destination."""

    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary_path, path)


class TaxonomyRunRecorder:
    """Structured timing, LLM usage, and cycle telemetry for one stage run."""

    def __init__(
        self,
        *,
        enabled: bool,
        telemetry_dir: str,
        stage: str,
        stage_summary_filename: str,
        cfg: Any,
    ) -> None:
        self.enabled = enabled
        self.telemetry_dir = telemetry_dir
        self.stage = stage
        self.stage_summary_filename = stage_summary_filename
        self.cfg = cfg
        self.run_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            + f"_{uuid4().hex[:8]}"
        )
        self.run_dir = os.path.join(telemetry_dir, "runs", self.run_id)
        self._lock = threading.Lock()
        self._request_counter = 0
        self._logical_call_counter = 0
        self._cycle_counter = 0
        self._requests: list[dict[str, Any]] = []
        self._units: list[dict[str, Any]] = []
        self._cycles: list[dict[str, Any]] = []
        self._thread_context = threading.local()
        self._start_perf = time.perf_counter()
        self._start_timestamp = _timestamp()
        self._end_timestamp: str | None = None
        self._console_log: str | None = None
        self._request_file = None
        self._unit_file = None

        if self.enabled:
            os.makedirs(self.run_dir, exist_ok=True)
            self._request_file = open(
                os.path.join(self.run_dir, "llm_requests.jsonl"),
                "a",
                encoding="utf-8",
            )
            unit_filename = (
                "topic_runtime.jsonl"
                if stage == "local_construction"
                else "step_runtime.jsonl"
            )
            self._unit_file = open(
                os.path.join(self.run_dir, unit_filename),
                "a",
                encoding="utf-8",
            )
            atomic_write_json([], os.path.join(self.run_dir, "cycle_resolutions.json"))

    @staticmethod
    def extract_usage(response: Any) -> tuple[int | None, int | None, int | None]:
        input_tokens = getattr(response, "prompt_tokens", None)
        output_tokens = getattr(response, "completion_tokens", None)
        total_tokens = getattr(response, "total_tokens", None)

        usage_metadata = getattr(response, "usage_metadata", None)
        if isinstance(usage_metadata, dict):
            input_tokens = usage_metadata.get(
                "input_tokens",
                usage_metadata.get("prompt_tokens", input_tokens),
            )
            output_tokens = usage_metadata.get(
                "output_tokens",
                usage_metadata.get("completion_tokens", output_tokens),
            )
            total_tokens = usage_metadata.get("total_tokens", total_tokens)

        response_metadata = getattr(response, "response_metadata", None)
        if isinstance(response_metadata, dict):
            input_tokens = response_metadata.get("prompt_eval_count", input_tokens)
            output_tokens = response_metadata.get("eval_count", output_tokens)

        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens
        return input_tokens, output_tokens, total_tokens

    def set_console_log(self, path: str) -> None:
        if self.enabled:
            self._console_log = path

    @contextmanager
    def context(self, **values: Any) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        previous = self._current_context()
        self._thread_context.value = {**previous, **values}
        try:
            yield
        finally:
            self._thread_context.value = previous

    def _current_context(self) -> dict[str, Any]:
        return getattr(self._thread_context, "value", {}).copy()

    def new_logical_call_id(self, request_type: str) -> str | None:
        if not self.enabled:
            return None
        with self._lock:
            self._logical_call_counter += 1
            return (
                f"{self.run_id}:{request_type}:"
                f"{self._logical_call_counter:06d}"
            )

    def invoke(
        self,
        model: Any,
        prompt: str,
        *,
        request_type: str,
        function_name: str,
        attempt_number: int,
        logical_call_id: str | None,
    ) -> Any:
        """Invoke an LLM and record the physical request without storing its prompt."""

        if not self.enabled:
            return invoke_chat_model(model, prompt, attempt_number)

        start_timestamp = _timestamp()
        start_perf = time.perf_counter()
        response = None
        status = "success"
        error = None
        input_tokens = None
        output_tokens = None
        total_tokens = None
        seed_for_attempt = getattr(model, "seed_for_attempt", None)
        request_seed = (
            seed_for_attempt(attempt_number)
            if callable(seed_for_attempt)
            else None
        )
        request_temperature = (
            getattr(model, "retry_temperature", None)
            if attempt_number > 1
            else getattr(model, "temperature", None)
        )
        try:
            response = invoke_chat_model(model, prompt, attempt_number)
            input_tokens, output_tokens, total_tokens = self.extract_usage(response)
            return response
        except Exception as exc:
            status = "failure"
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            response_content = getattr(response, "content", None)
            record = {
                "run_id": self.run_id,
                "stage": self.stage,
                **self._current_context(),
                "request_type": request_type,
                "function": function_name,
                "logical_call_id": logical_call_id,
                "attempt_number": attempt_number,
                "is_retry": attempt_number > 1,
                "start_timestamp": start_timestamp,
                "end_timestamp": _timestamp(),
                "wall_clock_seconds": round(time.perf_counter() - start_perf, 6),
                "provider": getattr(self.cfg.llm, "provider", None),
                "model": getattr(self.cfg.llm, "model", None),
                "model_class": (
                    f"{type(model).__module__}.{type(model).__qualname__}"
                ),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "seed": getattr(response, "seed", request_seed),
                "temperature": getattr(
                    response,
                    "temperature",
                    request_temperature,
                ),
                "finish_reason": getattr(response, "finish_reason", None),
                "status": status,
                "error": error,
                "response": response_content,
            }
            with self._lock:
                self._request_counter += 1
                record["request_id"] = self._request_counter
                safe_record = _json_safe(record)
                self._requests.append(safe_record)
                self._request_file.write(json.dumps(safe_record) + "\n")
                self._request_file.flush()

    def record_unit(self, record: dict[str, Any]) -> None:
        if not self.enabled:
            return
        safe_record = _json_safe({
            "run_id": self.run_id,
            "stage": self.stage,
            **record,
        })
        with self._lock:
            self._units.append(safe_record)
            self._unit_file.write(json.dumps(safe_record) + "\n")
            self._unit_file.flush()

    def usage_for_context(self, **filters: Any) -> dict[str, Any]:
        if not self.enabled:
            return {}
        with self._lock:
            records = [
                record
                for record in self._requests
                if all(
                    record.get(field) == value
                    for field, value in filters.items()
                )
            ]
        return self._usage_summary(records)

    def start_cycle(
        self,
        cycle: list[str] | tuple[str, ...],
    ) -> str | None:
        if not self.enabled:
            return None
        with self._lock:
            self._cycle_counter += 1
            cycle_id = f"{self.run_id}:cycle:{self._cycle_counter:06d}"
            nodes = list(cycle)
            record = {
                "cycle_id": cycle_id,
                "run_id": self.run_id,
                "stage": self.stage,
                **self._current_context(),
                "detected_timestamp": _timestamp(),
                "nodes": nodes,
                "cycle_edges": [
                    [nodes[index], nodes[index + 1]]
                    for index in range(max(0, len(nodes) - 1))
                ],
                "llm_resolutions": [],
                "auto_removed_edges": [],
                "manual_input": None,
                "manual_removed_edges": [],
                "resolution_mode": None,
                "status": "detected",
            }
            self._cycles.append(record)
            self._write_cycles()
            return cycle_id

    def record_cycle_llm_resolution(
        self,
        cycle_id: str | None,
        *,
        resolution_attempt: int,
        proposed_edges: Any,
        valid_edges: Any,
    ) -> None:
        with self._lock:
            record = self._cycle_record(cycle_id)
            if record is None:
                return
            record["llm_resolutions"].append({
                "resolution_attempt": resolution_attempt,
                "proposed_edges": proposed_edges,
                "valid_existing_edges": valid_edges,
            })
            self._write_cycles()

    def finish_cycle_automatic(
        self,
        cycle_id: str | None,
        *,
        removed_edges: Any,
        resolved: bool,
    ) -> None:
        with self._lock:
            record = self._cycle_record(cycle_id)
            if record is None:
                return
            record["auto_removed_edges"] = removed_edges
            record["resolution_mode"] = (
                "automatic" if resolved else "awaiting_manual"
            )
            record["status"] = "resolved" if resolved else "unresolved"
            record["automatic_end_timestamp"] = _timestamp()
            self._write_cycles()

    def record_cycle_manual(
        self,
        cycle: list[str] | tuple[str, ...],
        *,
        raw_input: str,
        removed_edges: Any,
        status: str,
        error: str | None = None,
        decision_requested_timestamp: str | None = None,
        decision_applied_timestamp: str | None = None,
        human_wait_seconds: float | None = None,
    ) -> None:
        if not self.enabled:
            return
        normalized = list(cycle)
        with self._lock:
            record = next(
                (
                    item
                    for item in reversed(self._cycles)
                    if item["nodes"] == normalized
                    and item["status"] == "unresolved"
                ),
                None,
            )
            if record is None:
                return
            record["manual_input"] = raw_input
            record["manual_removed_edges"] = removed_edges
            record["resolution_mode"] = "manual"
            record["status"] = status
            record["manual_error"] = error
            record["manual_end_timestamp"] = _timestamp()
            record["decision_requested_timestamp"] = (
                decision_requested_timestamp
            )
            record["decision_applied_timestamp"] = decision_applied_timestamp
            record["human_wait_seconds"] = human_wait_seconds
            self._write_cycles()

    def _cycle_record(self, cycle_id: str | None) -> dict[str, Any] | None:
        if not self.enabled or cycle_id is None:
            return None
        return next(
            (record for record in self._cycles if record["cycle_id"] == cycle_id),
            None,
        )

    def _write_cycles(self) -> None:
        if self.enabled:
            atomic_write_json(
                self._cycles,
                os.path.join(self.run_dir, "cycle_resolutions.json"),
            )

    def finish(
        self,
        *,
        error: str | None = None,
        status: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        self._end_timestamp = _timestamp()
        total_duration = round(time.perf_counter() - self._start_perf, 6)
        if self._request_file is not None:
            self._request_file.flush()
            self._request_file.close()
            self._request_file = None
        if self._unit_file is not None:
            self._unit_file.flush()
            self._unit_file.close()
            self._unit_file = None

        summary = {
            "schema_version": 1,
            "run_id": self.run_id,
            "stage": self.stage,
            "status": status or ("failure" if error else "success"),
            "error": error,
            "start_timestamp": self._start_timestamp,
            "end_timestamp": self._end_timestamp,
            "total_wall_clock_seconds": total_duration,
            "console_log": self._console_log,
            "provider": getattr(self.cfg.llm, "provider", None),
            "model": getattr(self.cfg.llm, "model", None),
            "temperature": getattr(self.cfg.llm, "temperature", None),
            "retry_temperature": getattr(
                self.cfg.llm,
                "retry_temperature",
                None,
            ),
            "top_p": getattr(self.cfg.llm, "top_p", None),
            "seed": getattr(self.cfg.llm, "seed", None),
            **self._usage_summary(self._requests),
            "unit_count": len(self._units),
            "cycle_count": len(self._cycles),
            "cycles_automatically_resolved": sum(
                record["resolution_mode"] == "automatic"
                for record in self._cycles
            ),
            "cycles_manually_resolved": sum(
                record["resolution_mode"] == "manual"
                and record["status"] == "resolved"
                for record in self._cycles
            ),
            "cycles_unresolved": sum(
                record["status"] in {"unresolved", "skipped", "aborted", "invalid"}
                for record in self._cycles
            ),
            "total_human_wait_seconds": round(
                sum(
                    record.get("human_wait_seconds") or 0
                    for record in self._cycles
                ),
                6,
            ),
            "metadata": metadata or {},
        }
        atomic_write_json(summary, os.path.join(self.run_dir, "run_summary.json"))
        self._write_stage_summary()

    @staticmethod
    def _usage_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
        by_type: dict[str, dict[str, Any]] = {}
        for record in records:
            request_type = record["request_type"]
            bucket = by_type.setdefault(request_type, {
                "attempts": 0,
                "successful_attempts": 0,
                "failed_attempts": 0,
                "retries": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "missing_token_usage_records": 0,
                "logical_call_ids": set(),
            })
            bucket["attempts"] += 1
            bucket["successful_attempts"] += record["status"] == "success"
            bucket["failed_attempts"] += record["status"] == "failure"
            bucket["retries"] += bool(record["is_retry"])
            bucket["input_tokens"] += record["input_tokens"] or 0
            bucket["output_tokens"] += record["output_tokens"] or 0
            bucket["total_tokens"] += record["total_tokens"] or 0
            bucket["missing_token_usage_records"] += any(
                record[field] is None
                for field in ("input_tokens", "output_tokens", "total_tokens")
            )
            if record["logical_call_id"] is not None:
                bucket["logical_call_ids"].add(record["logical_call_id"])

        serializable_by_type = {}
        for request_type, bucket in by_type.items():
            logical_ids = bucket.pop("logical_call_ids")
            serializable_by_type[request_type] = {
                "logical_calls": len(logical_ids),
                **bucket,
            }

        logical_calls = {
            record["logical_call_id"]
            for record in records
            if record["logical_call_id"] is not None
        }
        return {
            "llm_logical_calls": len(logical_calls),
            "llm_attempts": len(records),
            "successful_attempts": sum(
                record["status"] == "success" for record in records
            ),
            "failed_attempts": sum(
                record["status"] == "failure" for record in records
            ),
            "retries": sum(bool(record["is_retry"]) for record in records),
            "input_tokens": sum(record["input_tokens"] or 0 for record in records),
            "output_tokens": sum(record["output_tokens"] or 0 for record in records),
            "total_tokens": sum(record["total_tokens"] or 0 for record in records),
            "missing_token_usage_records": sum(
                any(
                    record[field] is None
                    for field in ("input_tokens", "output_tokens", "total_tokens")
                )
                for record in records
            ),
            "usage_by_request_type": serializable_by_type,
        }

    def _write_stage_summary(self) -> None:
        run_summaries = []
        runs_dir = os.path.join(self.telemetry_dir, "runs")
        if os.path.isdir(runs_dir):
            for run_name in sorted(os.listdir(runs_dir)):
                summary_path = os.path.join(runs_dir, run_name, "run_summary.json")
                if not os.path.exists(summary_path):
                    continue
                with open(summary_path, encoding="utf-8") as handle:
                    run_summaries.append(json.load(handle))

        usage_by_request_type: dict[str, dict[str, int]] = {}
        for run in run_summaries:
            for request_type, usage in run.get(
                "usage_by_request_type",
                {},
            ).items():
                bucket = usage_by_request_type.setdefault(request_type, {
                    "logical_calls": 0,
                    "attempts": 0,
                    "successful_attempts": 0,
                    "failed_attempts": 0,
                    "retries": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "missing_token_usage_records": 0,
                })
                for field in bucket:
                    bucket[field] += usage.get(field, 0)

        latest = run_summaries[-1] if run_summaries else {}
        summary = {
            "schema_version": 1,
            "stage": self.stage,
            "status": latest.get("status"),
            "latest_run_id": latest.get("run_id"),
            "first_start_timestamp": (
                run_summaries[0].get("start_timestamp")
                if run_summaries
                else None
            ),
            "latest_end_timestamp": latest.get("end_timestamp"),
            "run_count": len(run_summaries),
            "successful_runs": sum(
                run.get("status") == "success" for run in run_summaries
            ),
            "failed_runs": sum(
                run.get("status") == "failure" for run in run_summaries
            ),
            "awaiting_human_decision_runs": sum(
                run.get("status") == "awaiting_human_decision"
                for run in run_summaries
            ),
            "aborted_runs": sum(
                run.get("status") == "aborted" for run in run_summaries
            ),
            "total_stage_active_wall_clock_seconds": round(
                sum(run.get("total_wall_clock_seconds", 0) for run in run_summaries),
                6,
            ),
            "llm_logical_calls": sum(
                run.get("llm_logical_calls", 0) for run in run_summaries
            ),
            "llm_attempts": sum(
                run.get("llm_attempts", 0) for run in run_summaries
            ),
            "successful_attempts": sum(
                run.get("successful_attempts", 0) for run in run_summaries
            ),
            "failed_attempts": sum(
                run.get("failed_attempts", 0) for run in run_summaries
            ),
            "retries": sum(run.get("retries", 0) for run in run_summaries),
            "input_tokens": sum(
                run.get("input_tokens", 0) for run in run_summaries
            ),
            "output_tokens": sum(
                run.get("output_tokens", 0) for run in run_summaries
            ),
            "total_tokens": sum(
                run.get("total_tokens", 0) for run in run_summaries
            ),
            "missing_token_usage_records": sum(
                run.get("missing_token_usage_records", 0)
                for run in run_summaries
            ),
            "usage_by_request_type": usage_by_request_type,
            "cycle_count": sum(
                run.get("cycle_count", 0) for run in run_summaries
            ),
            "cycles_automatically_resolved": sum(
                run.get("cycles_automatically_resolved", 0)
                for run in run_summaries
            ),
            "cycles_manually_resolved": sum(
                run.get("cycles_manually_resolved", 0)
                for run in run_summaries
            ),
            "cycles_unresolved": sum(
                run.get("cycles_unresolved", 0) for run in run_summaries
            ),
            "total_human_wait_seconds": round(
                sum(
                    run.get("total_human_wait_seconds", 0)
                    for run in run_summaries
                ),
                6,
            ),
            "latest_metadata": latest.get("metadata", {}),
            "runs": [
                {
                    "run_id": run.get("run_id"),
                    "status": run.get("status"),
                    "start_timestamp": run.get("start_timestamp"),
                    "end_timestamp": run.get("end_timestamp"),
                    "wall_clock_seconds": run.get("total_wall_clock_seconds"),
                }
                for run in run_summaries
            ],
        }
        atomic_write_json(
            summary,
            os.path.join(self.telemetry_dir, self.stage_summary_filename),
        )
