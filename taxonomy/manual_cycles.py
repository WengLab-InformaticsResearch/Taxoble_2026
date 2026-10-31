from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from typing import Any, Iterable

from .cycle_utils import find_next_cycle
from .telemetry import atomic_write_json


class ManualCycleDecisionRequired(RuntimeError):
    """Raised after unresolved cycles are persisted for human review."""

    def __init__(self, decision_file: str, cycle_count: int) -> None:
        self.decision_file = decision_file
        self.cycle_count = cycle_count
        super().__init__(
            f"{cycle_count} unresolved cycle(s) require human decisions. "
            f"Edit {decision_file} and rerun with resume=True."
        )


class ManualCycleResolutionAborted(RuntimeError):
    """Raised when a human explicitly aborts cycle resolution."""


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _cycle_edges(cycle: Iterable[str]) -> list[list[str]]:
    nodes = list(cycle)
    return [
        [nodes[index], nodes[index + 1]]
        for index in range(max(0, len(nodes) - 1))
    ]


def _cycle_key(cycle: Iterable[str]) -> str:
    payload = json.dumps(list(cycle), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


class FileCycleDecisionStore:
    """Persist unresolved cycles and apply validated file-based decisions."""

    def __init__(
        self,
        path: str,
        *,
        domain: str,
        experiment_name: str,
    ) -> None:
        self.path = path
        self.domain = domain
        self.experiment_name = experiment_name

    def resolve_or_pause(
        self,
        edges: Iterable[tuple[str, str]],
        unresolved_cycles: Iterable[Iterable[str]],
        *,
        telemetry: Any = None,
    ) -> list[tuple[str, str]]:
        edge_set = {tuple(edge) for edge in edges}
        cycles = [list(cycle) for cycle in unresolved_cycles]
        previous = self._load()
        previous_by_key = {
            item.get("cycle_key"): item
            for item in previous.get("cycles", [])
        }

        requested_at = previous.get("requested_at") or _timestamp()
        records = [
            self._cycle_record(
                cycle,
                previous_by_key.get(_cycle_key(cycle)),
                requested_at=requested_at,
            )
            for cycle in cycles
        ]
        document = self._document(
            records,
            status="awaiting_human_decision",
            requested_at=requested_at,
        )

        pending = []
        for record, cycle in zip(records, cycles):
            cycle_edge_set = {
                tuple(edge)
                for edge in record["cycle_edges"]
            }
            if not cycle_edge_set.issubset(edge_set):
                record["status"] = "resolved_by_other_decision"
                if telemetry:
                    telemetry.record_cycle_manual(
                        cycle,
                        raw_input=json.dumps(
                            {"action": "resolved_by_other_decision"}
                        ),
                        removed_edges=[],
                        status="resolved",
                    )
                continue

            decision = record["decision"]
            action = str(decision.get("action", "pending")).strip().lower()
            decision["action"] = action
            decision.pop("error", None)

            if action in {"", "pending", "defer"}:
                pending.append(record)
                continue
            if action == "abort":
                document["status"] = "aborted"
                document["updated_at"] = _timestamp()
                atomic_write_json(document, self.path)
                raise ManualCycleResolutionAborted(
                    f"Cycle resolution was aborted in {self.path}."
                )
            if action != "remove_edges":
                decision["error"] = (
                    "action must be 'pending', 'remove_edges', or 'abort'"
                )
                pending.append(record)
                continue

            try:
                remove_edges = self._validate_remove_edges(
                    decision.get("remove_edges"),
                    cycle,
                    edge_set,
                )
            except ValueError as exc:
                decision["error"] = str(exc)
                pending.append(record)
                continue

            for edge in remove_edges:
                edge_set.remove(edge)
            applied_at = _timestamp()
            record["status"] = "applied"
            record["decision_applied_at"] = applied_at
            requested = _parse_timestamp(record.get("decision_requested_at"))
            applied = _parse_timestamp(applied_at)
            human_wait_seconds = (
                round((applied - requested).total_seconds(), 6)
                if requested is not None and applied is not None
                else None
            )
            record["human_wait_seconds"] = human_wait_seconds
            if telemetry:
                telemetry.record_cycle_manual(
                    cycle,
                    raw_input=json.dumps(decision, ensure_ascii=False),
                    removed_edges=remove_edges,
                    status="resolved",
                    decision_requested_timestamp=record.get(
                        "decision_requested_at"
                    ),
                    decision_applied_timestamp=applied_at,
                    human_wait_seconds=human_wait_seconds,
                )

        if pending:
            document["status"] = "awaiting_human_decision"
            document["updated_at"] = _timestamp()
            atomic_write_json(document, self.path)
            raise ManualCycleDecisionRequired(self.path, len(pending))

        remaining_cycle = find_next_cycle(list(edge_set))
        if remaining_cycle is not None:
            remaining_record = self._cycle_record(
                list(remaining_cycle),
                previous_by_key.get(_cycle_key(remaining_cycle)),
                requested_at=_timestamp(),
            )
            remaining_record["decision"]["error"] = (
                "The selected removals did not make the taxonomy acyclic."
            )
            document = self._document(
                [remaining_record],
                status="awaiting_human_decision",
                requested_at=remaining_record["decision_requested_at"],
            )
            atomic_write_json(document, self.path)
            raise ManualCycleDecisionRequired(self.path, 1)

        document["status"] = "resolved"
        document["resolved_at"] = _timestamp()
        document["updated_at"] = document["resolved_at"]
        atomic_write_json(document, self.path)
        return list(edge_set)

    def _load(self) -> dict[str, Any]:
        if not os.path.exists(self.path):
            return {}
        with open(self.path, encoding="utf-8") as handle:
            data = json.load(handle)
        if data.get("schema_version") != 1:
            raise ValueError(
                f"Unsupported cycle-decision schema in {self.path}."
            )
        return data

    def _cycle_record(
        self,
        cycle: list[str],
        previous: dict[str, Any] | None,
        *,
        requested_at: str,
    ) -> dict[str, Any]:
        prior_decision = (previous or {}).get("decision", {})
        return {
            "cycle_key": _cycle_key(cycle),
            "nodes": cycle,
            "cycle_edges": _cycle_edges(cycle),
            "status": "pending",
            "decision_requested_at": (
                (previous or {}).get("decision_requested_at")
                or requested_at
            ),
            "decision": {
                "action": prior_decision.get("action", "pending"),
                "remove_edges": prior_decision.get("remove_edges", []),
                "note": prior_decision.get("note", ""),
            },
        }

    def _document(
        self,
        cycles: list[dict[str, Any]],
        *,
        status: str,
        requested_at: str,
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": status,
            "domain": self.domain,
            "experiment_name": self.experiment_name,
            "requested_at": requested_at,
            "updated_at": _timestamp(),
            "instructions": {
                "edge_direction": "[child, parent]",
                "allowed_actions": [
                    "pending",
                    "remove_edges",
                    "abort",
                ],
                "resume": (
                    "For every cycle, set action to remove_edges and select "
                    "one or more entries from cycle_edges, then rerun with "
                    "resume=True."
                ),
            },
            "cycles": cycles,
        }

    @staticmethod
    def _validate_remove_edges(
        raw_edges: Any,
        cycle: Iterable[str],
        current_edges: set[tuple[str, str]],
    ) -> list[tuple[str, str]]:
        if not isinstance(raw_edges, list) or not raw_edges:
            raise ValueError("remove_edges must be a non-empty JSON list")

        cycle_edge_set = {
            tuple(edge)
            for edge in _cycle_edges(cycle)
        }
        parsed = []
        for raw_edge in raw_edges:
            if (
                not isinstance(raw_edge, list)
                or len(raw_edge) != 2
                or not all(isinstance(node, str) for node in raw_edge)
            ):
                raise ValueError(
                    "every remove_edges entry must be [child, parent]"
                )
            edge = tuple(raw_edge)
            if edge not in cycle_edge_set:
                raise ValueError(
                    f"selected edge {list(edge)!r} is not in cycle_edges"
                )
            if edge not in current_edges:
                raise ValueError(
                    f"selected edge {list(edge)!r} is not in the current taxonomy"
                )
            if edge not in parsed:
                parsed.append(edge)
        return parsed
