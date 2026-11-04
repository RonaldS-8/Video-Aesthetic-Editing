"""Step timings as OTLP JSON (an ExportTraceServiceRequest): one span per
step, named by the step id of pipeline.yaml, plus a readable steps.json
with the seconds of each phase."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SERVICE = "anas-video-edit-plan-b"


@dataclass
class Span:
    name: str
    category: str
    start_ns: int
    end_ns: int
    ok: bool
    attributes: dict[str, Any] = field(default_factory=dict)


def _value(v: Any) -> dict[str, Any]:
    if isinstance(v, bool):
        return {"boolValue": v}
    if isinstance(v, int):
        return {"intValue": str(v)}
    if isinstance(v, float):
        return {"doubleValue": v}
    return {"stringValue": str(v)}


class Trace:
    def __init__(self) -> None:
        self.trace_id = os.urandom(16).hex()
        self.spans: list[Span] = []

    @contextmanager
    def step(self, name: str, category: str, **attributes: Any) -> Iterator[Span]:
        """Time the block as the step ``name``; a step that raises is kept
        with an error status and the exception goes on."""
        span = Span(name, category, time.time_ns(), 0, True, dict(attributes))
        try:
            yield span
        except BaseException:
            span.ok = False
            raise
        finally:
            span.end_ns = time.time_ns()
            self.spans.append(span)

    def call(
        self, name: str, start_ns: int, seconds: float, attributes: dict[str, Any]
    ) -> None:
        """A planner call made inside the edit step."""
        end_ns = start_ns + int(seconds * 1e9)
        self.spans.append(Span(name, "arrange", start_ns, end_ns, True, attributes))

    def otlp(self) -> dict[str, Any]:
        spans = [
            {
                "traceId": self.trace_id,
                "spanId": os.urandom(8).hex(),
                "name": s.name,
                "kind": 1,
                "startTimeUnixNano": str(s.start_ns),
                "endTimeUnixNano": str(s.end_ns),
                "attributes": [
                    {"key": k, "value": _value(v)}
                    for k, v in {"anas.category": s.category, **s.attributes}.items()
                ],
                "status": {"code": 1 if s.ok else 2},
            }
            for s in self.spans
        ]
        return {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [
                            {"key": "service.name", "value": _value(SERVICE)}
                        ]
                    },
                    "scopeSpans": [{"scope": {"name": "ab_eval"}, "spans": spans}],
                }
            ]
        }

    def write(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "trace.json").write_text(
            json.dumps(self.otlp(), ensure_ascii=False, indent=1), encoding="utf-8"
        )
        steps = [
            {
                "step": s.name,
                "category": s.category,
                "seconds": round((s.end_ns - s.start_ns) / 1e9, 2),
                "ok": s.ok,
                **s.attributes,
            }
            for s in self.spans
        ]
        # Seconds per phase (the step attribute "phase"): the footage's
        # analysis and the cut for the instruction are timed apart.
        phases: dict[str, float] = {}
        for s in self.spans:
            if "phase" in s.attributes:
                name = s.attributes["phase"]
                phases[name] = phases.get(name, 0.0) + (s.end_ns - s.start_ns) / 1e9
        summary = {
            "phases": {k: round(v, 2) for k, v in phases.items()},
            "steps": steps,
        }
        text = json.dumps(summary, ensure_ascii=False, indent=1)
        (folder / "steps.json").write_text(text, encoding="utf-8")
        # A run resumed after a pause (a planner answer to wait for) writes
        # steps.json again; each run's steps are also kept under runs/.
        if self.spans:
            started = time.localtime(self.spans[0].start_ns / 1e9)
            runs = folder / "runs"
            runs.mkdir(exist_ok=True)
            name = time.strftime("steps-%Y%m%d-%H%M%S.json", started)
            (runs / name).write_text(text, encoding="utf-8")
