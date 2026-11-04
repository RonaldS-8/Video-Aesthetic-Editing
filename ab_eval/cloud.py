"""Planners for the calls video_demo.outline lets another model answer
(outline.PLANNER_CALLS: the outline and the review of the chosen clips).

Only text is sent: outline._request keeps prompts with pictures local.

``ClaudeApiPlanner`` calls the Claude API (structured output against the
call's JSON schema); it needs the ``anthropic`` package and credentials
(ANTHROPIC_API_KEY).

Every call is recorded for trace.json.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

DEFAULT_MODEL = "claude-opus-5-5"
# USD per million input / output tokens.
PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0)}
MAX_TOKENS = 64000
# Structured output accepts none of these; the caller's own validation
# (outline._ask) still checks them on the answer.
_UNSUPPORTED = frozenset(
    {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"}
    | {"minLength", "maxLength", "minItems", "maxItems", "uniqueItems"}
)


class PlannerError(RuntimeError):
    pass


@dataclass
class Call:
    name: str
    transport: str
    seconds: float
    prompt_chars: int
    answer_chars: int
    start_ns: int = 0
    upload_bytes: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


def api_schema(schema: Any) -> Any:
    """``schema`` as structured output takes it: no numeric, length or item
    count constraints, and every object closed."""
    if isinstance(schema, list):
        return [api_schema(s) for s in schema]
    if not isinstance(schema, dict):
        return schema
    out = {k: api_schema(v) for k, v in schema.items() if k not in _UNSUPPORTED}
    if out.get("type") == "object":
        out["additionalProperties"] = False
    return out


@dataclass
class ClaudeApiPlanner:
    """Answers through the Claude API; ``client`` is an anthropic.Anthropic
    (see ``connect``)."""

    client: Any
    model: str = DEFAULT_MODEL
    effort: str = "high"
    calls: list[Call] = field(default_factory=list)

    def __call__(
        self,
        system: str,
        prompt: str,
        schema: dict[str, Any],
        name: str,
        temperature: float,
    ) -> tuple[str, float]:
        # Sampling parameters are not accepted by current models; the
        # temperature of outline._ask's retries is not sent.
        request = {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": {
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": api_schema(schema)},
            },
            # A policy decline is answered again by a fallback model.
            "betas": ["server-side-fallback-2026-07-01"],
            "fallbacks": "default",
        }
        start_ns = time.time_ns()
        t0 = time.perf_counter()
        with self.client.beta.messages.stream(**request) as stream:
            message = stream.get_final_message()
        seconds = time.perf_counter() - t0
        if message.stop_reason == "refusal":
            raise PlannerError(f"{name}: the request was declined")
        if message.stop_reason == "max_tokens":
            raise PlannerError(f"{name}: the answer hit max_tokens")
        text = next(b.text for b in message.content if b.type == "text")
        price_in, price_out = PRICES[self.model]
        usage = message.usage
        self.calls.append(
            Call(
                name,
                "api",
                seconds,
                len(prompt),
                len(text),
                start_ns,
                upload_bytes=len(
                    json.dumps(
                        {
                            k: request[k]
                            for k in ("system", "messages", "output_config")
                        },
                        ensure_ascii=False,
                    ).encode("utf-8")
                ),
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cost_usd=(
                    usage.input_tokens * price_in + usage.output_tokens * price_out
                )
                / 1e6,
            )
        )
        return text, seconds


def connect(model: str = DEFAULT_MODEL) -> ClaudeApiPlanner:
    """A planner on a client that finds its credentials in the environment."""
    import anthropic

    return ClaudeApiPlanner(anthropic.Anthropic(), model)
