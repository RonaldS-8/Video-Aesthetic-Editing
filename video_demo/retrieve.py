"""Segment index (text and image vectors), scenes, and hybrid retrieval.

Each segment gets two vectors from EmbeddingGemma 2: one for its tag text
(description, scene, action, subjects, tags) and the mean of its keyframe
image vectors. Consecutive segments are grouped into scenes by how alike
their vectors are. Each scene slot of an outline part is matched three ways
against the scenes inside the part's chunk range: text cosine, image cosine
(a scene's vectors are the mean of its segments') and character-bigram BM25
over the tag text; the three rankings are fused with reciprocal rank fusion.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image

INDEX_VERSION = 2  # 2: segment text includes overlapping speech
SAVE_EVERY = 20  # segments between partial saves, so a run can resume
RRF_K = 60  # the usual constant; damps the weight of the very top ranks
BM25_K1, BM25_B = 1.5, 0.75


class RetrievalError(RuntimeError):
    """The index is missing, stale, or from another model."""


class Embedder(Protocol):
    model: str

    def embed_text(self, text: str) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...

    def embed_image(self, image: Image.Image) -> np.ndarray: ...


def segment_text(tags: dict[str, Any], speech: str = "") -> str:
    """What a segment is about, as one text for vectors and BM25; ``speech``
    is what is said during it (subtitles), so instructions can match it."""
    parts = [
        tags["description"],
        tags["scene"],
        tags["action"],
        "、".join(tags["subjects"]),
        "、".join(tags["tags"]),
    ]
    if tags.get("landmark"):
        parts.append(f"地标：{tags['landmark']}")
    if speech:
        parts.append(f"台词：{speech}")
    return "；".join(parts)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def build_index(
    path: Path,
    model: str,
    segments: list[tuple[int, dict[str, Any], list[Path], str]],
    embedder: Embedder,
    speech: str,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Vectors for (segment index, tags, keyframe files, speech); resumes
    from ``path`` when it holds a partial index from the same model and
    speech (``speech`` fingerprints the subtitles, "" for none)."""
    done: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if path.exists():
        old = load_index(path, model, check_complete=None, speech=speech)
        done = {
            int(i): (t, m)
            for i, t, m in zip(old["index"], old["text"], old["image"], strict=True)
        }
    for n, (idx, tags, frames, said) in enumerate(segments, start=1):
        if idx not in done:
            if not frames:
                raise RetrievalError(f"segment {idx} has no keyframes")
            text = embedder.embed_text(segment_text(tags, said))
            images = [embedder.embed_image(Image.open(f)) for f in frames]
            done[idx] = (text, _unit(np.mean(images, axis=0)))
            if len(done) % SAVE_EVERY == 0:
                _save(path, model, done, len(segments), speech)
        if progress:
            progress(n, len(segments))
    _save(path, model, done, len(segments), speech)
    return load_index(path, model, check_complete=len(segments), speech=speech)


def _save(
    path: Path,
    model: str,
    done: dict[int, tuple[np.ndarray, np.ndarray]],
    total: int,
    speech: str,
) -> None:
    keys = sorted(done)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial.npz")
    np.savez(
        partial,
        meta=json.dumps(
            {"version": INDEX_VERSION, "model": model, "total": total, "speech": speech}
        ),
        index=np.array(keys),
        text=np.stack([done[k][0] for k in keys]),
        image=np.stack([done[k][1] for k in keys]),
    )
    partial.replace(path)


def load_index(
    path: Path, model: str, check_complete: int | None, speech: str | None = None
) -> dict[str, Any]:
    """The index at ``path``; raises unless it is from ``model`` (and, with
    ``check_complete``, has that many segments; with ``speech``, was built
    from subtitles with that fingerprint)."""
    if not path.exists():
        raise RetrievalError(f"{path} not found; run `python -m video_demo index`")
    with np.load(path) as data:
        meta = json.loads(str(data["meta"]))
        out = {k: data[k] for k in ("index", "text", "image")}
    if meta["version"] != INDEX_VERSION or meta["model"] != model:
        raise RetrievalError(
            f"{path} is from {meta['model']} v{meta['version']}, "
            f"need {model} v{INDEX_VERSION}; rebuild with --force"
        )
    if speech is not None and meta["speech"] != speech:
        raise RetrievalError(
            f"{path} was built from other subtitles; rebuild with --force"
        )
    if check_complete is not None and len(out["index"]) != check_complete:
        raise RetrievalError(
            f"{path} has {len(out['index'])}/{check_complete} segments; "
            "run `python -m video_demo index` to finish it"
        )
    return out


def read_model(path: Path) -> str:
    """The model id an index was built with (no embedder needed)."""
    with np.load(path) as data:
        return json.loads(str(data["meta"]))["model"]


# --- BM25 over character bigrams (no Chinese tokenizer needed) -------------


def bigrams(text: str) -> list[str]:
    chars = [c for c in text if c.isalnum()]
    return ["".join(chars[i : i + 2]) for i in range(len(chars) - 1)] or chars


def bm25(query: str, docs: list[str]) -> list[float]:
    """BM25 score of each doc for the query, terms are character bigrams."""
    toks = [Counter(bigrams(d)) for d in docs]
    n = len(docs)
    avg = sum(sum(t.values()) for t in toks) / n
    df = Counter(term for t in toks for term in t)
    scores = []
    for t in toks:
        length = sum(t.values())
        s = 0.0
        for term in set(bigrams(query)):
            if term not in t:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            tf = t[term]
            s += (
                idf
                * tf
                * (BM25_K1 + 1)
                / (tf + BM25_K1 * (1 - BM25_B + BM25_B * length / avg))
            )
        scores.append(s)
    return scores


def rrf(rankings: Iterable[list[int]], k: int = RRF_K) -> dict[int, float]:
    """Reciprocal rank fusion of rankings (best first)."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused[item] = fused.get(item, 0.0) + 1.0 / (k + rank)
    return fused


def rank_by(scores: dict[int, float]) -> list[int]:
    return sorted(scores, key=lambda i: (-scores[i], i))


def relevance(
    query: str,
    query_vec: np.ndarray,
    members: list[int],
    texts: dict[int, str],
    text_vecs: dict[int, np.ndarray],
    image_vecs: dict[int, np.ndarray],
) -> tuple[dict[int, float], dict[str, list[int]]]:
    """(relevance 0-1 per member, the three rankings) for one query.

    Relevance is the fused RRF score divided by the best possible one
    (rank 1 in all three), so it does not depend on the pool size.
    """
    text_sim = {i: float(text_vecs[i] @ query_vec) for i in members}
    image_sim = {i: float(image_vecs[i] @ query_vec) for i in members}
    bm = dict(zip(members, bm25(query, [texts[i] for i in members]), strict=True))
    rankings = {
        "text": rank_by(text_sim),
        "image": rank_by(image_sim),
        "bm25": rank_by(bm),
    }
    fused = rrf(rankings.values())
    best = len(rankings) / (RRF_K + 1)
    return {i: fused[i] / best for i in members}, rankings


# --- scenes ------------------------------------------------------------------

# A scene is a run of consecutive segments that show the same place or
# moment; an outline slot is filled from one scene, not from loose segments.
# Shot cuts do not find them in hour-long walks (Edinburgh: 3 shots in
# 67 minutes), so a scene ends where the content changes: the similarity of
# the mean vectors of SCENE_WINDOW segments either side of a boundary
# (text and image averaged) is a local minimum below SCENE_CUT_SIM. On
# Edinburgh 0.89 is about the 15th percentile and gives 63 scenes (median
# 50 s) that match what the tags describe: cathedral front, its interior,
# a market, the street with the double-decker buses.
SCENE_WINDOW = 2
SCENE_CUT_SIM = 0.89
MIN_SCENE_SEC, MAX_SCENE_SEC = 20.0, 120.0
# Clips shot on a trip (or the shots of a cut vlog) are a few seconds each,
# so a scene may span shot cuts and video boundaries: such a boundary is
# joinable when the content either side is at least this similar, and a
# hard cut otherwise, even if that leaves a scene shorter than
# MIN_SCENE_SEC. On the vlog collection (145 shots in 13.5 min) 0.80 gives
# 33 scenes (median 25 s) that each keep to one place, e.g. Chinatown
# lanterns, the forest walk, the Merlion by day and night; at 0.78 the
# first unrelated videos join. Edinburgh (3 shot cuts) goes 61 -> 60.
FRAGMENT_JOIN_SIM = 0.80


def pooled(vectors: list[np.ndarray]) -> np.ndarray:
    """Unit-length mean, e.g. a scene's vector from its segments'."""
    return _unit(np.mean(vectors, axis=0))


def boundary_similarity(
    text: np.ndarray, image: np.ndarray, window: int = SCENE_WINDOW
) -> list[float]:
    """Similarity across each boundary i|i+1 of consecutive segments."""

    def across(v: np.ndarray, i: int) -> float:
        before = _unit(v[max(0, i - window + 1) : i + 1].mean(axis=0))
        return float(before @ _unit(v[i + 1 : i + 1 + window].mean(axis=0)))

    return [(across(text, i) + across(image, i)) / 2 for i in range(len(text) - 1)]


def split_scenes(
    starts: list[float],
    ends: list[float],
    sim: list[float],
    hard: set[int],
    soft: set[int] | None = None,
) -> list[list[int]]:
    """Positions 0..n-1 grouped into scenes.

    Cuts at every boundary in ``hard`` and ``soft`` and at local similarity
    minima below SCENE_CUT_SIM. Scenes shorter than MIN_SCENE_SEC then join
    the more similar neighbour (across a soft cut, never across a hard
    one); scenes longer than MAX_SCENE_SEC are cut at their least similar
    boundary that leaves both sides at least MIN_SCENE_SEC.
    """
    n = len(starts)
    if len(sim) != n - 1:
        raise ValueError(f"{len(sim)} similarities for {n} segments")
    cuts = set(hard) | set(soft or ())
    for i, s in enumerate(sim):
        left = sim[i - 1] if i > 0 else math.inf
        right = sim[i + 1] if i + 1 < len(sim) else math.inf
        if s < SCENE_CUT_SIM and s <= min(left, right):
            cuts.add(i)

    def groups() -> list[list[int]]:
        out = [[0]]
        for i in range(1, n):
            if i - 1 in cuts:
                out.append([])
            out[-1].append(i)
        return out

    def length(g: list[int]) -> float:
        return ends[g[-1]] - starts[g[0]]

    def join_short() -> bool:
        for g in groups():
            if length(g) >= MIN_SCENE_SEC:
                continue
            sides = [b for b in (g[0] - 1, g[-1]) if 0 <= b < n - 1 and b not in hard]
            if sides:
                cuts.discard(max(sides, key=lambda b: sim[b]))
                return True
        return False

    def split_long() -> bool:
        for g in groups():
            if length(g) <= MAX_SCENE_SEC:
                continue
            inner = [
                i
                for i in g[:-1]
                if ends[i] - starts[g[0]] >= MIN_SCENE_SEC
                and ends[g[-1]] - starts[i + 1] >= MIN_SCENE_SEC
            ]
            if inner:
                cuts.add(min(inner, key=lambda i: sim[i]))
                return True
        return False

    while join_short():
        pass
    while split_long():
        pass
    return groups()
