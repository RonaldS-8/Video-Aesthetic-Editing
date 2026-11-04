from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from video_demo import embed, retrieve


def _tags(description: str, tags: list[str]) -> dict:
    return {
        "description": description,
        "scene": "街道",
        "action": "行走",
        "subjects": ["行人"],
        "tags": tags,
    }


class FakeEmbedder:
    """Vectors from a hash of the input; counts calls."""

    model = "fake@1"

    def __init__(self) -> None:
        self.calls = 0

    def _vec(self, seed: int) -> np.ndarray:
        self.calls += 1
        v = np.random.default_rng(seed).normal(size=8).astype(np.float32)
        return v / np.linalg.norm(v)

    def embed_text(self, text: str) -> np.ndarray:
        return self._vec(len(text))

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(len(text) + 1)

    def embed_image(self, image: Image.Image) -> np.ndarray:
        return self._vec(int(np.asarray(image)[0, 0, 0]))


def _frames(tmp_path: Path, idx: int, n: int) -> list[Path]:
    paths = []
    for k in range(n):
        path = tmp_path / f"seg{idx:03d}_{k}.jpg"
        Image.new("RGB", (16, 9), (idx * 10 + k, 0, 0)).save(path)
        paths.append(path)
    return paths


def test_segment_text_joins_tag_fields() -> None:
    text = retrieve.segment_text(_tags("教堂外景", ["教堂", "巴士"]))
    assert text == "教堂外景；街道；行走；行人；教堂、巴士"
    spoken = retrieve.segment_text(_tags("教堂外景", ["教堂"]), "这是圣母教堂")
    assert spoken.endswith("；台词：这是圣母教堂")


def test_bm25_prefers_matching_docs() -> None:
    docs = ["红色双层巴士驶过街道", "教堂内部的彩色玻璃窗", "街道上的行人"]
    scores = retrieve.bm25("双层巴士", docs)
    assert scores[0] > 0 and scores[1] == 0 and scores[2] == 0
    assert retrieve.bigrams("巴士 a1") == ["巴士", "士a", "a1"]
    assert retrieve.bigrams("车") == ["车"]


def test_rrf_and_relevance_scale() -> None:
    fused = retrieve.rrf([[1, 2], [2, 1], [1, 2]])
    assert fused[1] > fused[2]
    q = np.array([1.0, 0.0], dtype=np.float32)
    near, far = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    rel, rankings = retrieve.relevance(
        "巴士",
        q,
        [0, 1],
        {0: "双层巴士", 1: "教堂"},
        {0: near, 1: far},
        {0: near, 1: far},
    )
    # First in all three rankings is relevance 1.
    assert rel[0] == pytest.approx(1.0) and rel[1] < rel[0]
    assert rankings["text"] == rankings["image"] == rankings["bm25"] == [0, 1]


def test_build_index_resumes_and_checks_model(tmp_path: Path) -> None:
    path = tmp_path / "index_v.npz"
    segs = [(i, _tags(f"d{i}", ["t"]), _frames(tmp_path, i, 2), "") for i in range(3)]
    emb = FakeEmbedder()
    idx = retrieve.build_index(path, emb.model, segs[:2], emb, "s1")
    assert list(idx["index"]) == [0, 1] and emb.calls == 2 * 3
    # Image vector is the renormalised mean of the keyframes.
    assert np.linalg.norm(idx["image"][0]) == pytest.approx(1.0)
    emb.calls = 0
    idx = retrieve.build_index(path, emb.model, segs, emb, "s1")
    assert list(idx["index"]) == [0, 1, 2] and emb.calls == 3
    with pytest.raises(retrieve.RetrievalError, match="fake@1"):
        retrieve.load_index(path, "other@2", check_complete=3)
    with pytest.raises(retrieve.RetrievalError, match="3/4"):
        retrieve.load_index(path, emb.model, check_complete=4)
    with pytest.raises(retrieve.RetrievalError, match="not found"):
        retrieve.load_index(tmp_path / "none.npz", emb.model, check_complete=1)
    with pytest.raises(retrieve.RetrievalError, match="other subtitles"):
        retrieve.load_index(path, emb.model, check_complete=3, speech="s2")
    with pytest.raises(retrieve.RetrievalError, match="other subtitles"):
        retrieve.build_index(path, emb.model, segs, emb, "s2")
    with pytest.raises(retrieve.RetrievalError, match="no keyframes"):
        retrieve.build_index(path, emb.model, [(9, _tags("x", []), [], "")], emb, "s1")


def test_thumbnail_matches_client_rules() -> None:
    big = Image.new("RGBA", (1920, 1080), (0, 0, 0, 0))
    out = Image.open(io.BytesIO(embed.thumbnail(big)))
    assert out.size == (512, 288) and out.format == "JPEG"
    # Transparent pixels are flattened on white.
    assert out.getpixel((10, 10))[0] > 240
    small = Image.new("RGB", (300, 200), "red")
    assert Image.open(io.BytesIO(embed.thumbnail(small))).size == (300, 200)


def test_verify_model_rejects_missing_and_wrong(tmp_path: Path) -> None:
    manifest = {"source": "s", "revision": "r", "file": "m", "sha256": "0" * 64}
    with pytest.raises(embed.EmbedError, match="not found"):
        embed.verify_model(tmp_path / "m", manifest)
    (tmp_path / "m").write_bytes(b"x")
    with pytest.raises(embed.EmbedError, match="SHA-256"):
        embed.verify_model(tmp_path / "m", manifest)


def test_boundary_similarity_uses_windows() -> None:
    e1, e2 = np.eye(2, dtype=np.float32)
    v = np.stack([e1, e1, e1, e2, e2, e2])
    sim = retrieve.boundary_similarity(v, v)
    assert np.round(sim, 3).tolist() == [1.0, 0.707, 0.0, 0.707, 1.0]
    assert retrieve.pooled([e1, e2]) == pytest.approx([0.7071, 0.7071], abs=1e-4)


def _split(sim: list[float], hard: set[int] | None = None) -> list[list[int]]:
    n = len(sim) + 1
    starts = [10.0 * i for i in range(n)]
    return retrieve.split_scenes(starts, [s + 10.0 for s in starts], sim, hard or set())


def test_split_scenes_cuts_joins_and_splits() -> None:
    flat = [0.95] * 11
    # A dip below the threshold cuts.
    assert _split([*flat[:5], 0.5, *flat[6:]]) == [list(range(6)), list(range(6, 12))]
    # Only the lower of two neighbouring dips.
    dips = [*flat[:5], 0.6, 0.5, *flat[7:]]
    assert _split(dips) == [list(range(7)), list(range(7, 12))]
    # A 10 s scene joins its neighbour, except across a shot cut.
    early = [0.5, *flat[1:5], 0.5, *flat[6:]]
    assert _split(early) == [list(range(6)), list(range(6, 12))]
    assert _split(early, hard={0}) == [[0], list(range(1, 6)), list(range(6, 12))]
    # 150 s without a dip: cut at the least similar boundary.
    long = [0.95] * 14
    long[9] = 0.93
    assert _split(long) == [list(range(10)), list(range(10, 15))]
    with pytest.raises(ValueError, match="similarities"):
        retrieve.split_scenes([0.0, 10.0], [10.0, 20.0], [], set())


def test_split_scenes_joins_short_scenes_across_soft_cuts_only() -> None:
    flat = [0.95] * 5
    # A 10 s fragment at the start: joined to the rest across a soft cut,
    # kept apart by a hard one even though it stays short.
    assert retrieve.split_scenes(
        [10.0 * i for i in range(6)],
        [10.0 * i + 10.0 for i in range(6)],
        flat,
        set(),
        {0},
    ) == [list(range(6))]
    assert retrieve.split_scenes(
        [10.0 * i for i in range(6)],
        [10.0 * i + 10.0 for i in range(6)],
        flat,
        {0},
    ) == [[0], list(range(1, 6))]
