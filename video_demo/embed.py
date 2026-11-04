"""Text and image vectors from EmbeddingGemma 2, as the A-NAS client makes them.

Same model file (pinned in models/embeddinggemma/embeddinggemma-2-740m.json,
copied from the client's deploy/models), same runtime (MediaPipe Universal Embedder,
L2-normalised, 70 vision tokens per image), same retrieval prompt for
queries, and images go through the client's photo thumbnail first, so
vectors here are comparable with the client's.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from video_demo import layout

MODEL_DIR = layout.models_dir(Path(__file__).resolve().parent.parent) / (
    "embeddinggemma"
)
MANIFEST = MODEL_DIR / "embeddinggemma-2-740m.json"
# The client's choice: the LiteRT package has 70 and 140 token image
# signatures; 70 was about 3x faster with similar zero-shot results.
VISION_TOKENS = 70
# EmbeddingGemma's retrieval prompt; segment texts and images get none.
QUERY_PROMPT = "task: search result | query: {}"
# The client's photo thumbnail: fit within 512 px (Lanczos), white
# background for transparency, JPEG quality 82.
THUMB_EDGE = 512
THUMB_QUALITY = 82


class EmbedError(RuntimeError):
    """The model file is missing or not the pinned one, or input is invalid."""


def read_manifest(path: Path = MANIFEST) -> dict:
    return json.loads(path.read_text("utf-8"))


def verify_model(model_path: Path, manifest: dict) -> None:
    if not model_path.exists():
        raise EmbedError(
            f"{model_path} not found; download {manifest['source']}/resolve/"
            f"{manifest['revision']}/{manifest['file']}"
        )
    digest = hashlib.sha256()
    with model_path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != manifest["sha256"]:
        raise EmbedError(f"{model_path} does not match the manifest's SHA-256")


def thumbnail(image: Image.Image, edge: int = THUMB_EDGE) -> bytes:
    """JPEG bytes of the client's thumbnail (ai/eval/embed.py ``render``)."""
    image = ImageOps.exif_transpose(image)
    width, height = image.size
    if width > edge or height > edge:
        # imaging.Fit: keep the aspect ratio, truncate the short side.
        if width / height > 1:
            size = (edge, max(1, int(edge / (width / height))))
        else:
            size = (max(1, int(edge * (width / height))), edge)
        image = image.resize(size, Image.Resampling.LANCZOS)
    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGBA")
        flat = Image.new("RGB", image.size, "white")
        flat.paste(image, mask=image.getchannel("A"))
        image = flat
    out = io.BytesIO()
    image.convert("RGB").save(out, format="JPEG", quality=THUMB_QUALITY)
    return out.getvalue()


class EmbeddingGemmaAdapter:
    """EmbeddingGemma 2 through MediaPipe's Universal Embedder."""

    def __init__(self, model_dir: Path = MODEL_DIR, manifest_path: Path = MANIFEST):
        manifest = read_manifest(manifest_path)
        model_path = model_dir / manifest["file"]
        verify_model(model_path, manifest)
        import mediapipe
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.retrieval.universal_embedder import (
            UniversalEmbedder,
            UniversalEmbedderOptions,
        )

        # XNNPack keeps repacked weights here instead of next to the model;
        # the runtime requires the directory to exist.
        cache_dir = (model_dir / "cache").resolve()
        cache_dir.mkdir(exist_ok=True)
        self._embedder = UniversalEmbedder.create_from_options(
            UniversalEmbedderOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                l2_normalize=True,
                vision_tokens_per_image=VISION_TOKENS,
                cache_dir=str(cache_dir),
            )
        )
        self.dimensions = len(self.embed_text("probe"))
        # Same identity string as the client: runtime and token budget
        # change the vectors.
        self.model = (
            f"{manifest['name']}@{manifest['sha256'][:12]}"
            f"+mediapipe-{mediapipe.__version__}+tok{VISION_TOKENS}+l2"
        )

    def embed_text(self, text: str) -> np.ndarray:
        return self._vector(self._embedder.embed_text(text))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_text(QUERY_PROMPT.format(text))

    def embed_image(self, image: Image.Image) -> np.ndarray:
        try:
            result = self._embedder.embed_image(thumbnail(image))
        except (RuntimeError, ValueError) as exc:
            raise EmbedError(str(exc)) from exc
        return self._vector(result)

    def close(self) -> None:
        self._embedder.close()

    @staticmethod
    def _vector(result) -> np.ndarray:  # noqa: ANN001 - MediaPipe result type
        return np.asarray(result.embeddings[0].embedding, dtype=np.float32)
