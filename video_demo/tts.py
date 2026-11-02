"""Narration voice: Kokoro-82M v1.1-zh (StyleTTS 2, 82 M parameters, 103
voices) on the CPU through the kokoro package; weights from ModelScope
(hexgrad/Kokoro-82M-v1.1-zh), see models/README.md."""

from __future__ import annotations

from pathlib import Path

import numpy as np

KOKORO_DIR = "Kokoro-82M-v1.1-zh"
KOKORO_REPO = "hexgrad/Kokoro-82M-v1.1-zh"
KOKORO_WEIGHTS = "kokoro-v1_1-zh.pth"
KOKORO_RATE = 24_000
# Picked by listening among the clearest voices (see README); also the
# fastest of them, so lines fit their gaps more easily.
DEFAULT_VOICE = "zf_088"


class TTSError(RuntimeError):
    """The voice model is missing or produced no audio."""


class KokoroAdapter:
    """Mandarin text to mono float32 speech at KOKORO_RATE. Everything is
    read from ``models_dir``; nothing is downloaded."""

    rate = KOKORO_RATE

    def __init__(self, models_dir: Path, voice: str = DEFAULT_VOICE) -> None:
        root = models_dir / KOKORO_DIR
        files = (
            root / "config.json",
            root / KOKORO_WEIGHTS,
            root / "voices" / f"{voice}.pt",
        )
        for need in files:
            if not need.exists():
                raise TTSError(f"{need} not found; see models/README.md (Kokoro)")
        from kokoro import KModel, KPipeline

        model = KModel(
            repo_id=KOKORO_REPO, config=str(files[0]), model=str(files[1])
        ).eval()
        self._pipeline = KPipeline(
            lang_code="z", repo_id=KOKORO_REPO, model=model, en_callable=_no_english
        )
        self._voice = str(files[2])
        self.voice = voice

    def synthesize(self, text: str) -> np.ndarray:
        chunks = [
            r.audio.numpy()
            for r in self._pipeline(text, voice=self._voice, speed=1.0)
            if r.audio is not None
        ]
        if not chunks:
            raise TTSError(f"no audio for {text!r}")
        return np.concatenate(chunks).astype(np.float32)


def _no_english(text: str) -> str:
    """Narration is Mandarin only (the English G2P would need extra models)."""
    raise TTSError(f"narration has English words: {text!r}")
