"""Audio module - lightweight, offline speech-activity detection.

Works on WAV only (stdlib `wave` + numpy), so no ffmpeg / system packages are
needed on Streamlit Cloud. Nothing is transcribed: only loudness-based
speech/silence activity is kept.
"""
from __future__ import annotations

import io
import wave

import numpy as np
import pandas as pd


def load_wav(data: bytes) -> tuple[np.ndarray, int]:
    """Decode WAV bytes into a mono float32 signal in [-1, 1]."""
    try:
        with wave.open(io.BytesIO(data)) as wf:
            sr, ch, sw, n = wf.getframerate(), wf.getnchannels(), wf.getsampwidth(), wf.getnframes()
            raw = wf.readframes(n)
    except wave.Error as exc:
        raise ValueError(f"Not a readable WAV file: {exc}") from exc
    if sw not in (1, 2, 4):
        raise ValueError("Only 8/16/32-bit PCM WAV is supported (24-bit is not).")
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[sw]
    x = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    x = (x - 128.0) / 128.0 if sw == 1 else x / float(np.iinfo(dtype).max)
    if ch > 1:
        x = x[: len(x) // ch * ch].reshape(-1, ch).mean(axis=1)
    return x, sr


def audio_activity(x: np.ndarray, sr: int, frame_s: float = 0.5) -> pd.DataFrame:
    """Frame-level loudness (dB) + adaptive speech/silence decision."""
    n = int(sr * frame_s)
    frames = len(x) // n
    if frames < 2:
        raise ValueError("Audio is too short to analyse.")
    chunks = x[: frames * n].reshape(frames, n)
    rms = np.sqrt((chunks ** 2).mean(axis=1))
    db = 20 * np.log10(rms + 1e-9)
    floor, ceil = np.percentile(db, 10), np.percentile(db, 95)
    if ceil - floor < 6:  # basically flat -> nothing to call speech
        speech = np.zeros(frames, dtype=bool)
    else:
        speech = db > floor + max(6.0, 0.3 * (ceil - floor))
    return pd.DataFrame({"t": np.arange(frames) * frame_s, "db": db, "speech": speech})


def window_speech_ratio(audio_df: pd.DataFrame | None, win_s: float, n_win: int) -> np.ndarray:
    """Share of speech frames per analysis window (NaN when no audio)."""
    if audio_df is None or audio_df.empty:
        return np.full(n_win, np.nan)
    w = (audio_df["t"] // win_s).astype(int)
    ratio = audio_df.groupby(w)["speech"].mean()
    return ratio.reindex(range(n_win)).to_numpy(dtype=float)
