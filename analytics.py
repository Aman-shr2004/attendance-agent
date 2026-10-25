"""Analytics - sustained-presence attendance, multimodal engagement fusion, insights."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

import audio as audio_mod


def fmt_time(sec: float) -> str:
    sec = int(round(sec))
    return f"{sec // 60:02d}:{sec % 60:02d}"


def _step(times: np.ndarray) -> float:
    return float(np.median(np.diff(times))) if len(times) > 1 else 1.0


def _longest_run(present: np.ndarray, gap_tol: int = 2) -> int:
    """Longest stretch of samples where gaps of <= gap_tol missed samples are forgiven."""
    idx = np.flatnonzero(present)
    if idx.size == 0:
        return 0
    groups = np.split(idx, np.where(np.diff(idx) > gap_tol + 1)[0] + 1)
    return int(max(g[-1] - g[0] + 1 for g in groups))


# --------------------------------------------------------------------------- attendance
def attendance_table(obs: pd.DataFrame, meta: dict, min_sustain_s: float,
                     session_start: dt.datetime) -> pd.DataFrame:
    """Mark attendance on *sustained* presence, not single-frame detections."""
    times = np.asarray(meta["sample_times"], float)
    step = _step(times)
    rows = []
    for pid, g in obs.groupby("pid"):
        present = np.zeros(len(times), bool)
        present[g["k"].to_numpy(int)] = True
        longest = _longest_run(present) * step
        total = present.sum() * step
        first, last = float(g["t"].min()), float(g["t"].max())
        ok = longest >= min_sustain_s and total >= min_sustain_s
        clock = lambda s: (session_start + dt.timedelta(seconds=s)).strftime("%H:%M:%S")
        rows.append(dict(pid=int(pid), status="Present" if ok else "Transient (not counted)",
                         arrived=clock(first), left=clock(last + step),
                         present_min=round(total / 60, 1), longest_min=round(longest / 60, 1),
                         first_seen_s=first, last_seen_s=last))
    cols = ["pid", "status", "arrived", "left", "present_min", "longest_min", "first_seen_s", "last_seen_s"]
    return pd.DataFrame(rows, columns=cols).sort_values("pid").reset_index(drop=True)


def presence_matrix(obs: pd.DataFrame, meta: dict, pids) -> tuple[np.ndarray, list[int]]:
    """0 = not seen, 1 = seen looking away, 2 = seen facing front (per sample)."""
    n = len(meta["sample_times"])
    pids = list(pids)
    z = np.zeros((len(pids), n))
    pos = {p: i for i, p in enumerate(pids)}
    for r in obs[obs["pid"].isin(pids)].itertuples():
        z[pos[r.pid], int(r.k)] = 2 if r.frontal else 1
    return z, pids


# --------------------------------------------------------------------------- engagement
def _movement_score(m: float) -> float:
    """Natural, moderate movement = engaged; frozen or very restless = less so."""
    return float(np.interp(m, [0, 0.02, 0.04, 0.12, 0.30], [0.4, 0.6, 1.0, 1.0, 0.3]))


def engagement_tables(obs: pd.DataFrame, meta: dict, audio_df: pd.DataFrame | None,
                      win_s: float, weights: tuple[float, float, float], pids):
    """Fuse head-orientation, movement and audio activity per participant and window.

    attention  = frontal detections / samples while the person was in the room
                 (no detection = head down or looking away -> counted as not attending)
    movement   = score from mean normalised head displacement
    audio      = class-level speech activity in the window
    """
    times = np.asarray(meta["sample_times"], float)
    n_win = max(1, int(np.ceil(float(meta["duration"]) / win_s)))
    starts = np.arange(n_win) * win_s
    speech = audio_mod.window_speech_ratio(audio_df, win_s, n_win)
    w_att, w_mov, w_aud = weights
    if audio_df is None:
        w_aud = 0.0
    total_w = (w_att + w_mov + w_aud) or 1.0

    rows = []
    for pid, g in obs[obs["pid"].isin(list(pids))].groupby("pid"):
        first, last = g["t"].min(), g["t"].max()
        for wi, ws in enumerate(starts):
            we = ws + win_s
            n_exp = int(((times >= max(ws, first)) & (times < we) & (times <= last)).sum())
            if n_exp < 1:
                continue
            gw = g[(g["t"] >= ws) & (g["t"] < we)]
            attention = min(1.0, float(gw["frontal"].sum()) / n_exp)
            mot = gw["motion"].dropna()
            motion_mean = float(mot.mean()) if len(mot) else np.nan
            movement = _movement_score(motion_mean) if len(mot) else 0.5
            inactivity = float((mot < 0.015).mean()) if len(mot) >= 2 else np.nan
            aud = np.nan if np.isnan(speech[wi]) else min(1.0, speech[wi] / 0.5)
            score = (w_att * attention + w_mov * movement + (w_aud * aud if w_aud else 0.0)) / total_w
            rows.append(dict(pid=int(pid), window_start=ws, attention=attention, movement=movement,
                             inactivity=inactivity, motion_mean=motion_mean, audio=aud, engagement=score))
    eng = pd.DataFrame(rows, columns=["pid", "window_start", "attention", "movement", "inactivity",
                                      "motion_mean", "audio", "engagement"])

    base = pd.DataFrame({"window_start": starts, "speech_ratio": speech})
    if eng.empty:
        cls = base.assign(engagement=np.nan, attention=np.nan, movement=np.nan,
                          inactivity=np.nan, n_participants=0)
    else:
        agg = eng.groupby("window_start").agg(engagement=("engagement", "mean"),
                                              attention=("attention", "mean"),
                                              movement=("movement", "mean"),
                                              inactivity=("inactivity", "mean"),
                                              n_participants=("pid", "nunique")).reset_index()
        cls = base.merge(agg, on="window_start", how="left")
    cls["minute"] = cls["window_start"] / 60
    return eng, cls


# --------------------------------------------------------------------------- insights
def session_insights(cls: pd.DataFrame, win_s: float, low_thr: float) -> dict | None:
    valid = cls.dropna(subset=["engagement"]).reset_index(drop=True)
    if valid.empty:
        return None
    eng = valid["engagement"].to_numpy()
    low = valid["engagement"] < low_thr

    runs, start = [], None
    for i, flag in enumerate(low):
        if flag and start is None:
            start = i
        if (not flag or i == len(low) - 1) and start is not None:
            end = i if flag else i - 1
            runs.append((float(valid.loc[start, "window_start"]), float(valid.loc[end, "window_start"] + win_s)))
            start = None

    slope = float(np.polyfit(valid["minute"], eng, 1)[0]) if len(valid) >= 3 else 0.0
    half = len(eng) // 2
    first_half = float(eng[:half].mean()) if half else float(eng.mean())
    second_half = float(eng[half:].mean())
    drops = [(float(valid.loc[i, "window_start"]), float(eng[i - 1] - eng[i]))
             for i in range(1, len(eng)) if eng[i - 1] - eng[i] >= 0.15]
    peak_i = int(np.argmax(eng))
    speech = cls["speech_ratio"].dropna()
    return dict(
        avg_eng=float(eng.mean()), avg_att=float(valid["attention"].mean()),
        avg_mov=float(valid["movement"].mean()),
        avg_inact=float(valid["inactivity"].mean()) if valid["inactivity"].notna().any() else None,
        avg_speech=float(speech.mean()) if len(speech) else None,
        slope_per_min=slope, first_half=first_half, second_half=second_half,
        low_runs=runs, low_windows=int(low.sum()), n_windows=len(valid), drops=drops,
        peak=(float(valid.loc[peak_i, "window_start"]), float(eng[peak_i])),
        lowest=(float(valid.loc[int(np.argmin(eng)), "window_start"]), float(eng.min())),
        win_s=float(win_s), n_participants=int(valid["n_participants"].max()),
    )
