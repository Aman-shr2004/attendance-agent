"""Synthetic classroom session so the live app is explorable without any upload.

Produces data in exactly the same schema as the real vision/audio modules, so the
whole analytics pipeline is exercised end-to-end. Story: strong start, attention
dip around minute 11, recovery after a class activity at minute ~15.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _g(t, m, s):
    return np.exp(-0.5 * ((t - m) / s) ** 2)


def generate_demo(seed: int = 7, minutes: int = 20, n_students: int = 14, step: float = 2.0):
    rng = np.random.default_rng(seed)
    times = np.arange(0, minutes * 60, step)
    tm = times / 60
    p_att = np.clip(0.9 - 0.010 * tm - 0.55 * _g(tm, 11, 2.2) + 0.30 * _g(tm, 15.5, 1.7), 0.1, 0.97)

    arrive = np.zeros(n_students)
    arrive[3], arrive[9] = 3 * 60, 5.5 * 60          # two late arrivals
    leave = np.full(n_students, np.inf)
    leave[12] = 17 * 60                              # one early leaver

    rows = []
    for i in range(n_students):
        pid, trait = i + 1, rng.normal(0, 0.07)
        cx0, cy0 = 80 + (i % 5) * 120, 90 + (i // 5) * 110
        has_prev = False
        for k, (t, p) in enumerate(zip(times, p_att)):
            if t < arrive[i] or t > leave[i]:
                has_prev = False
                continue
            attentive = rng.random() < np.clip(p + trait, 0.05, 0.98)
            if attentive:
                detected, frontal, motion = rng.random() < 0.97, 1, abs(rng.normal(0.04, 0.02))
            else:
                detected, frontal, motion = rng.random() < 0.5, 0, abs(rng.normal(0.14, 0.05))
            if not detected:
                has_prev = False
                continue
            rows.append((k, t, pid, cx0 + rng.normal(0, 2), cy0 + rng.normal(0, 2), 60 + rng.normal(0, 2),
                         frontal, motion if has_prev else np.nan))
            has_prev = True

    # one transient false positive (e.g. someone walking past the camera)
    for k in range(200, 203):
        rows.append((k, times[k], 99, 560, 300, 55, 1, 0.3 if k > 200 else np.nan))

    obs = pd.DataFrame(rows, columns=["k", "t", "pid", "cx", "cy", "w", "frontal", "motion"])

    n_a = minutes * 120  # 0.5 s audio frames
    ta = np.arange(n_a) * 0.5
    q = 0.72 - 0.50 * _g(ta / 60, 11, 2.2) + 0.15 * _g(ta / 60, 15.5, 1.7)
    speech = rng.random(n_a) < q
    db = np.where(speech, rng.normal(-24, 4, n_a), rng.normal(-50, 3, n_a))
    audio_df = pd.DataFrame({"t": ta, "db": db, "speech": speech})

    meta = dict(sample_times=times, duration=float(minutes * 60), previews=[], source="demo")
    return obs, meta, audio_df
