"""Self-check against synthetic ground truth (accuracy + speed) shown in the app and used by tests."""
from __future__ import annotations

import datetime as dt
import time

import analytics
import demo


def self_check(min_sustain: float = 30) -> dict:
    t0 = time.perf_counter()
    obs, meta, aud = demo.generate_demo()
    att = analytics.attendance_table(obs, meta, min_sustain, dt.datetime(2026, 1, 1, 10, 0))
    pred = set(att.loc[att["status"] == "Present", "pid"])
    truth = set(range(1, 15))          # 14 real learners; pid 99 is a planted false positive
    tp = len(pred & truth)
    eng, cls = analytics.engagement_tables(obs, meta, aud, 60, (0.5, 0.2, 0.3), sorted(pred))
    ins = analytics.session_insights(cls, 60, 0.55)
    dip = bool(ins) and any(8 * 60 <= s <= 13 * 60 for s, _ in ins["low_runs"])
    return dict(precision=tp / len(pred) if pred else 0.0, recall=tp / len(truth),
                false_positives=len(pred - truth), dip_found=dip,
                samples=len(meta["sample_times"]), seconds=time.perf_counter() - t0)
