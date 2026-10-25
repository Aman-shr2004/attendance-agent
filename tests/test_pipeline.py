"""Run with:  pytest -q"""
import datetime as dt
import io
import os
import sys
import tempfile
import wave

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import analytics, audio, demo, recommender, vision  # noqa: E402

START = dt.datetime(2026, 1, 1, 10, 0)


def _demo_pipeline(win_s=60, thr=0.55):
    obs, meta, aud = demo.generate_demo()
    att = analytics.attendance_table(obs, meta, 30, START)
    ids = att.loc[att.status == "Present", "pid"]
    eng, cls = analytics.engagement_tables(obs, meta, aud, win_s, (0.5, 0.2, 0.3), ids)
    return att, eng, cls, analytics.session_insights(cls, win_s, thr)


def test_attendance_ignores_transient_detection():
    att, *_ = _demo_pipeline()
    assert (att.status == "Present").sum() == 14
    assert att.loc[att.pid == 99, "status"].iloc[0].startswith("Transient")


def test_engagement_dip_is_found_around_minute_11():
    _, _, cls, ins = _demo_pipeline()
    assert ins["low_runs"], "expected a low-engagement run"
    s, e = ins["low_runs"][0]
    assert 8 * 60 <= s <= 13 * 60
    assert 0.0 <= cls.engagement.min() < cls.engagement.max() <= 1.0


def test_suggestions_generated():
    att, _, _, ins = _demo_pipeline()
    sug = recommender.suggest(ins, att, 0.55)
    assert sug and any("dip" in s["title"].lower() for s in sug)


def test_audio_roundtrip():
    sr = 8000
    t = np.arange(sr * 10) / sr
    sig = np.where((t % 4) < 2, 0.4 * np.sin(2 * np.pi * 220 * t), 0.001 * np.random.randn(len(t)))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes((sig * 32767).astype(np.int16).tobytes())
    x, sr2 = audio.load_wav(buf.getvalue())
    df = audio.audio_activity(x, sr2)
    assert 0.3 < df.speech.mean() < 0.7


def test_video_pipeline_detects_a_face():
    from skimage import data
    face = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    path = os.path.join(tempfile.mkdtemp(), "t.avi")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (512, 512))
    for i in range(60):  # 6 s
        M = np.float32([[1, 0, np.sin(i / 5) * 2], [0, 1, 0]])
        vw.write(cv2.warpAffine(face, M, (512, 512)))
    vw.release()
    obs, meta = vision.process_video(path, sample_fps=2)
    assert not obs.empty and obs.pid.nunique() <= 2
    att = analytics.attendance_table(obs, meta, 3, START)
    assert (att.status == "Present").any()


def test_self_check_accuracy():
    import evaluate
    r = evaluate.self_check()
    assert r["precision"] == 1.0 and r["recall"] == 1.0 and r["false_positives"] == 0
    assert r["dip_found"]


def test_report_html():
    import report
    att, _, _, ins = _demo_pipeline()
    sug = recommender.suggest(ins, att, 0.55)
    html = report.build_html("<b>Class</b>", START, att[["pid", "status"]], ins, sug, 0.55, 30)
    assert "&lt;b&gt;Class" in html and "Teaching suggestions" in html  # escaped, complete
