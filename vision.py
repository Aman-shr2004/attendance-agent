"""Vision module - face detection, anonymous tracking and head-orientation cues.

Pure OpenCV (Haar cascades), so it runs offline on CPU and installs cleanly on
Streamlit Community Cloud. No face *recognition* is performed: people are only
tracked as anonymous IDs for the duration of one session and no frames are
stored.

Head orientation heuristic
    frontal face detected  -> looking towards the front / board  (frontal = 1)
    only profile detected  -> head turned away                   (frontal = 0)
    nothing detected       -> head down / occluded               (handled in analytics)
"""
from __future__ import annotations

import os
import time

import cv2
import numpy as np
import pandas as pd

_CASCADES: dict = {}


def _cascade(name: str):
    if name not in _CASCADES:
        c = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, name))
        if c.empty():
            raise RuntimeError(f"Could not load OpenCV cascade: {name}")
        _CASCADES[name] = c
    return _CASCADES[name]


def _iou(a, b) -> float:
    ax, ay, aw, ah = a[:4]
    bx, by, bw, bh = b[:4]
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def detect_faces(frame_bgr: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    """Return a list of (x, y, w, h, frontal_flag) boxes."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    h, w = gray.shape
    min_size = (max(24, int(w * 0.04)),) * 2

    frontal = _cascade("haarcascade_frontalface_alt2.xml").detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=min_size
    )
    dets = [(int(x), int(y), int(bw), int(bh), 1) for x, y, bw, bh in frontal]

    profile = _cascade("haarcascade_profileface.xml")
    for flipped in (False, True):  # the cascade only knows one side
        g = cv2.flip(gray, 1) if flipped else gray
        for x, y, bw, bh in profile.detectMultiScale(
            g, scaleFactor=1.1, minNeighbors=4, minSize=min_size
        ):
            if flipped:
                x = w - x - bw
            box = (int(x), int(y), int(bw), int(bh), 0)
            if all(_iou(box, d) < 0.3 for d in dets):
                dets.append(box)
    return dets


class FaceTracker:
    """Greedy nearest-neighbour tracker producing anonymous, session-local IDs."""

    def __init__(self, max_missed: int = 6, dist_factor: float = 1.2):
        self.tracks: dict[int, dict] = {}
        self.next_id = 1
        self.max_missed = max_missed
        self.dist_factor = dist_factor

    def update(self, dets):
        """Returns list of (track_id, det, motion). motion is NaN for a new track."""
        pairs = []
        for tid, tr in self.tracks.items():
            for i, (x, y, w, h, _) in enumerate(dets):
                cx, cy = x + w / 2, y + h / 2
                d = float(np.hypot(cx - tr["cx"], cy - tr["cy"]))
                lim = self.dist_factor * max(w, tr["w"])
                if d <= lim:
                    pairs.append((d / lim, tid, i))
        pairs.sort()

        used_t, used_d, out = set(), set(), []
        for _, tid, i in pairs:
            if tid in used_t or i in used_d:
                continue
            used_t.add(tid)
            used_d.add(i)
            x, y, w, h, f = dets[i]
            cx, cy = x + w / 2, y + h / 2
            tr = self.tracks[tid]
            motion = float(np.hypot(cx - tr["cx"], cy - tr["cy"]) / max(w, 1))
            tr.update(cx=cx, cy=cy, w=w, missed=0)
            out.append((tid, dets[i], motion))

        for tid in list(self.tracks):
            if tid not in used_t:
                self.tracks[tid]["missed"] += 1
                if self.tracks[tid]["missed"] > self.max_missed:
                    del self.tracks[tid]

        for i, (x, y, w, h, f) in enumerate(dets):
            if i in used_d:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = dict(cx=x + w / 2, cy=y + h / 2, w=w, missed=0)
            out.append((tid, dets[i], float("nan")))
        return out


def annotate(frame_bgr: np.ndarray, tracked) -> np.ndarray:
    """Draw anonymous boxes; returns an RGB image. tracked = [(id, (x,y,w,h,f)), ...]"""
    out = frame_bgr.copy()
    for tid, (x, y, w, h, frontal) in tracked:
        color = (80, 200, 80) if frontal else (0, 165, 255)  # BGR: green / orange
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cv2.putText(out, f"P{tid}", (x, max(14, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def _resize(frame: np.ndarray, target_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    return cv2.resize(frame, (target_width, int(h * target_width / w)), interpolation=cv2.INTER_AREA)


def process_video(path: str, sample_fps: float = 1.0, max_seconds: float = 600,
                  target_width: int = 480, progress_cb=None):
    """Sample a video, detect + track faces and return (observations, meta).

    observations columns: k (sample index), t (seconds), pid, cx, cy, w, frontal, motion
    meta: sample_times, duration, previews (annotated RGB frames), source
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Could not open the video file. Try an MP4 (H.264).")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 1 or fps > 240:
        fps = 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps / sample_fps)))
    limit = int(max_seconds * fps)
    expected = (min(total, limit) // step) if total > 0 else 0
    preview_every = max(1, expected // 4) if expected else 20

    t_start = time.perf_counter()
    tracker = FaceTracker()
    rows, times, previews = [], [], []
    idx, k = 0, 0
    while idx < limit and cap.grab():
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            frame = _resize(frame, target_width)
            t = idx / fps
            tracked = tracker.update(detect_faces(frame))
            for tid, (x, y, w, h, f), motion in tracked:
                rows.append((k, t, tid, x + w / 2, y + h / 2, w, f, motion))
            if len(previews) < 4 and k % preview_every == 0:
                previews.append((t, annotate(frame, [(tid, d) for tid, d, _ in tracked])))
            times.append(t)
            k += 1
            if progress_cb and expected:
                progress_cb(k / expected)
        idx += 1
    cap.release()

    if not times:
        raise ValueError("No frames could be read from the video.")
    obs = pd.DataFrame(rows, columns=["k", "t", "pid", "cx", "cy", "w", "frontal", "motion"])
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 1.0 / sample_fps
    meta = dict(sample_times=np.array(times), duration=float(times[-1] + dt),
                previews=previews, source="video", proc_seconds=time.perf_counter() - t_start)
    return obs, meta


def analyze_image(frame_bgr: np.ndarray, target_width: int = 800):
    """Single-snapshot analysis for the 'Live snapshot' tab."""
    frame = _resize(frame_bgr, target_width)
    dets = detect_faces(frame)
    img = annotate(frame, [(i + 1, d) for i, d in enumerate(dets)])
    n_front = sum(d[4] for d in dets)
    return img, dict(faces=len(dets), frontal=n_front, away=len(dets) - n_front)
