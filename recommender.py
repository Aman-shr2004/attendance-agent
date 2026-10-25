"""Agent reasoning layer - turns aggregate analytics into teaching suggestions.

The default engine is rule-based and fully offline. An optional Gemini call can
rewrite the suggestions in a friendlier tone; it only ever receives aggregate
numbers (never images, audio or names).
"""
from __future__ import annotations

import pandas as pd
import requests

from analytics import fmt_time

PRIORITY_ORDER = {"High": 0, "Medium": 1, "Low": 2}


def suggest(ins: dict | None, att: pd.DataFrame, low_thr: float) -> list[dict]:
    out: list[dict] = []

    def add(priority, title, why, action):
        out.append(dict(priority=priority, title=title, why=why, action=action))

    transient = int((att["status"] != "Present").sum()) if len(att) else 0
    if transient >= 3:
        add("Low", "Improve camera setup",
            f"{transient} short-lived detections were ignored as false positives.",
            "Place the camera at the front at eye level, improve lighting and avoid strong backlight.")

    if ins is None:
        add("High", "Not enough data to analyse engagement",
            "No participant was present long enough to compute engagement.",
            "Use a longer recording, lower the minimum-presence setting or check that faces are visible.")
        return out

    win = ins["win_s"]
    for s, e in ins["low_runs"]:
        long_run = (e - s) >= 2 * win
        add("High" if long_run else "Medium",
            f"Attention dip between {fmt_time(s)} and {fmt_time(e)}",
            f"Class engagement stayed below {low_thr:.0%} for {fmt_time(e - s)}.",
            "Insert an interactive moment here: a quick poll, think-pair-share or a worked example. "
            "If the dip lasts several minutes, a 2-minute stretch break is worth trying.")

    if ins["avg_eng"] < low_thr:
        add("High", "Overall engagement is low",
            f"Average engagement was {ins['avg_eng']:.0%}.",
            "Chunk the content into 10-12 minute segments and open each with a question or problem.")

    if ins["slope_per_min"] < -0.01 and ins["first_half"] - ins["second_half"] > 0.1:
        add("Medium", "Engagement fades as the session goes on",
            f"First half averaged {ins['first_half']:.0%}, second half {ins['second_half']:.0%}.",
            "Move the most demanding material earlier and plan an activity or recap for the second half.")
    elif ins["second_half"] - ins["first_half"] > 0.1:
        add("Low", "Strong recovery later in the session",
            f"Engagement rose from {ins['first_half']:.0%} to {ins['second_half']:.0%}.",
            "Whatever you did in the later part worked; consider using it earlier.")

    for s, d in ins["drops"][:2]:
        add("Medium", f"Sudden drop at {fmt_time(s)}",
            f"Engagement fell by {d:.0%} compared with the previous window.",
            "Review what happened at this moment (topic change, long explanation, technical issue) and smooth the transition.")

    sp = ins["avg_speech"]
    if sp is not None:
        if sp > 0.75 and ins["avg_att"] < 0.65:
            add("Medium", "Lecture-heavy pacing",
                f"Audio was active {sp:.0%} of the time while attention was only {ins['avg_att']:.0%}.",
                "Add deliberate pauses and questions; aim for at least one learner-led exchange every 10 minutes.")
        elif sp < 0.25:
            add("Low", "Very little spoken interaction",
                f"Speech was detected only {sp:.0%} of the time.",
                "If this was not a silent-work block, invite questions or discussion to raise participation.")

    if ins["avg_att"] < 0.5 and ins["avg_mov"] < 0.75:
        add("Medium", "Restless or distracted room",
            f"Only {ins['avg_att']:.0%} of the time were learners facing the front, with irregular movement.",
            "Re-state the objective, give a clear task with a time limit and walk through the room.")

    if ins["avg_inact"] is not None and ins["avg_inact"] > 0.6 and ins["avg_att"] < 0.6:
        add("Low", "Long periods of stillness",
            "Learners were frozen for long stretches without looking at the front.",
            "Check for fatigue (room temperature, time of day) and vary the activity type.")

    if not any(s["priority"] in ("High", "Medium") for s in out):
        add("Low", "Session looks healthy",
            f"Average engagement {ins['avg_eng']:.0%}, peak at {fmt_time(ins['peak'][0])}.",
            "Keep the current pacing. Note what you did around the peak and repeat it.")

    out.sort(key=lambda s: PRIORITY_ORDER[s["priority"]])
    return out


def polish_with_gemini(class_name: str, ins: dict, suggestions: list[dict], api_key: str,
                       model: str = "gemini-2.0-flash") -> str:
    """Optional: rewrite the rule-based suggestions with Gemini (aggregate data only)."""
    facts = (f"Session: {class_name}. Avg engagement {ins['avg_eng']:.0%}, attention {ins['avg_att']:.0%}, "
             f"low windows {ins['low_windows']}/{ins['n_windows']}, participants {ins['n_participants']}.")
    bullets = "\n".join(f"- {s['title']}: {s['why']} Action: {s['action']}" for s in suggestions)
    prompt = ("You are a supportive teaching coach. Using ONLY the facts below, write at most 5 short, "
              "practical, encouraging suggestions for the instructor. Do not mention individual students.\n\n"
              f"{facts}\n\nDraft findings:\n{bullets}")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    try:
        r = requests.post(url, params={"key": api_key}, timeout=30,
                          json={"contents": [{"parts": [{"text": prompt}]}]})
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as exc:  # noqa: BLE001 - surface a friendly message in the UI
        return f"Gemini request failed ({exc.__class__.__name__}). The rule-based suggestions above are still valid."
