"""Printable, self-contained HTML session report (open in a browser -> Print / Save as PDF)."""
from __future__ import annotations

import datetime as dt
import html

import pandas as pd

from analytics import fmt_time

CSS = ("<style>body{font-family:system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#111}"
       "table{border-collapse:collapse;width:100%;margin:.5rem 0 1.5rem}th,td{border:1px solid #ddd;padding:6px 8px;"
       "text-align:left}th{background:#f3f4f6}.High{color:#b91c1c}.Medium{color:#b45309}.Low{color:#15803d}"
       ".kpi span{display:inline-block;margin:0 1.5rem .5rem 0}</style>")


def build_html(class_name: str, session_start: dt.datetime, att_view: pd.DataFrame, ins: dict | None,
               suggestions: list[dict], low_thr: float, min_sustain: float) -> str:
    e = html.escape
    present = int((att_view["status"] == "Present").sum())
    out = [f"<html><head><meta charset='utf-8'><title>{e(class_name)} - session report</title>{CSS}</head><body>",
           f"<h1>{e(class_name)}</h1><p>Session start: {session_start.strftime('%d %b %Y, %H:%M')} &middot; "
           f"Attendance rule: continuous presence &ge; {min_sustain:.0f}s</p>",
           "<div class='kpi'>", f"<span><b>Present:</b> {present}</span>",
           f"<span><b>Ignored (transient):</b> {len(att_view) - present}</span>"]
    if ins:
        out += [f"<span><b>Avg engagement:</b> {ins['avg_eng']:.0%}</span>",
                f"<span><b>Head orientation:</b> {ins['avg_att']:.0%}</span>",
                f"<span><b>Peak:</b> {fmt_time(ins['peak'][0])} ({ins['peak'][1]:.0%})</span>",
                f"<span><b>Lowest:</b> {fmt_time(ins['lowest'][0])} ({ins['lowest'][1]:.0%})</span>"]
        if ins["avg_speech"] is not None:
            out.append(f"<span><b>Audio activity:</b> {ins['avg_speech']:.0%}</span>")
        if ins["avg_inact"] is not None:
            out.append(f"<span><b>Inactivity:</b> {ins['avg_inact']:.0%}</span>")
    out.append("</div><h2>Attendance log</h2>")
    out.append(att_view.to_html(index=False, border=0))
    if ins and ins["low_runs"]:
        runs = "".join(f"<li>{fmt_time(s)} - {fmt_time(e_)} (below {low_thr:.0%})</li>" for s, e_ in ins["low_runs"])
        out.append(f"<h2>Low-engagement periods</h2><ul>{runs}</ul>")
    out.append("<h2>Teaching suggestions</h2>")
    for s in suggestions:
        out.append(f"<p><b class='{s['priority']}'>[{s['priority']}] {e(s['title'])}</b><br>{e(s['why'])}<br>"
                   f"<i>Try:</i> {e(s['action'])}</p>")
    out.append("<hr><small>Generated offline by the Multimodal Attendance &amp; Engagement Agent. "
               "Decision-support only; no images or audio were stored.</small></body></html>")
    return "".join(out)
