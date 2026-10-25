"""Multimodal Attendance & Engagement Agent - Streamlit app (offline, privacy-preserving)."""
from __future__ import annotations

import datetime as dt
import hashlib
import os
import tempfile

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import analytics
import audio
import demo
import evaluate
import recommender
import report
import vision

st.set_page_config(page_title="Multimodal Attendance & Engagement Agent", page_icon="🎓", layout="wide")

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🎓 Attendance & Engagement Agent")
    st.caption("Computer vision + audio signals + agent reasoning. Runs offline, stores no frames.")
    mode = st.radio("Data source", ["Demo session", "Upload recording"])
    class_name = st.text_input("Class / session name", "CS101 - Lecture 5")
    start_time = st.time_input("Session start time", dt.time(10, 0))
    with st.expander("Attendance settings", expanded=True):
        min_sustain = st.slider("Minimum sustained presence (seconds)", 10, 180, 30, step=5,
                                help="A face must be seen continuously for this long to count as present.")
    with st.expander("Engagement settings"):
        win_s = st.select_slider("Analysis window (seconds)", [10, 15, 30, 60, 120], value=60)
        w_att = st.slider("Weight: head orientation", 0.0, 1.0, 0.5, 0.05)
        w_mov = st.slider("Weight: movement", 0.0, 1.0, 0.2, 0.05)
        w_aud = st.slider("Weight: audio activity", 0.0, 1.0, 0.3, 0.05)
        low_thr = st.slider("Low-engagement threshold", 0.2, 0.8, 0.55, 0.05)
    with st.expander("Video processing (uploads)"):
        sample_fps = st.select_slider("Frames analysed per second", [0.5, 1.0, 2.0, 3.0], value=1.0)
        max_minutes = st.slider("Max minutes to analyse", 1, 30, 10)
    with st.expander("Optional: Gemini wording"):
        try:
            secret_key = st.secrets.get("GEMINI_API_KEY", "")
        except Exception:  # noqa: BLE001 - no secrets file on a fresh deployment
            secret_key = ""
        api_key = st.text_input("Gemini API key", value=secret_key, type="password",
                                help="Only aggregate numbers are sent - never images, audio or names.")
        gemini_model = st.text_input("Model", "gemini-2.0-flash")

session_start = dt.datetime.combine(dt.date.today(), start_time)

# ----------------------------------------------------------------------------- data source
st.title("Multimodal Attendance & Engagement Agent")

if mode == "Demo session":
    if "demo_result" not in st.session_state:
        obs_d, meta_d, audio_d = demo.generate_demo()
        st.session_state["demo_result"] = dict(obs=obs_d, meta=meta_d, audio_df=audio_d)
    result = st.session_state["demo_result"]
    st.info("Showing a **synthetic 20-minute demo session** (attention dip around minute 11, recovery after a "
            "class activity). Switch to *Upload recording* in the sidebar to analyse your own video.")
else:
    st.subheader("Upload a classroom recording")
    c1, c2 = st.columns(2)
    video_file = c1.file_uploader("Classroom video", type=["mp4", "avi", "mov", "mkv", "webm"])
    audio_file = c2.file_uploader("Session audio (WAV, optional)", type=["wav"])
    mic = c2.audio_input("...or record audio now") if hasattr(st, "audio_input") else None
    if st.button("Analyze session", type="primary", disabled=video_file is None):
        try:
            with st.status("Analysing session...", expanded=True) as status:
                ext = os.path.splitext(video_file.name)[1] or ".mp4"
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
                tmp.write(video_file.getvalue())
                tmp.close()
                bar = st.progress(0.0, text="Detecting and tracking faces...")
                try:
                    obs_u, meta_u = vision.process_video(
                        tmp.name, sample_fps, max_minutes * 60,
                        progress_cb=lambda f: bar.progress(min(1.0, f), text="Detecting and tracking faces..."))
                finally:
                    os.unlink(tmp.name)  # nothing is kept on disk
                audio_u = None
                src = audio_file or mic
                if src is not None:
                    st.write("Analysing audio activity...")
                    samples, sr = audio.load_wav(src.getvalue())
                    audio_u = audio.audio_activity(samples, sr)
                    audio_u = audio_u[audio_u["t"] < meta_u["duration"]]
                st.session_state["upload_result"] = dict(obs=obs_u, meta=meta_u, audio_df=audio_u)
                st.session_state["names"] = {}
                status.update(label="Analysis complete", state="complete")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not analyse the recording: {exc}")
    result = st.session_state.get("upload_result")
    if result is None:
        st.caption("Tip: use MP4 (H.264) with faces clearly visible. Long videos are trimmed to the "
                   "'Max minutes' setting in the sidebar.")

# ----------------------------------------------------------------------------- analytics
tab_att, tab_eng, tab_aud, tab_sug, tab_rep, tab_snap, tab_about = st.tabs(
    ["📋 Attendance", "📈 Engagement", "🎙️ Audio", "💡 Teaching suggestions", "📄 Reports",
     "📸 Live snapshot", "ℹ️ About & privacy"])

if result is not None and result["obs"].empty:
    st.warning("No faces were detected in this recording. Check lighting / camera angle and try again.")
    result = None

if result is not None:
    obs, meta, audio_df = result["obs"], result["meta"], result["audio_df"]
    names = st.session_state.setdefault("names", {})
    label = lambda p: names.get(p, f"Participant {p}")  # noqa: E731

    att = analytics.attendance_table(obs, meta, min_sustain, session_start)
    present_ids = att.loc[att["status"] == "Present", "pid"].tolist()
    eng, cls = analytics.engagement_tables(obs, meta, audio_df, win_s, (w_att, w_mov, w_aud), present_ids)
    ins = analytics.session_insights(cls, win_s, low_thr)
    suggestions = recommender.suggest(ins, att, low_thr)

    # ---- KPIs
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Present", len(present_ids))
    k2.metric("Ignored (transient)", int((att["status"] != "Present").sum()))
    k3.metric("Avg engagement", f"{ins['avg_eng']:.0%}" if ins else "n/a")
    k4.metric("Low-engagement windows", f"{ins['low_windows']}/{ins['n_windows']}" if ins else "n/a")
    k5.metric("Audio activity", f"{ins['avg_speech']:.0%}" if ins and ins["avg_speech"] is not None else "no audio")

    # ---- Attendance
    with tab_att:
        st.markdown(f"**{class_name}** - attendance is marked on *sustained presence* "
                    f"(>= {min_sustain}s continuous), not single-frame detections.")
        view = att.copy()
        view["name"] = view["pid"].map(label)
        cols = ["pid", "name", "status", "arrived", "left", "present_min", "longest_min"]
        edited = st.data_editor(
            view[cols], hide_index=True, key="roster",
            disabled=[c for c in cols if c != "name"],
            column_config={"pid": "ID", "name": "Name (editable)", "status": "Status", "arrived": "Arrived",
                           "left": "Left", "present_min": "Present (min)", "longest_min": "Longest stretch (min)"})
        for p, n in zip(view["pid"], edited["name"]):
            names[int(p)] = str(n)
        st.download_button("Download attendance log (CSV)",
                           edited.to_csv(index=False).encode(), f"attendance_{class_name.replace(' ', '_')}.csv",
                           "text/csv")

        z, pids = analytics.presence_matrix(obs, meta, present_ids)
        if pids:
            fig = go.Figure(go.Heatmap(
                z=z, x=np.asarray(meta["sample_times"]) / 60, y=[label(p) for p in pids], zmin=0, zmax=2,
                colorscale=[[0, "#e5e7eb"], [0.33, "#e5e7eb"], [0.34, "#f59e0b"], [0.66, "#f59e0b"],
                            [0.67, "#16a34a"], [1, "#16a34a"]], showscale=False,
                hovertemplate="%{y}<br>minute %{x:.1f}<extra></extra>"))
            fig.update_layout(title="Presence timeline (green = facing front, amber = looking away, grey = not seen)",
                              xaxis_title="Minutes into session", height=max(280, 28 * len(pids) + 120),
                              margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig)
        if meta.get("previews"):
            with st.expander("Detection previews (anonymous IDs, not stored)"):
                cols_p = st.columns(len(meta["previews"]))
                for c, (t, img) in zip(cols_p, meta["previews"]):
                    c.image(img, caption=f"t = {analytics.fmt_time(t)}")

    # ---- Engagement
    with tab_eng:
        if ins is None:
            st.warning("Not enough sustained presence to compute engagement.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=cls["minute"], y=cls["engagement"], mode="lines+markers",
                                     name="Engagement", line=dict(width=3, color="#2563eb")))
            fig.add_trace(go.Scatter(x=cls["minute"], y=cls["attention"], mode="lines", name="Head orientation",
                                     line=dict(dash="dot", color="#16a34a")))
            fig.add_trace(go.Scatter(x=cls["minute"], y=cls["inactivity"], mode="lines", name="Inactivity",
                                     line=dict(dash="dashdot", color="#f59e0b")))
            if audio_df is not None:
                fig.add_trace(go.Scatter(x=cls["minute"], y=cls["speech_ratio"], mode="lines",
                                         name="Audio activity", line=dict(dash="dash", color="#9333ea")))
            fig.add_hline(y=low_thr, line_dash="dot", line_color="#dc2626",
                          annotation_text="low threshold", annotation_position="bottom right")
            for s, e in ins["low_runs"]:
                fig.add_vrect(x0=s / 60, x1=e / 60, fillcolor="#dc2626", opacity=0.08, line_width=0)
            fig.update_layout(title="Class engagement over time", yaxis_range=[0, 1], height=380,
                              xaxis_title="Minutes into session", margin=dict(l=10, r=10, t=50, b=10),
                              legend=dict(orientation="h", y=-0.25))
            st.plotly_chart(fig)

            piv = eng.pivot(index="pid", columns="window_start", values="engagement")
            hm = go.Figure(go.Heatmap(z=piv.to_numpy(), x=piv.columns / 60, y=[label(p) for p in piv.index],
                                      zmin=0, zmax=1, colorscale="RdYlGn", colorbar=dict(title="Engagement"),
                                      hovertemplate="%{y}<br>minute %{x:.0f}: %{z:.0%}<extra></extra>"))
            hm.update_layout(title="Engagement per participant and window", xaxis_title="Minutes into session",
                             height=max(280, 28 * len(piv) + 120), margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(hm)

            a, b, c, d = st.columns(4)
            d.metric("Avg inactivity", f"{ins['avg_inact']:.0%}" if ins["avg_inact"] is not None else "n/a",
                     help="Share of the time head position was almost completely still.")
            a.metric("Peak", f"{analytics.fmt_time(ins['peak'][0])}", f"{ins['peak'][1]:.0%}")
            b.metric("Lowest point", f"{analytics.fmt_time(ins['lowest'][0])}", f"{ins['lowest'][1]:.0%}",
                     delta_color="off")
            c.metric("First -> second half", f"{ins['first_half']:.0%} -> {ins['second_half']:.0%}")
            st.caption("Engagement = weighted fusion of head orientation (is the person facing the front?), "
                       "movement (natural vs frozen/restless) and class-level audio activity. "
                       "It is a decision-support signal, not a judgement of any individual.")

    # ---- Audio
    with tab_aud:
        if audio_df is None:
            st.info("No audio supplied. Upload a WAV file (or record from the mic) in *Upload recording* mode "
                    "to add audio activity to the engagement score.")
        else:
            sp = audio_df[audio_df["speech"]]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=audio_df["t"] / 60, y=audio_df["db"], mode="lines", name="Loudness",
                                     line=dict(color="#9ca3af", width=1)))
            fig.add_trace(go.Scatter(x=sp["t"] / 60, y=sp["db"], mode="markers", name="Speech activity",
                                     marker=dict(size=3, color="#9333ea")))
            fig.update_layout(title="Audio loudness (dB) and detected speech", xaxis_title="Minutes",
                              yaxis_title="dB", height=340, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig)
            m1, m2 = st.columns(2)
            m1.metric("Speech activity", f"{audio_df['speech'].mean():.0%}")
            m2.metric("Audio length", analytics.fmt_time(audio_df["t"].max()))
            st.caption("Only loudness-based speech/silence activity is used. Nothing is transcribed or stored.")

    # ---- Suggestions
    with tab_sug:
        st.markdown("The agent reads the engagement patterns and proposes **assistive** suggestions. "
                    "You stay in control of every decision.")
        icons = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}
        for s in suggestions:
            with st.container(border=True):
                st.markdown(f"{icons[s['priority']]} **{s['title']}** - *{s['priority']} priority*")
                st.write(s["why"])
                st.markdown(f"**Try:** {s['action']}")
        if ins is not None:
            if api_key:
                if st.button("Rewrite with Gemini"):
                    with st.spinner("Asking Gemini..."):
                        st.session_state["gemini_text"] = recommender.polish_with_gemini(
                            class_name, ins, suggestions, api_key, gemini_model)
                if st.session_state.get("gemini_text"):
                    st.markdown("#### Gemini coaching summary")
                    st.write(st.session_state["gemini_text"])
            else:
                st.caption("Optional: add a Gemini API key in the sidebar to get a friendlier written summary.")
        txt_report = "\n".join(f"[{s['priority']}] {s['title']}\n  Why: {s['why']}\n  Try: {s['action']}\n" for s in suggestions)
        st.download_button("Download suggestions (TXT)", f"{class_name}\n\n{txt_report}".encode(), "teaching_suggestions.txt")

    # ---- Reports
    with tab_rep:
        st.markdown("Download a printable report (open in a browser, then *Print -> Save as PDF*), or save the "
                    "session files to a local folder when you run the app on your own machine.")
        rep_view = view[["pid", "name", "status", "arrived", "left", "present_min", "longest_min"]].copy()
        rep_view["name"] = rep_view["pid"].map(label)
        html_report = report.build_html(class_name, session_start, rep_view, ins, suggestions, low_thr, min_sustain)
        slug = class_name.replace(" ", "_")
        r1, r2 = st.columns(2)
        r1.download_button("Download session report (HTML)", html_report.encode(), f"report_{slug}.html", "text/html")
        r1.download_button("Download engagement data (CSV)", eng.to_csv(index=False).encode(), f"engagement_{slug}.csv",
                           "text/csv", disabled=eng.empty)
        if r2.button("Save to local folder (data/sessions)"):
            folder = os.path.join("data", "sessions", f"{slug}_{dt.datetime.now():%Y%m%d_%H%M%S}")
            os.makedirs(folder, exist_ok=True)
            rep_view.to_csv(os.path.join(folder, "attendance.csv"), index=False)
            eng.to_csv(os.path.join(folder, "engagement.csv"), index=False)
            with open(os.path.join(folder, "report.html"), "w", encoding="utf-8") as fh:
                fh.write(html_report)
            r2.success(f"Saved to `{folder}`. (On Streamlit Cloud this storage is temporary - use the downloads.)")
        with st.expander("Preview report"):
            if hasattr(st, "iframe"):  # newer Streamlit; components.v1.html is being removed
                st.iframe(html_report, height=520)
            else:
                st.components.v1.html(html_report, height=520, scrolling=True)
else:
    with tab_rep:
        st.info("Run an analysis (or use the demo session) to generate reports.")

# ----------------------------------------------------------------------------- snapshot
with tab_snap:
    st.markdown("Take a quick photo of the room to count faces and see who is facing the front. "
                "The image is processed in memory and discarded.")
    shot = st.camera_input("Classroom snapshot")
    if shot is not None:
        frame = cv2.imdecode(np.frombuffer(shot.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        img, stats = vision.analyze_image(frame)
        st.image(img, caption="Green = facing front, orange = looking away")
        s1, s2, s3 = st.columns(3)
        s1.metric("Faces detected", stats["faces"])
        s2.metric("Facing front", stats["frontal"])
        s3.metric("Looking away", stats["away"])
        log = st.session_state.setdefault("snap_log", [])
        digest = hashlib.md5(shot.getvalue()).hexdigest()
        if not log or log[-1]["digest"] != digest:
            log.append(dict(digest=digest, time=dt.datetime.now().strftime("%H:%M:%S"), faces=stats["faces"],
                            facing_front=stats["frontal"], looking_away=stats["away"]))
    log = st.session_state.get("snap_log", [])
    if log:
        st.markdown("#### Snapshot log (timestamped, this browser session only)")
        slog = pd.DataFrame(log).drop(columns="digest")
        st.dataframe(slog, hide_index=True)
        if len(slog) > 1:
            figs = go.Figure()
            figs.add_trace(go.Scatter(x=slog["time"], y=slog["faces"], mode="lines+markers", name="Faces"))
            figs.add_trace(go.Scatter(x=slog["time"], y=slog["facing_front"], mode="lines+markers", name="Facing front"))
            figs.update_layout(height=280, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(figs)
        c_a, c_b = st.columns(2)
        c_a.download_button("Download snapshot log (CSV)", slog.to_csv(index=False).encode(), "snapshot_log.csv", "text/csv")
        if c_b.button("Clear log"):
            st.session_state["snap_log"] = []
            st.rerun()

# ----------------------------------------------------------------------------- about
with tab_about:
    st.markdown("""
### How it works
1. **Vision** - OpenCV detects faces and tracks them as anonymous IDs; frontal vs profile detection gives a head-orientation cue.
2. **Audio** - loudness-based speech activity per window (no transcription).
3. **Fusion** - head orientation + movement + audio activity are combined into an engagement score per window.
4. **Agent reasoning** - rules turn trends (dips, fade-outs, lecture-heavy pacing) into teaching suggestions.

### Privacy by design
- No face recognition: people are anonymous IDs for one session. Names are typed in manually and kept only in the browser session.
- Frames are never saved; uploaded video is deleted from disk right after processing.
- The core analysis needs no internet. Gemini wording is optional and receives only aggregate numbers.
- Attendance and engagement are **decision-support signals**; the instructor makes the final call.

### Limitations
Haar-cascade detection works best with front-facing cameras and good lighting; small or far-away faces may be missed,
and a person who leaves and re-enters may get a new ID.
""")

with tab_about:
    st.markdown("### System architecture")
    st.graphviz_chart(r"""
    digraph G {
      rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef2ff", fontname="Helvetica"];
      Video [label="Video input\n(upload / snapshot)"]; Audio [label="Audio input\n(WAV / microphone)"];
      Det [label="Face detection\n& anonymous tracking"]; Pres [label="Sustained presence\n+ timestamp log"];
      Sig [label="Head orientation\nmovement, inactivity"]; Sp [label="Speech &\ninteraction level"];
      Fus [label="Multimodal fusion", fillcolor="#dbeafe"]; Score [label="Engagement score\n& session trends"];
      Agent [label="AI agent\n(teaching suggestions)", fillcolor="#dcfce7"]; Dash [label="Dashboard\nreports & exports", fillcolor="#fef9c3"];
      Video -> Det -> Pres -> Dash; Det -> Sig -> Fus; Audio -> Sp -> Fus; Fus -> Score -> Agent -> Dash; Score -> Dash;
    }""")
    st.markdown("### Accuracy & performance check")
    st.caption("Runs the full pipeline on the synthetic session, where the true attendance (14 learners and one planted "
               "false positive) and the true engagement dip are known.")
    if st.button("Run self-check"):
        r = evaluate.self_check(min_sustain)
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Attendance precision", f"{r['precision']:.0%}")
        e2.metric("Attendance recall", f"{r['recall']:.0%}")
        e3.metric("Engagement dip found", "yes" if r["dip_found"] else "no")
        e4.metric("Pipeline time", f"{r['seconds']:.2f}s", f"{r['samples']} samples", delta_color="off")
    up = st.session_state.get("upload_result")
    if up and up["meta"].get("proc_seconds"):
        m = up["meta"]
        n = len(m["sample_times"])
        st.markdown(f"**Last uploaded video:** {n} frames analysed in {m['proc_seconds']:.1f}s "
                    f"({n / max(m['proc_seconds'], 1e-6):.1f} frames/s).")
