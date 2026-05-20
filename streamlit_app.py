import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import cv2
import av
import queue
import threading
from app.orchestrator.graph import create_orchestrator

st.set_page_config(layout="wide", page_title="Live Orchestrator Demo")

# Custom Video Processor
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.result_queue = queue.Queue()
        self.frames = []
        self.frame_count = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        self.frames.append(img)
        self.frame_count += 1
        
        # Dispatch to queue every 30 frames (approx 1-2 seconds)
        if self.frame_count % 30 == 0:
            self.result_queue.put(self.frames.copy())
            self.frames.clear()
            
        return frame

from streamlit_webrtc import AudioProcessorBase
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.result_queue = queue.Queue()
        self.audio_frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        snd = frame.to_ndarray()
        self.audio_frames.append(snd)
        
        # 60 frames ≈ 1.2s of audio — enough for Sarvam language ID, faster than 100-frame buffer
        if len(self.audio_frames) > 60:
            import numpy as np
            audio_data = np.concatenate(self.audio_frames, axis=1)
            self.result_queue.put((audio_data, frame.sample_rate))
            self.audio_frames.clear()
            
        return frame

from app.core.persona import persona_framework, performer_roles, target_roles

# ── Persona & Prompt Configuration ────────────────────────────────────────────
with st.expander("🎥 Session Setup — Persona & Prompt", expanded=True):
    cfg_col1, cfg_col2 = st.columns(2)

    with cfg_col1:
        performer_names = [r.name for r in performer_roles]
        selected_performer = st.selectbox(
            " Performer Role (You)",
            performer_names,
            key="sel_performer",
            help="Your role in this practice session."
        )
        performer_role_obj = persona_framework.get_role(selected_performer)
        if performer_role_obj:
            st.caption(f" {performer_role_obj.requirements['description']}")

    with cfg_col2:
        target_names = [r.name for r in target_roles]
        selected_target = st.selectbox(
            "👥 Target Audience (Customer)",
            target_names,
            key="sel_target",
            help="The persona of the customer you are speaking to."
        )
        target_role_obj = persona_framework.get_role(selected_target)
        if target_role_obj:
            st.caption(f" {target_role_obj.requirements['description']}")
            expectations = target_role_obj.requirements.get("expectations", [])
            if expectations:
                st.caption("Expectations: " + " • ".join(expectations))

    # Default prompt hint based on target audience
    _default_prompt = "Provide your query"
    prompt_input = st.text_area(
        "Your Prompt / Question",
        value=_default_prompt,
        height=80,
        key="prompt_input",
        help="Type the specific question or scenario the candidate must answer. "
             "Relevance will be scored against this prompt AND the selected audience profile."
    )

if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = create_orchestrator()
    st.session_state.state = {
        "messages": [],
        "metadata": {
            "session_id": "live_streamlit",
            "prompt": prompt_input,
            "performer_role": selected_performer,
            "target_role": selected_target,
        },
        "stream_status": {},
        "transcript_state": {
            "rolling_transcript": "",
            "api_lang_counts": {},
            "api_lang_max_prob": {},
        },
        "recent_video_features": [],
        "recent_acoustic_features": [],
        "recent_text_markers": [],
        "scores": {
            "visual_performance_score": 0.0,
            "audio_performance_score": 0.0,
            "text_performance_score": 0.0,
            "relevance_score": 0.0,
            "overall_score": 0.0
        },
        "active_alerts": [],
        "cooldown_timers": {},
        "report": {}
    }
    # Background graph thread state
    st.session_state.graph_result_queue = queue.Queue(maxsize=1)
    st.session_state.graph_thread = None

# Always sync latest prompt + persona into metadata (user may change mid-session)
st.session_state.state["metadata"]["prompt"]             = prompt_input
st.session_state.state["metadata"]["performer_role"]     = selected_performer
st.session_state.state["metadata"]["target_role"]        = selected_target


col1, col2 = st.columns([1, 1])

with col1:
    webrtc_ctx = webrtc_streamer(
        key="live-webcam",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"video": True, "audio": True},
        async_processing=True,
    )

with col2:
    st.subheader("Live Workflow Output")
    
    score_placeholder = st.empty()
    lang_placeholder  = st.empty()   # live language badge bar
    alert_placeholder = st.empty()
    debug_placeholder = st.empty()

if webrtc_ctx.state.playing:
    if webrtc_ctx.video_processor:
        
        # Initialise background thread state on first render
        if "graph_result_queue" not in st.session_state:
            st.session_state.graph_result_queue = queue.Queue(maxsize=1)
            st.session_state.graph_thread = None
        if "last_known_languages" not in st.session_state:
            st.session_state.last_known_languages = set()

        def _run_full_graph(orchestrator, state, result_q):
            """Full graph (STT + text + relevance) — slow, runs in background."""
            try:
                new_state = orchestrator.invoke(state)
                try:
                    result_q.put_nowait(new_state)
                except queue.Full:
                    pass
            except Exception:
                pass

        # ── Import video agent for the fast inline path ──────────────────────
        from app.agents import video as _video_agent

        # Loop to consume queue continuously while streaming
        while True:
            try:
                # Wait for next video frame batch (fires every ~1s)
                frames_chunk = webrtc_ctx.video_processor.result_queue.get(timeout=1.0)

                audio_chunk = None
                if webrtc_ctx.audio_processor:
                    try:
                        audio_chunk = webrtc_ctx.audio_processor.result_queue.get_nowait()
                    except queue.Empty:
                        pass

                # ── Pick up full-graph result if background thread finished ──
                try:
                    finished_state = st.session_state.graph_result_queue.get_nowait()
                    # Merge: keep video scores from inline path, take rest from graph
                    cur = st.session_state.state
                    merged = dict(finished_state)
                    merged_scores = dict(finished_state.get("scores", {}))
                    # Preserve the freshest visual scores from inline processing
                    for vk in ("visual_performance_score", "posture_score",
                               "eye_contact_score", "expression_score"):
                        if vk in cur.get("scores", {}):
                            merged_scores[vk] = cur["scores"][vk]
                    merged["scores"] = merged_scores
                    st.session_state.state = merged
                except queue.Empty:
                    pass

                # ── FAST PATH: run video agent inline (MediaPipe, ~50-150ms) ──
                try:
                    vid_input = dict(st.session_state.state)
                    vid_input["recent_video_features"] = [{"raw_frames": frames_chunk}]
                    vid_result = _video_agent.process(vid_input)
                    if vid_result:
                        new_scores = dict(st.session_state.state.get("scores", {}))
                        new_scores.update(vid_result.get("scores", {}))
                        st.session_state.state["scores"] = new_scores
                        if "active_alerts" in vid_result:
                            st.session_state.state["active_alerts"] = vid_result["active_alerts"]
                except Exception:
                    pass  # video errors never block the UI

                # ── SLOW PATH: full graph only when new audio is available ────
                thread = st.session_state.graph_thread
                if audio_chunk and (thread is None or not thread.is_alive()):
                    next_state = dict(st.session_state.state)
                    next_state["metadata"] = dict(next_state.get("metadata", {}))
                    next_state["metadata"]["prompt"] = prompt_input
                    next_state["recent_video_features"] = [{"raw_frames": frames_chunk}]
                    next_state["recent_acoustic_features"] = [{"raw_audio": audio_chunk}]

                    t = threading.Thread(
                        target=_run_full_graph,
                        args=(st.session_state.orchestrator, next_state,
                              st.session_state.graph_result_queue),
                        daemon=True,
                    )
                    t.start()
                    st.session_state.graph_thread = t

                # ── Render immediately from last known state (no blocking) ──

                state = st.session_state.state

                with score_placeholder.container():
                    sc = state["scores"].get("overall_score", 0)
                    st.metric("Overall Score Tracking", f"{sc} / 10")

                # ── Language badge bar (fires toast on new detection) ─────────
                _ts   = state.get("transcript_state", {})
                _langs = _ts.get("stt_detected_languages", [])
                _new  = [l for l in _langs
                         if l not in st.session_state.last_known_languages]
                for _nl in _new:
                    st.toast(f"🌐 {_nl} detected!", icon="🌐")
                    st.session_state.last_known_languages.add(_nl)

                with lang_placeholder.container():
                    if _langs:
                        _LANG_FLAGS = {
                            "English": "🇬🇧", "Hindi": "🇮🇳",
                            "Bengali": "🇧🇩", "Tamil": "🇮🇳",
                            "Telugu": "🇮🇳", "Kannada": "🇮🇳",
                            "Malayalam": "🇮🇳", "Gujarati": "🇮🇳",
                            "Punjabi": "🇮🇳", "Marathi": "🇮🇳",
                            "Odia": "🇮🇳", "Urdu": "🇵🇰",
                        }
                        badges = " • ".join(
                            f"{_LANG_FLAGS.get(l, '🗺️')} **{l}**" for l in _langs
                        )
                        st.markdown(f"🌐 **Languages detected:** {badges}")
                    else:
                        st.caption("🌐 Listening for languages…")

                with alert_placeholder.container():
                    alerts = state.get("active_alerts", [])
                    if alerts:
                        for a in alerts:
                            level = a.get("level", "info")
                            msg = a["message"]
                            if level == "success":
                                st.success(msg)
                            elif level == "warning":
                                st.warning(msg)
                            else:
                                st.info(msg)
                    else:
                        st.info(" Performance looks solid — keep going!")

                with debug_placeholder.container():
                    with st.expander("Internal Scores Buffer"):
                        st.json(state["scores"])
                    with st.expander("Language Detection (API)"):
                        ts = state.get("transcript_state", {})
                        confirmed_langs = ts.get("stt_detected_languages", [])
                        raw_counts = ts.get("api_lang_counts", {})
                        raw_conf = ts.get("api_lang_max_prob", {})
                        
                        st.markdown("**Detected Languages:**")
                        st.json(confirmed_langs)
                        
                        st.markdown("**API Counts:**")
                        st.json(raw_counts)
                        
                        st.markdown("**API Max Confidence:**")
                        st.json(raw_conf)
                        
                        if confirmed_langs:
                            st.success(f"Languages detected: {', '.join(confirmed_langs)}")
                        else:
                            st.warning("No languages confirmed yet. Please speak clearly.")
                        
                        filtered_out = [
                            f"{lang} (conf={raw_conf.get(lang, 0):.2f}, "
                            f"pct={raw_counts.get(lang,0)/max(sum(raw_counts.values()),1)*100:.0f}%, "
                            f"count={raw_counts.get(lang,0)})"
                            for lang in raw_counts
                            if lang not in confirmed_langs
                        ]
                        if filtered_out:
                            st.caption(
                                f"⚠️ Filtered out (need count≥2, ≥5%, conf≥0.15): "
                                f"{', '.join(filtered_out)}"
                            )
                    
                    with st.expander("🔬 Sarvam Raw Response (last chunk)"):
                        ts = state.get("transcript_state", {})
                        last_chunk = ts.get("last_transcript_chunk", "")
                        last_resp  = ts.get("last_sarvam_response", {})
                        
                        st.markdown(f"**Last transcript chunk:** `{last_chunk}`")
                        st.markdown("**Raw API response fields:**")
                        
                        # Highlight the key language fields prominently
                        lang_code = last_resp.get("language_code", "⚠️ NOT FOUND")
                        lang_prob  = last_resp.get("language_probability", "⚠️ NOT FOUND")
                        st.error(f"language_code → **{lang_code}**")
                        st.error(f"language_probability → **{lang_prob}**")
                        
                        # Show all other fields
                        st.json(last_resp)
                    

            except queue.Empty:
                pass
else:
    scores = st.session_state.state.get("scores", {})
    if scores.get("overall_score", 0) > 0:
        st.markdown("---")
        st.subheader("Session Summary")
        st.success("Your session has concluded. Here is your final scorecard:")
        
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Final Overall Score", f"{scores.get('overall_score', 0)} / 10")
        sc2.metric("Visual Performance", f"{scores.get('visual_performance_score', 0)} / 10")
        sc3.metric("Audio Performance", f"{scores.get('audio_performance_score', 0)} / 10")
        
        sc4, sc5 = st.columns(2)
        sc4.metric("Text Quality & Grammar", f"{scores.get('text_performance_score', 0)} / 10")
        sc5.metric("Prompt Relevance", f"{scores.get('relevance_score', 0)} / 10")
        
        langs_used = scores.get('all_detected_languages', ['English'])
        st.info(f"** Languages Spoken Profile:** {', '.join(langs_used)}")
        
        st.markdown("### Key Takeaways")
        
        from app.core.persona import persona_framework
        
        # Fetch selected roles and category from session state metadata
        meta = st.session_state.state.get("metadata", {})
        performer_role_name = meta.get("performer_role", None)
        target_role_name = meta.get("target_role", None)
        practice_category = meta.get("practice_category", None)
        
        performer_role = persona_framework.get_role(performer_role_name) if performer_role_name else None
        target_role = persona_framework.get_role(target_role_name) if target_role_name else None
        
        # Contextual feedback messages
        if performer_role and target_role:
            st.markdown(f"**Performer Role:** {performer_role.name} - {performer_role.requirements['description']}")
            st.markdown(f"**Target Audience:** {target_role.name} - {target_role.requirements['description']}")
        
        if practice_category:
            st.markdown(f"**Practice Session Category:** {practice_category}")
        
        # --- Granular Visual Feedback ---
        posture = scores.get('posture_score', 5.0)
        eye_contact = scores.get('eye_contact_score', 5.0)
        expression = scores.get('expression_score', 5.0)
        
        visual_warnings = []
        if posture < 6.0:
            visual_warnings.append("posture (try sitting upright with shoulders back)")
        if eye_contact < 6.0:
            visual_warnings.append("eye contact (look at the camera more consistently)")
        if expression < 6.0:
            visual_warnings.append("facial expression (try to appear more engaged)")
        
        if visual_warnings:
            st.warning(f" **Visuals:** Improve your {' and '.join(visual_warnings)}.")
        else:
            st.info(" **Visuals:** Great posture, eye contact, and engagement!")
        
        # Show sub-scores in an expander:
        with st.expander("Visual Sub-Scores"):
            vs1, vs2, vs3 = st.columns(3)
            vs1.metric("Posture", f"{posture} / 10")
            vs2.metric("Eye Contact", f"{eye_contact} / 10")
            vs3.metric("Expression", f"{expression} / 10")
            
        # --- Audio Feedback ---
        if scores.get('audio_performance_score', 0) < 5:
            st.warning(" **Audio:** Try to modulate your speaking pace and pitch. Reduce background noise.")
        else:
            st.info(" **Audio:** Strong pacing and volume control!")
        
        # --- Text Feedback ---
        text_score = scores.get('text_performance_score', 0)
        if text_score < 2:
            st.warning(f" **Text Quality ({text_score}/10):** Very poor grammar and many filler words. Significant improvement needed.")
        elif text_score < 4:
            st.warning(f" **Text Quality ({text_score}/10):** Focus on clearer sentence structure and reducing filler words.")
        elif text_score < 7:
            st.info(f"**Text Quality ({text_score}/10):** Acceptable grammar. Minor improvements in fluency would help.")
        elif text_score < 9:
            st.info(f"**Text Quality ({text_score}/10):** Good grammar with minor errors.")
        else:
            st.success(f" **Text Quality ({text_score}/10):** Strong grammar and professional language!")
        
        # --- Relevance Feedback & Recommendations ---
        rel_score      = scores.get("relevance_score", 0)
        rel_label      = scores.get("relevance_label", "")
        rel_reason     = scores.get("relevance_reason", "")
        rel_deductions = scores.get("relevance_deductions", [])
        rel_prompt     = scores.get("relevance_prompt_used", meta.get("prompt", ""))

        st.markdown("###  Prompt Relevance")

        # Score + label row
        rel_col1, rel_col2 = st.columns([1, 3])
        rel_col1.metric("Relevance Score", f"{rel_score} / 10")
        label_color = {"relevant": "✅", "partially_relevant": "⚠️", "irrelevant": "❌"}
        rel_col2.markdown(
            f"**Status:** {label_color.get(rel_label, '📊')} `{rel_label.replace('_', ' ').title() if rel_label else 'Pending'}`"
        )

        # Prompt used for evaluation
        if rel_prompt:
            with st.expander(" Evaluated Against", expanded=False):
                st.info(rel_prompt)

        # LLM reason
        if rel_reason:
            if rel_score >= 8:
                st.success(f"**Evaluator Feedback:** {rel_reason}")
            elif rel_score >= 5:
                st.info(f"**Evaluator Feedback:** {rel_reason}")
            else:
                st.warning(f"**Evaluator Feedback:** {rel_reason}")

        # Deductions applied
        if rel_deductions:
            with st.expander(" Deductions Applied", expanded=rel_score < 6):
                for d in rel_deductions:
                    st.markdown(f"- {d}")

        # Improvement recommendations based on target audience expectations
        st.markdown("####  How to Be More Precise")
        if rel_score >= 8:
            st.success(
                "Your answer was highly relevant. Keep anchoring responses to the "
                "specific needs and context of your audience."
            )
        else:
            tips = []

            # Generic tips based on score band
            if rel_score < 4:
                tips += [
                    "**Re-read the prompt** before answering — identify the core question.",
                    "**Open with a direct statement** that addresses what was asked.",
                    "**Avoid lengthy preambles** — get to the point within the first 2 sentences.",
                ]
            elif rel_score < 6:
                tips += [
                    "**Stay closer to the specific scenario** described in the prompt.",
                    "**Reduce tangents** — if you drift, consciously redirect back.",
                ]
            else:
                tips += [
                    "**Add specifics** — mention concrete examples or numbers relevant to the question.",
                ]

            # Audience-specific tips from persona expectations
            if target_role:
                expectations = target_role.requirements.get("expectations", [])
                if expectations:
                    tips.append(
                        f"**Speak to *{target_role.name}*'s expectations specifically:**"
                    )
                    for exp in expectations:
                        tips.append(f"  - {exp}")

            # Structural tip
            tips.append(
                "**Use the PREP structure:** Point → Reason → Example → Point (restate). "
                "This keeps answers focused and memorable."
            )

            for tip in tips:
                st.markdown(f"- {tip}")

