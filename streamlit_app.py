import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import cv2
import av
import queue
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
        
        if len(self.audio_frames) > 50:
            import numpy as np
            audio_data = np.concatenate(self.audio_frames, axis=1)
            self.result_queue.put((audio_data, frame.sample_rate))
            self.audio_frames.clear()
            
        return frame

# Header
st.title("Live Multimodal Orchestration Test")

# Prompt / Interview Question input
prompt_input = st.text_input(
    "Prompt",
    value="Introduce yourself briefly.",
    help="Enter the question the candidate should answer. Relevance is scored against this."
)

if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = create_orchestrator()
    st.session_state.state = {
        "messages": [],
        "metadata": {"session_id": "live_streamlit", "prompt": prompt_input},
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
    alert_placeholder = st.empty()
    debug_placeholder = st.empty()

if webrtc_ctx.state.playing:
    if webrtc_ctx.video_processor:
        # Loop to consume queue continuously while streaming
        while True:
            try:
                # Wait for frames buffer from WEBRTC processor
                frames_chunk = webrtc_ctx.video_processor.result_queue.get(timeout=1.0)
                
                audio_chunk = None
                if webrtc_ctx.audio_processor:
                    try:
                        audio_chunk = webrtc_ctx.audio_processor.result_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                state = st.session_state.state
                # Sync the prompt in case the user changed it mid-session
                state["metadata"]["prompt"] = prompt_input
                state["recent_video_features"] = [{"raw_frames": frames_chunk}]
                if audio_chunk:
                    state["recent_acoustic_features"] = [{"raw_audio": audio_chunk}]
                
                # Invoke graph asynchronously internally 
                state = st.session_state.orchestrator.invoke(state)
                st.session_state.state = state
                
                # UI Render dynamically
                with score_placeholder.container():
                    sc = state["scores"].get("overall_score", 0)
                    st.metric("Overall Score Tracking", f"{sc} / 10")
                
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
                        st.write("**Detected Languages:**", ts.get("stt_detected_languages", []))
                        st.write("**API Counts:**", ts.get("api_lang_counts", {}))
                        st.write("**API Max Confidence:**", ts.get("api_lang_max_prob", {}))
                    
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
        st.info(f"**🗣️ Languages Spoken Profile:** {', '.join(langs_used)}")
        
        st.markdown("### Key Takeaways")
        
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
        
        # Show sub-scores in an expander
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
        if text_score < 4:
            st.warning(f" **Text Quality ({text_score}/10):** Focus on clearer sentence structure and reducing filler words.")
        elif text_score < 7:
            st.info(f"**Text Quality ({text_score}/10):** Acceptable grammar. Minor improvements in fluency would help.")
        else:
            st.info(f" **Text Quality ({text_score}/10):** Strong grammar and professional language!")
        
        # --- Relevance Feedback ---
        rel_score = scores.get('relevance_score', 0)
        if rel_score < 3:
            st.warning(f" **Relevance ({rel_score}/10):** Your answer didn't clearly address the question. Stay focused on the prompt.")
        elif rel_score < 6:
            st.info(f"**Relevance ({rel_score}/10):** Partially relevant. Try to address the question more directly.")
        else:
            st.info(f"**Relevance ({rel_score}/10):** Good job staying on topic!")
