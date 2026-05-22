"""
Persona AI — Record & Analyse Session
======================================
Architecture:
  - Recorder opens in its OWN browser tab (localhost:8502) → full camera permissions
  - JS MediaRecorder auto-POSTs blob to /upload on same server
  - User returns to Streamlit tab, hits Analyse
  - Full batch pipeline runs → scorecard
"""

import os
import json
import pathlib
import tempfile
import threading
import http.server
import socketserver
import streamlit as st
from app.core.persona import persona_framework, performer_roles, target_roles

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Persona AI — Session Analyser",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Shared paths
# ─────────────────────────────────────────────────────────────────────────────
RECORDER_PORT = 8502
_RECORDER_DIR = pathlib.Path(tempfile.gettempdir()) / "persona_ai_recorder"
_RECORDER_DIR.mkdir(parents=True, exist_ok=True)
_UPLOAD_DIR   = pathlib.Path("uploads")
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
_READY_FILE   = _RECORDER_DIR / "latest.json"

# ─────────────────────────────────────────────────────────────────────────────
# Recorder HTML — full-page, opens in its own tab at localhost:8502
# After Stop: auto-POSTs blob to /upload (same origin = no CORS)
# ─────────────────────────────────────────────────────────────────────────────
RECORDER_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Persona AI — Recorder</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;background:#eef4fb;font-family:'Inter',system-ui,sans-serif;color:#172033}
body{display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:100vh;padding:24px}

.card{
  width:100%;max-width:760px;
  background:rgba(255,255,255,.94);
  border:1px solid rgba(23,32,51,.1);
  border-radius:20px;padding:28px;
  display:flex;flex-direction:column;gap:18px;
  box-shadow:0 24px 60px rgba(44,62,95,.16)
}

.header{text-align:center}
.logo{font-size:2rem;margin-bottom:6px}
.title{font-size:1.4rem;font-weight:800;
  background:linear-gradient(135deg,#4f8ef7,#8b5cf6);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.sub{font-size:.85rem;color:#60708a;margin-top:4px}

.preview-wrap{
  position:relative;border-radius:14px;overflow:hidden;
  background:#dce7f3;border:2px solid rgba(23,32,51,.08);
  aspect-ratio:16/9;width:100%
}
#preview,#playback{
  width:100%;height:100%;object-fit:cover;
  display:block;border-radius:12px
}
#playback{display:none}

.badge{
  position:absolute;top:12px;left:12px;
  background:rgba(0,0,0,.65);padding:5px 12px;border-radius:999px;
  font-size:11px;font-weight:700;letter-spacing:.07em;
  display:none;align-items:center;gap:6px
}
.dot{width:9px;height:9px;background:#f43f5e;border-radius:50%;animation:blink 1s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.15}}

.timer-badge{
  position:absolute;top:12px;right:12px;
  background:rgba(0,0,0,.55);padding:4px 12px;border-radius:999px;
  font-size:13px;font-weight:700;font-variant-numeric:tabular-nums;display:none
}

.controls{display:flex;gap:12px}
.btn{
  flex:1;padding:14px 20px;border:none;border-radius:12px;
  font-size:15px;font-weight:700;cursor:pointer;
  display:flex;align-items:center;justify-content:center;gap:8px;
  transition:opacity .2s,transform .15s
}
.btn:hover:not(:disabled){opacity:.82;transform:translateY(-1px)}
.btn:disabled{opacity:.3;cursor:not-allowed;transform:none}
.btn-rec {background:linear-gradient(135deg,#2563eb,#7c3aed);color:#fff}
.btn-stop{background:linear-gradient(135deg,#64748b,#475569);color:#fff}

.prog-wrap{display:none;height:5px;border-radius:99px;background:rgba(23,32,51,.08);overflow:hidden}
.prog-fill{height:100%;background:linear-gradient(90deg,#4f8ef7,#8b5cf6);width:0%;transition:width .3s}

.status-box{
  background:#f7faff;border:1px solid rgba(23,32,51,.08);
  border-radius:12px;padding:13px 16px;font-size:13px;color:#60708a;
  display:flex;align-items:flex-start;gap:10px;line-height:1.5;min-height:50px
}
.s-ico{font-size:18px;flex-shrink:0;margin-top:1px}

.done-box{
  display:none;background:rgba(16,185,129,.1);border:1px solid rgba(16,185,129,.3);
  border-radius:12px;padding:18px 20px;text-align:center
}
.done-box .done-title{font-size:1.05rem;font-weight:700;color:#10b981;margin-bottom:6px}
.done-box .done-sub{font-size:.85rem;color:#60708a;line-height:1.6}
.back-btn{
  display:inline-block;margin-top:14px;padding:10px 24px;
  background:linear-gradient(135deg,#4f8ef7,#8b5cf6);
  color:#fff;border-radius:10px;font-weight:700;font-size:13px;
  text-decoration:none;transition:opacity .2s
}
.back-btn:hover{opacity:.82}
</style>
</head>
<body>
<div class="card">
  <div class="header">
    <div class="logo">Persona AI</div>
    <div class="title">Persona AI — Session Recorder</div>
    <div class="sub">Record your session · it uploads automatically when you stop</div>
  </div>

  <div class="preview-wrap">
    <video id="preview" autoplay muted playsinline></video>
    <video id="playback" controls playsinline></video>
    <div class="badge" id="badge"><div class="dot"></div>REC</div>
    <div class="timer-badge" id="timerBadge">00:00</div>
  </div>

  <div class="controls">
    <button class="btn btn-rec" id="btnRec" onclick="startRec()">Start Recording</button>
    <button class="btn btn-stop" id="btnStop" onclick="stopRec()" disabled>Stop</button>
  </div>

  <div class="prog-wrap" id="progWrap">
    <div class="prog-fill" id="progFill"></div>
  </div>

  <div class="status-box" id="statusBox">
    <span class="s-ico"></span>
    <span id="statusText">Click <strong>Start Recording</strong> — your camera and microphone will open.</span>
  </div>

  <div class="done-box" id="doneBox">
    <div class="done-title">Recording uploaded successfully!</div>
    <div class="done-sub">
      Switch back to the <strong>Persona AI</strong> tab in your browser,<br>
      click <strong>"I've finished recording"</strong> then hit <strong>"Analyse Session"</strong>.
    </div>
    <a class="back-btn" href="javascript:window.close()">Close this tab</a>
  </div>
</div>

<script>
let mr=null,chunks=[],stream=null,ivl=null,sec=0,durationAlertShown=false;

const preview =document.getElementById('preview'),
      playback=document.getElementById('playback'),
      btnRec  =document.getElementById('btnRec'),
      btnStop =document.getElementById('btnStop'),
      badge   =document.getElementById('badge'),
      timerEl =document.getElementById('timerBadge'),
      progWrap=document.getElementById('progWrap'),
      progFill=document.getElementById('progFill'),
      statusEl=document.getElementById('statusText'),
      statusBox=document.getElementById('statusBox'),
      doneBox =document.getElementById('doneBox');

function setStatus(ico,html){
  statusBox.querySelector('.s-ico').textContent=ico;
  statusEl.innerHTML=html;
}
function fmt(s){return String(Math.floor(s/60)).padStart(2,'0')+':'+String(s%60).padStart(2,'0')}

async function startRec(){
  try{
    stream=await navigator.mediaDevices.getUserMedia({video:true,audio:true});
  }catch(e){
    setStatus('','<strong>Camera access denied.</strong> Please allow camera &amp; microphone access in your browser and refresh.');
    return;
  }
  preview.srcObject=stream;
  playback.style.display='none';
  preview.style.display='block';

  const mime=
    MediaRecorder.isTypeSupported('video/webm;codecs=vp9,opus') ? 'video/webm;codecs=vp9,opus' :
    MediaRecorder.isTypeSupported('video/webm')                 ? 'video/webm' : 'video/mp4';

  chunks=[];
  mr=new MediaRecorder(stream,{mimeType:mime});
  mr.ondataavailable=e=>{if(e.data&&e.data.size>0)chunks.push(e.data)};
  mr.onstop=onStop;
  mr.start(200);

  btnRec.disabled=true; btnStop.disabled=false;
  badge.style.display='flex'; timerEl.style.display='block';
  sec=0; durationAlertShown=false; timerEl.textContent=fmt(sec);
  ivl=setInterval(()=>{
    sec++;
    timerEl.textContent=fmt(sec);
    if(sec>300 && !durationAlertShown){
      durationAlertShown=true;
      alert('You have crossed 5 minutes. Please keep the speech shorter.');
    }
  },1000);
  setStatus('','Recording… speak clearly and look at your camera.');
}

function stopRec(){
  if(mr&&mr.state!=='inactive')mr.stop();
  if(stream)stream.getTracks().forEach(t=>t.stop());
  clearInterval(ivl);
  badge.style.display='none'; timerEl.style.display='none';
  btnStop.disabled=true; btnRec.disabled=true;
  setStatus('','Finalising recording…');
}

async function onStop(){
  const blob=new Blob(chunks,{type:mr.mimeType||'video/webm'});
  playback.src=URL.createObjectURL(blob);
  playback.style.display='block'; preview.style.display='none';

  setStatus('','Uploading to Persona AI analyser…');
  progWrap.style.display='block'; progFill.style.width='25%';

  try{
    const ext=(mr.mimeType||'').includes('mp4') ? '.mp4' : '.webm';
    progFill.style.width='60%';
    const res=await fetch('/upload',{
      method:'POST',
      headers:{'Content-Type':blob.type,'X-Ext':ext},
      body:blob
    });
    progFill.style.width='100%';
    if(res.ok){
      statusBox.style.display='none';
      doneBox.style.display='block';
    }else{
      setStatus('','Upload failed (HTTP '+res.status+'). Please try again.');
      btnRec.disabled=false;
    }
  }catch(e){
    setStatus('','Upload failed: '+e.message+'. Make sure Persona AI is running.');
    btnRec.disabled=false;
  }
}
</script>
</body>
</html>
"""

# ─────────────────────────────────────────────────────────────────────────────
# Background HTTP server — serves recorder page + receives uploaded recording
# ─────────────────────────────────────────────────────────────────────────────
_RECORDER_DIR.joinpath("index.html").write_text(RECORDER_HTML, encoding="utf-8")


class _RecorderHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_): pass

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Ext")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self):
        data = (_RECORDER_DIR / "index.html").read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if self.path != "/upload":
            self.send_response(404); self.end_headers(); return
        length  = int(self.headers.get("Content-Length", 0))
        ext     = self.headers.get("X-Ext", ".webm")
        payload = self.rfile.read(length)
        fd, save_path = tempfile.mkstemp(suffix=ext, dir=str(_UPLOAD_DIR))
        os.write(fd, payload); os.close(fd)
        _READY_FILE.write_text(json.dumps({"path": save_path}), encoding="utf-8")
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"OK")


class _ReusingServer(socketserver.TCPServer):
    allow_reuse_address = True


def _start_server():
    try:
        with _ReusingServer(("127.0.0.1", RECORDER_PORT), _RecorderHandler) as s:
            s.serve_forever()
    except Exception:
        pass


if "recorder_server_started" not in st.session_state:
    threading.Thread(target=_start_server, daemon=True).start()
    st.session_state["recorder_server_started"] = True

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
:root{
  --bg-primary:#eef4fb;--bg-card:rgba(255,255,255,.92);--bg-glass:rgba(255,255,255,.72);
  --border:rgba(23,32,51,.1);--blue:#2563eb;--violet:#7c3aed;
  --emerald:#059669;--amber:#d97706;--rose:#e11d48;--text:#172033;--muted:#60708a;
}
html,body,[class*="css"]{font-family:'Inter',sans-serif;background:var(--bg-primary);color:var(--text)}
.stApp{background:linear-gradient(135deg,#eef4fb 0%,#f8fbff 48%,#edf1ff 100%);background-attachment:fixed}
header[data-testid="stHeader"]{display:none}

.hero{background:linear-gradient(135deg,#ffffff,#eef6ff 52%,#f3f0ff);border:1px solid var(--border);
  border-radius:20px;padding:28px 36px;margin-bottom:22px;display:flex;align-items:center;gap:16px;
  position:relative;overflow:hidden;box-shadow:0 16px 42px rgba(44,62,95,.11)}
.hero::before{content:'';position:absolute;top:-60px;right:-60px;width:220px;height:220px;
  background:radial-gradient(circle,rgba(37,99,235,.1) 0%,transparent 70%);pointer-events:none}
.hero-icon{font-size:2.6rem}
.hero-title{font-size:1.8rem;font-weight:800;margin:0;
  background:linear-gradient(135deg,#4f8ef7,#8b5cf6);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.hero-sub{font-size:.88rem;color:var(--muted);margin:3px 0 0}

.step-card{background:var(--bg-card);border:1px solid var(--border);border-radius:16px;
  padding:20px 24px;margin-bottom:16px;backdrop-filter:blur(10px);box-shadow:0 10px 30px rgba(44,62,95,.08)}
.step-label{font-size:.66rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;
  color:var(--blue);margin-bottom:4px}
.step-title{font-size:1.08rem;font-weight:700;color:var(--text)}

.rec-launch{
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  gap:16px;background:rgba(37,99,235,.06);border:2px dashed rgba(37,99,235,.24);
  border-radius:16px;padding:36px 24px;text-align:center;margin:4px 0
}
.rec-btn-link{
  display:inline-flex;align-items:center;gap:10px;
  background:linear-gradient(135deg,#2563eb,#7c3aed);
  color:#fff;padding:15px 36px;border-radius:12px;
  font-size:1rem;font-weight:800;text-decoration:none;letter-spacing:.02em;
  transition:opacity .2s,transform .15s;box-shadow:0 8px 24px rgba(37,99,235,.28)
}
.rec-btn-link:hover{opacity:.85;transform:translateY(-2px)}
.rec-hint{font-size:.82rem;color:var(--muted);line-height:1.6;max-width:420px}

.metric-card{background:var(--bg-glass);border:1px solid var(--border);border-radius:12px;
  padding:18px 20px;text-align:center;transition:transform .2s,border-color .2s}
.metric-card:hover{transform:translateY(-2px);border-color:rgba(37,99,235,.3)}
.metric-value{font-size:2.2rem;font-weight:900;line-height:1;margin-bottom:4px}
.metric-label{font-size:.74rem;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.05em}
.score-great{color:#10b981}.score-good{color:#4f8ef7}.score-ok{color:#f59e0b}.score-poor{color:#f43f5e}

.hr{height:1px;background:var(--border);margin:20px 0}
.tag-pill{display:inline-block;padding:3px 10px;border-radius:999px;font-size:.7rem;font-weight:700;
  letter-spacing:.05em;text-transform:uppercase}
.tag-relevant{background:rgba(16,185,129,.15);color:#10b981}
.tag-partial{background:rgba(245,158,11,.15);color:#f59e0b}
.tag-irrelevant{background:rgba(244,63,94,.15);color:#f43f5e}

.stTabs [role="tablist"]{background:var(--bg-glass);border-radius:12px;border:1px solid var(--border);padding:4px;gap:4px}
.stTabs [role="tab"]{border-radius:8px !important;font-weight:600 !important;color:var(--muted) !important}
.stTabs [role="tab"][aria-selected="true"]{background:linear-gradient(135deg,#4f8ef7,#8b5cf6) !important;color:#fff !important}
[data-testid="stExpander"]{background:var(--bg-glass);border:1px solid var(--border) !important;border-radius:12px}
.stMetric label{color:var(--muted) !important;font-weight:500 !important}
.stMetric [data-testid="stMetricValue"]{color:var(--text) !important;font-weight:800 !important}
button[kind="primary"]{background:linear-gradient(135deg,#4f8ef7,#8b5cf6) !important;border:none !important;
  border-radius:10px !important;font-weight:700 !important}
button[kind="primary"]:hover{opacity:.86 !important}
button[kind="secondary"]{background:var(--bg-glass) !important;border-color:var(--border) !important;
  color:var(--text) !important;border-radius:10px !important;font-weight:600 !important}
.stSelectbox>div,.stTextArea>div{background:var(--bg-glass) !important;border-color:var(--border) !important;border-radius:10px !important}
.stProgress>div>div>div{background:linear-gradient(90deg,#4f8ef7,#8b5cf6) !important;border-radius:99px}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""<div class="hero">
  <div class="hero-icon">Persona AI</div>
  <div>
    <div class="hero-title">Persona AI — Session Analyser</div>
    <div class="hero-sub">Record · Analyse · Improve — full AI scorecard in seconds</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def score_cls(v): return "score-great" if v>=8 else "score-good" if v>=6 else "score-ok" if v>=4 else "score-poor"
def score_label(v): return "Strong" if v>=8 else "Good" if v>=6 else "Needs work" if v>=4 else "Low"
def metric_card(col, label, value):
    col.markdown(f"""<div class="metric-card">
      <div class="metric-value {score_cls(value)}">{value}</div>
      <div class="metric-label">{label}</div></div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Session Setup
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("Step 1 — Session Setup (Persona & Prompt)", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        selected_performer = st.selectbox("Performer Role (You)", [r.name for r in performer_roles], key="sel_p")
        p_obj = persona_framework.get_role(selected_performer)
        if p_obj: st.caption(f"{p_obj.requirements['description']}")
    with c2:
        selected_target = st.selectbox("Target Audience", [r.name for r in target_roles], key="sel_t")
        t_obj = persona_framework.get_role(selected_target)
        if t_obj:
            st.caption(f"{t_obj.requirements['description']}")
            exps = t_obj.requirements.get("expectations", [])
            if exps: st.caption("Expects: "+ " • ".join(exps))
    prompt_input = st.text_area(
        "Prompt / Question",
        value="Introduce yourself and explain the value you bring to the customer.",
        height=72, key="prompt_input",
    )


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Record
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""<div class="step-card">
  <div class="step-label">Step 2</div>
  <div class="step-title"> Record Your Session</div>
</div>""", unsafe_allow_html=True)

st.markdown(f"""<div class="rec-launch">
  <a class="rec-btn-link" href="http://localhost:{RECORDER_PORT}" target="_blank">
    Open Recorder
  </a>
  <div class="rec-hint">
    A <strong>full-screen recorder</strong> opens in a new tab — camera &amp; mic permissions work there.<br>
    Record your session, hit <strong>Stop</strong>, then come back here and click below.
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Check status + Analyse
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""<div class="step-card">
  <div class="step-label">Step 3</div>
  <div class="step-title"> Analyse Session</div>
</div>""", unsafe_allow_html=True)

# Detect recording
def _check_recording():
    if not _READY_FILE.exists(): return False, None
    try:
        p = json.loads(_READY_FILE.read_text())["path"]
        return os.path.isfile(p), p
    except Exception:
        return False, None

recording_ready, recording_path = _check_recording()

col_refresh, col_analyse = st.columns([1, 3])
with col_refresh:
    if st.button("Recording Ready", key="btn_refresh", use_container_width=True):
        st.rerun()

with col_analyse:
    analyse_clicked = st.button(
        "Analyse Session",
        type="primary",
        disabled=not recording_ready,
        use_container_width=True,
        key="btn_analyse",
    )

if recording_ready:
    st.success("Recording received and ready! Hit **Analyse Session** to get your scorecard.")
else:
    st.info("After recording in the new tab, click **Recording Ready** to check status.")


# ─────────────────────────────────────────────────────────────────────────────
# Analysis pipeline
# ─────────────────────────────────────────────────────────────────────────────
if analyse_clicked and recording_ready and recording_path:
    try: _READY_FILE.unlink()
    except Exception: pass

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown("### Analysis in Progress…")
    STEP_LABELS = [
        "Extracting audio & transcribing speech",
        "Analysing video — posture, eye contact, expression",
        "Evaluating speech quality with LLM",
        "Generating body language interpretation",
        "Scoring text quality & grammar",
        "Evaluating prompt relevance",
        "Computing final scores",
    ]
    prog  = st.progress(0)
    stxt  = st.empty()

    def _prog(step, total, label):
        prog.progress(step / total, text=label)
        stxt.markdown(f"**{STEP_LABELS[step - 1]}…**")

    from app.pipeline.batch_analyser import run_batch_analysis
    try:
        results = run_batch_analysis(
            video_path=recording_path,
            prompt=prompt_input,
            performer_role=selected_performer,
            target_role=selected_target,
            progress_fn=_prog,
        )
        prog.progress(1.0, text="Complete!")
        stxt.markdown("**Analysis complete!**")
        st.session_state["results"]       = results
        st.session_state["res_prompt"]    = prompt_input
        st.session_state["res_performer"] = selected_performer
        st.session_state["res_target"]    = selected_target
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Results Dashboard
# ─────────────────────────────────────────────────────────────────────────────
if "results" in st.session_state:
    results  = st.session_state["results"]
    scores   = results.get("scores", {})
    transcript    = results.get("transcript", "")
    audio_metrics = results.get("audio_metrics", {})
    audio_signal  = results.get("audio_signal_analysis", {})
    audio_llm     = results.get("audio_llm", {})
    video_metrics = results.get("video_metrics", {})
    body_llm      = results.get("body_llm", {})
    text_result   = results.get("text_result", {})
    enh_prompt    = results.get("enhanced_prompt", st.session_state.get("res_prompt", ""))

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown("""<div class="step-card">
      <div class="step-label">Step 4</div>
      <div class="step-title">Your Session Scorecard</div>
    </div>""", unsafe_allow_html=True)

    cx1, cx2, cx3 = st.columns(3)
    cx1.markdown(f"**Performer:** {st.session_state.get('res_performer','—')}")
    cx2.markdown(f"**Target:** {st.session_state.get('res_target','—')}")
    cx3.markdown(f"**Time:** {results.get('total_time_sec','—')} s")

    ov = scores.get("overall_score", 0)
    st.markdown(f"""<div style="background:linear-gradient(135deg,rgba(79,142,247,.1),rgba(139,92,246,.1));
      border:1px solid rgba(79,142,247,.25);border-radius:18px;padding:28px;
      text-align:center;margin:16px 0 24px">
      <div style="font-size:.72rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;
        color:#8b949e;margin-bottom:8px">OVERALL PERFORMANCE SCORE</div>
      <div class="{score_cls(ov)}"style="font-size:4rem;font-weight:900;line-height:1">{ov}</div>
      <div style="font-size:.92rem;color:#8b949e;margin-top:4px">out of 10</div>
    </div>""", unsafe_allow_html=True)

    sc1, sc2, sc3, sc4 = st.columns(4)
    metric_card(sc1, "Video",     scores.get("visual_performance_score", 0))
    metric_card(sc2, "Audio",    scores.get("audio_performance_score", 0))
    metric_card(sc3, "Text",      scores.get("text_performance_score", 0))
    metric_card(sc4, "Relevance", scores.get("relevance_score", 0))
    st.markdown("<br>", unsafe_allow_html=True)

    tab_ov, tab_vid, tab_aud, tab_txt = st.tabs(
        ["Overall", "Video", "Audio", "Text & Relevance"]
    )

    # ── Overall ───────────────────────────────────────────────────────────────
    with tab_ov:
        st.markdown("### Performance Snapshot")
        try:
            import pandas as pd, altair as alt
            df = pd.DataFrame({
                "Dimension": ["Video","Audio","Text","Relevance"],
                "Score": [scores.get("visual_performance_score",0), scores.get("audio_performance_score",0),
                          scores.get("text_performance_score",0), scores.get("relevance_score",0)],
            })
            chart = (alt.Chart(df).mark_bar(cornerRadiusTopLeft=6,cornerRadiusTopRight=6)
                .encode(
                    x=alt.X("Dimension:N",sort=None,axis=alt.Axis(labelColor="#8b949e",labelFontSize=13)),
                    y=alt.Y("Score:Q",scale=alt.Scale(domain=[0,10]),axis=alt.Axis(labelColor="#8b949e")),
                    color=alt.Color("Score:Q",scale=alt.Scale(domain=[0,5,8,10],
                                    range=["#f43f5e","#f59e0b","#4f8ef7","#10b981"]),legend=None),
                    tooltip=["Dimension","Score"])
                .properties(height=220)
                .configure_view(strokeWidth=0,fill="#00000000")
                .configure_axis(grid=False,domain=False))
            st.altair_chart(chart, use_container_width=True)
        except Exception:
            for d,v in zip(["Video","Audio","Text","Relevance"],
                           [scores.get(k,0) for k in ["visual_performance_score","audio_performance_score",
                                                       "text_performance_score","relevance_score"]]):
                st.metric(d, f"{v}/10")

        with st.expander("Full Transcript", expanded=False):
            st.markdown(f"_{transcript or 'No transcript available.'}_")

        strengths, gaps = [], []
        for lbl, key in [("Video body language","visual_performance_score"),
                         ("Audio delivery","audio_performance_score"),
                         ("Text quality","text_performance_score"),
                         ("Prompt relevance","relevance_score")]:
            v = scores.get(key, 0)
            if v >= 7: strengths.append(f"{score_label(v)} {lbl} ({v}/10)")
            elif v < 5: gaps.append(f"{score_label(v)} {lbl} ({v}/10)")

        ga, gb = st.columns(2)
        with ga:
            st.markdown("#### Strengths")
            for s in (strengths or ["Keep practising to build strengths!"]): st.markdown(f"- {s}")
        with gb:
            st.markdown("#### Areas to Improve")
            for g in (gaps or ["No major gaps — great session!"]): st.markdown(f"- {g}")

    # ── Video ─────────────────────────────────────────────────────────────────
    with tab_vid:
        st.markdown("### Body Language Analysis")
        v1,v2,v3,v4 = st.columns(4)
        metric_card(v1,"Posture",       scores.get("posture_score",0))
        metric_card(v2,"Eye Contact",   scores.get("eye_contact_score",0))
        metric_card(v3,"Expression",    scores.get("expression_score",0))
        metric_card(v4,"Head Stability",scores.get("head_stability_score",0))
        if body_llm:
            st.markdown(f"<br>**Overall:** {body_llm.get('overall_body_language','—')}", unsafe_allow_html=True)
            if (interp := body_llm.get("body_language_interpretation","")): st.info(f"{interp}")
            ba,bb,bc = st.columns(3)
            with ba: st.markdown("**Posture**"); st.caption(body_llm.get("posture_analysis","—"))
            with bb: st.markdown("**Eye Contact**"); st.caption(body_llm.get("eye_contact_analysis","—"))
            with bc: st.markdown("**Expression**"); st.caption(body_llm.get("expression_analysis","—"))
            if (sugg := body_llm.get("improvement_suggestions","")): st.warning(f"{sugg}")
        st.markdown("#### Coaching Tips")
        for lbl, key, bad, good in [
            ("Posture","posture_score","Sit upright — shoulders back, spine tall.","Great posture!"),
            ("Eye Contact","eye_contact_score","Look at the camera lens, not the screen.","Excellent gaze!"),
            ("Expression","expression_score","Vary expressions — appear engaged and warm.","Good facial engagement!"),
            ("Head Stability","head_stability_score","Minimise excessive head movement.","Steady and composed!"),
        ]:
            v = scores.get(key, 5)
            (st.warning if v < 6 else st.success)(f"{lbl}: {bad if v < 6 else good}")

    # ── Audio ─────────────────────────────────────────────────────────────────
    with tab_aud:
        st.markdown("### Audio & Speech Analysis")
        a1,a2,a3 = st.columns(3)
        metric_card(a1,"Clarity",      scores.get("speech_clarity_score",0))
        metric_card(a2,"Confidence",    scores.get("confidence_score",0))
        metric_card(a3,"Communication", scores.get("communication_score",0))
        st.markdown("<br>#### Speech Signal Metrics", unsafe_allow_html=True)
        sm1,sm2,sm3,sm4 = st.columns(4)
        sm1.metric("WPM",           audio_metrics.get("wpm","—"))
        sm2.metric("Filler Score",  audio_metrics.get("filler_score","—"))
        sm3.metric("Pause Rate",    audio_metrics.get("pause_rate","—"))
        sm4.metric("Silence Ratio", audio_signal.get("silence_ratio","—"))
        with st.expander("Interpretation Details"):
            for k,l in [("wpm_interpretation","WPM"),("filler_interpretation","Filler"),("pause_interpretation","Pause")]:
                if (v := audio_metrics.get(k,"—")) != "—": st.markdown(f"**{l}:** {v}")
            for k,l in [("energy_interpretation","Energy"),("pitch_interpretation","Pitch"),("silence_interpretation","Silence")]:
                if (v := audio_signal.get(k,"—")) != "—": st.markdown(f"**{l}:** {v}")
        if audio_llm:
            st.markdown("#### LLM Feedback")
            for key,label in [("tone_quality","Tone"),("voice_quality_feedback","Voice"),
                               ("engagement_level","Engagement"),("communication_style","Style"),
                               ("language_fluency","Fluency"),("professionalism_level","Professionalism"),
                               ("filler_word_usage","Fillers")]:
                if (v := audio_llm.get(key,"")): st.markdown(f"**{label}:** {v}")

    # ── Text & Relevance ──────────────────────────────────────────────────────
    with tab_txt:
        st.markdown("### Text Quality & Relevance")
        tr1,tr2 = st.columns(2)
        metric_card(tr1,"Text Quality", scores.get("text_performance_score",0))
        metric_card(tr2,"Relevance",    scores.get("relevance_score",0))
        st.markdown("<br>#### Grammar & Fluency", unsafe_allow_html=True)
        tm1,tm2 = st.columns(2)
        tm1.metric("Grammar Errors", text_result.get("error_count","—"))
        tm2.metric("Filler Words",   text_result.get("filler_count","—"))
        if (fb := text_result.get("feedback","")):
            t_sc = scores.get("text_performance_score",5)
            (st.success if t_sc>=8 else st.info if t_sc>=6 else st.warning)(f"**Evaluator:** {fb}")
        det_langs = text_result.get("detected_languages") or results.get("detected_languages", [])
        if det_langs and det_langs != ["unknown"]:
            st.markdown("**Languages:** " + ", ".join(f"**{l}**" for l in det_langs))
            regional_langs = [lang for lang in det_langs if lang.lower() != "english"]
            if len(regional_langs) > 2:
                st.warning(
                    "You are using more than 2 regional languages "
                    f"({', '.join(regional_langs)}), which is not recommended."
                )

        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
        st.markdown("#### Prompt Relevance")
        rel_label   = scores.get("relevance_label","")
        rel_reason  = scores.get("relevance_reason","")
        rel_deducts = scores.get("relevance_deductions",[])
        tag_map = {"relevant":'<span class="tag-pill tag-relevant"> Relevant</span>',
                   "partially_relevant":'<span class="tag-pill tag-partial"> Partially Relevant</span>',
                   "irrelevant":'<span class="tag-pill tag-irrelevant"> Irrelevant</span>'}
        st.markdown(f"**Status:** {tag_map.get(rel_label, rel_label)}", unsafe_allow_html=True)
        with st.expander("Prompt evaluated against", expanded=False): st.info(enh_prompt)
        r_sc = scores.get("relevance_score",0)
        if rel_reason:
            (st.success if r_sc>=8 else st.info if r_sc>=5 else st.warning)(f"**Evaluator:** {rel_reason}")
        if rel_deducts:
            with st.expander("Deductions Applied", expanded=r_sc<6):
                for d in rel_deducts: st.markdown(f"- {d}")

        st.markdown("#### How to Improve Relevance")
        t_role = persona_framework.get_role(st.session_state.get("res_target",""))
        if r_sc >= 8:
            st.success("Highly relevant — keep anchoring to your audience's specific needs.")
        else:
            tips = []
            if r_sc < 4:
                tips += ["**Re-read the prompt** — identify the core question before answering.",
                         "**Open with a direct statement** addressing what was asked.",
                         "**Avoid long preambles** — make your point within 2 sentences."]
            elif r_sc < 6:
                tips += ["**Stay closer to the scenario** described in the prompt.",
                         "**Reduce tangents** — consciously redirect back to the topic."]
            else:
                tips += ["**Add specifics** — concrete examples or numbers."]
            if t_role:
                exps = t_role.requirements.get("expectations",[])
                if exps:
                    tips.append(f"**Speak to *{t_role.name}*'s expectations:**")
                    tips += [f"- {e}" for e in exps]
            tips.append("**Use PREP:** Point → Reason → Example → Point (restate).")
            for tip in tips: st.markdown(f"- {tip}")
