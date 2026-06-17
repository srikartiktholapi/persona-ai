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
from app.core.db import fetch_activities, fetch_personas, fetch_tasks
from app.core.persona import persona_framework, performer_roles, target_roles

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Persona AI — Session Analyser",
  page_icon="🎥",
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
.hero-sub{font-size:.95rem;color:#374151;margin:8px 0 0;line-height:1.6}

.step-card{background:#ffffff;border:1px solid rgba(23,32,51,.12);border-radius:18px;
  padding:24px 26px;margin-bottom:18px;backdrop-filter:blur(12px);box-shadow:0 18px 40px rgba(44,62,95,.12)}
.step-label{font-size:.72rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;
  color:#2563eb;margin-bottom:6px}
.step-title{font-size:1.18rem;font-weight:800;color:#111827;line-height:1.25;margin-bottom:4px}
.section-heading{font-size:1rem;font-weight:800;color:#111827;margin:20px 0 12px;letter-spacing:.01em}

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
.score-na{color:#64748b}

.hr{height:1px;background:var(--border);margin:20px 0}
.tag-pill{display:inline-block;padding:3px 10px;border-radius:999px;font-size:.7rem;font-weight:700;
  letter-spacing:.05em;text-transform:uppercase}
.tag-relevant{background:rgba(16,185,129,.15) !important;color:#10b981 !important;-webkit-text-fill-color:#10b981 !important}
.tag-partial{background:rgba(245,158,11,.15) !important;color:#d97706 !important;-webkit-text-fill-color:#d97706 !important}
.tag-irrelevant{background:rgba(244,63,94,.15) !important;color:#e11d48 !important;-webkit-text-fill-color:#e11d48 !important}

/* ── Expander: header button only gets white text, NOT the body ── */
button[data-testid="stExpanderToggleButton"] {color:#111827 !important}
button[data-testid="stExpanderToggleButton"] > * {color:#111827 !important}
button[data-testid="stExpanderToggleButton"] p {color:#111827 !important}
[data-testid="stExpander"] > div:first-child button > div > p {color:#111827 !important}
/* Expander body: always dark text */
[data-testid="stExpanderDetails"] p,
[data-testid="stExpanderDetails"] li,
[data-testid="stExpanderDetails"] span,
[data-testid="stExpanderDetails"] div,
[data-testid="stExpanderDetails"] label{
  color:#111827 !important;
  -webkit-text-fill-color:#111827 !important;
}

.stSelectbox svg {color:#fff !important}
.stSelectbox [data-testid="stSelectbox-select"] {color:#fff !important}
.stSelectbox [role="combobox"] {color:#fff !important}
.stSelectbox > div > div {color:#fff !important}
.stTabs [role="tablist"]{background:var(--bg-glass);border-radius:12px;border:1px solid var(--border);padding:4px;gap:4px}
.stTabs [role="tab"]{border-radius:8px !important;font-weight:600 !important;color:var(--muted) !important}
.stTabs [role="tab"][aria-selected="true"]{background:linear-gradient(135deg,#4f8ef7,#8b5cf6) !important;color:#fff !important}
[data-testid="stExpander"]{background:var(--bg-glass);border:1px solid var(--border) !important;border-radius:12px;color:var(--text) !important}
.stMetric label{color:var(--muted) !important;font-weight:500 !important}
.stMetric [data-testid="stMetricValue"]{color:var(--text) !important;font-weight:800 !important}
button[kind="primary"]{background:linear-gradient(135deg,#4f8ef7,#8b5cf6) !important;border:none !important;
  border-radius:10px !important;font-weight:700 !important;color:#fff !important}
button[kind="primary"]:hover{opacity:.86 !important}
button[kind="secondary"]{background:var(--bg-glass) !important;border-color:var(--border) !important;
  color:var(--text) !important;border-radius:10px !important;font-weight:600 !important}
.stSelectbox label,.stTextArea label,[data-testid="stWidgetLabel"] *{color:var(--text) !important}
.stSelectbox>div,.stTextArea>div{background:#fff !important;border-color:var(--border) !important;border-radius:10px !important}
.stSelectbox [data-baseweb="select"],
.stSelectbox [data-baseweb="select"] > div,
.stTextArea textarea{
  background:#fff !important;
  border-color:rgba(31,41,55,.16) !important;
  color:#111827 !important;
  -webkit-text-fill-color:#111827 !important;
}
.stSelectbox [data-baseweb="select"] *,
.stSelectbox [role="combobox"] *{
  color:#111827 !important;
  -webkit-text-fill-color:#111827 !important;
}
.stTextArea textarea::placeholder{
  color:#6b7280 !important;
  -webkit-text-fill-color:#6b7280 !important;
  opacity:1 !important;
}
[data-baseweb="popover"] [role="listbox"],
[data-baseweb="popover"] [role="option"],
[data-baseweb="popover"] [role="option"] *{
  background:#fff !important;
  color:var(--text) !important;
  -webkit-text-fill-color:var(--text) !important;
}
.stProgress>div>div>div{background:linear-gradient(90deg,#4f8ef7,#8b5cf6) !important;border-radius:99px}
input, textarea, select {color:inherit !important}

/* ── Main content text always readable ── */
.stMarkdown p, .stMarkdown li, .stMarkdown span,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] ul,
[data-testid="stMarkdownContainer"] ol{
  color:#111827 !important;
  -webkit-text-fill-color:#111827 !important;
}
/* Status tag pills keep their own colors */
.tag-pill{color:unset !important; -webkit-text-fill-color:unset !important;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* UI polish pass: light, high-contrast dashboard styling */
.block-container{
  max-width:1180px;
  padding-top:22px;
  padding-bottom:52px;
}

.stApp{
  background:
    linear-gradient(180deg,rgba(255,255,255,.72),rgba(255,255,255,0) 280px),
    linear-gradient(135deg,#edf6ff 0%,#f8fbff 46%,#f1f5ff 100%) !important;
}

.hero{
  min-height:156px;
  align-items:flex-start;
  justify-content:space-between;
  border-radius:18px;
  padding:30px 34px;
  background:
    linear-gradient(135deg,rgba(255,255,255,.98),rgba(239,246,255,.95) 55%,rgba(245,243,255,.95)),
    radial-gradient(circle at 88% 18%,rgba(79,142,247,.22),transparent 32%) !important;
  border:1px solid rgba(37,99,235,.14) !important;
  box-shadow:0 18px 50px rgba(31,41,55,.12) !important;
}
.hero::before{display:none}
.hero-icon{
  color:#111827;
  font-size:2.8rem;
  font-weight:800;
  letter-spacing:0;
  line-height:1;
}
.hero-copy{max-width:620px}
.hero-kicker{
  color:#2563eb;
  font-size:.78rem;
  font-weight:800;
  letter-spacing:.12em;
  text-transform:uppercase;
  margin-bottom:10px;
}
.hero-title{
  font-size:2rem !important;
  color:#111827 !important;
  background:none !important;
  -webkit-text-fill-color:#111827 !important;
}
.hero-sub{
  max-width:560px;
  color:#4b5563 !important;
  font-size:1rem !important;
}
.hero-pills{
  display:flex;
  gap:10px;
  flex-wrap:wrap;
  margin-top:18px;
}
.hero-pill{
  display:inline-flex;
  align-items:center;
  min-height:30px;
  padding:6px 12px;
  border-radius:999px;
  color:#1f2937;
  background:rgba(255,255,255,.82);
  border:1px solid rgba(37,99,235,.16);
  font-size:.78rem;
  font-weight:700;
}

.step-card{
  display:flex;
  align-items:center;
  gap:14px;
  padding:18px 20px !important;
  border-radius:14px !important;
  background:#fff !important;
  box-shadow:0 12px 32px rgba(31,41,55,.08) !important;
}
.step-label{
  display:inline-flex;
  align-items:center;
  justify-content:center;
  min-width:76px;
  min-height:32px;
  margin:0 !important;
  border-radius:999px;
  color:#2563eb !important;
  background:#eff6ff;
  border:1px solid #bfdbfe;
  letter-spacing:.08em !important;
}
.step-title{
  margin:0 !important;
  color:#111827 !important;
}

[data-testid="stExpander"]{
  overflow:hidden;
  background:#fff !important;
  border:1px solid rgba(31,41,55,.1) !important;
  border-radius:14px !important;
  box-shadow:0 14px 36px rgba(31,41,55,.08);
}
[data-testid="stExpander"] details > summary,
[data-testid="stExpander"] button[data-testid="stExpanderToggleButton"]{
  background:linear-gradient(135deg,#f8fbff,#eef6ff) !important;
  border-bottom:1px solid rgba(31,41,55,.08);
}
/* Expander header label - dark text */
[data-testid="stExpander"] button *,
[data-testid="stExpander"] summary *,
[data-testid="stExpander"] > div:first-child button > div > p{
  color:#111827 !important;
  -webkit-text-fill-color:#111827 !important;
}
/* Expander body content - always dark readable text */
[data-testid="stExpanderDetails"],
[data-testid="stExpanderDetails"] *:not(.tag-pill):not(.tag-relevant):not(.tag-partial):not(.tag-irrelevant){
  color:#111827 !important;
  -webkit-text-fill-color:#111827 !important;
}
[data-testid="stExpanderDetails"]{
  padding:20px 20px 22px;
}

.section-heading{
  margin:22px 0 14px !important;
  color:#111827 !important;
}

.stSelectbox label,.stTextArea label,[data-testid="stWidgetLabel"] *{
  color:#172033 !important;
  font-weight:650 !important;
}
.stSelectbox>div,.stTextArea>div{
  background:#fff !important;
  border-radius:11px !important;
}
.stSelectbox [data-baseweb="select"],
.stSelectbox [data-baseweb="select"] > div,
.stTextArea textarea{
  min-height:48px;
  background:#fff !important;
  border:1px solid rgba(31,41,55,.16) !important;
  border-radius:11px !important;
  color:#111827 !important;
  -webkit-text-fill-color:#111827 !important;
  box-shadow:0 1px 2px rgba(31,41,55,.04);
}
.stTextArea textarea{
  padding:13px 14px !important;
  line-height:1.55 !important;
}
.stSelectbox [data-baseweb="select"] *,
.stSelectbox [role="combobox"] *,
.stTextArea textarea{
  color:#111827 !important;
  -webkit-text-fill-color:#111827 !important;
}
.stTextArea textarea::placeholder{
  color:#6b7280 !important;
  -webkit-text-fill-color:#6b7280 !important;
}
.stSelectbox svg{
  color:#4b5563 !important;
}
.stSelectbox [data-baseweb="select"]:focus-within,
.stTextArea textarea:focus{
  border-color:#4f8ef7 !important;
  box-shadow:0 0 0 3px rgba(79,142,247,.16) !important;
}

.rec-launch{
  background:#fff !important;
  border:1px solid rgba(37,99,235,.16) !important;
  border-radius:16px !important;
  box-shadow:0 14px 36px rgba(31,41,55,.08);
}
.rec-btn-link{
  border-radius:11px !important;
  box-shadow:0 12px 26px rgba(37,99,235,.24) !important;
}
.rec-hint{
  color:#4b5563 !important;
}

button[kind="primary"],button[kind="secondary"]{
  min-height:46px;
}
button[kind="primary"]{
  box-shadow:0 10px 22px rgba(79,142,247,.25) !important;
}
button[kind="secondary"]{
  background:#fff !important;
  border:1px solid rgba(31,41,55,.14) !important;
}

.metric-card{
  background:#fff !important;
  border:1px solid rgba(31,41,55,.1) !important;
  border-radius:12px !important;
  box-shadow:0 10px 26px rgba(31,41,55,.07);
}
.metric-value{
  font-size:2.35rem !important;
}
.metric-label{
  color:#4b5563 !important;
}

.stTabs [role="tablist"]{
  background:#fff !important;
  box-shadow:0 8px 24px rgba(31,41,55,.06);
}
.stTabs [role="tab"]{
  color:#4b5563 !important;
}
.stTabs [role="tab"][aria-selected="true"]{
  color:#fff !important;
}

[data-testid="stAlert"]{
  border-radius:12px !important;
  border:1px solid rgba(31,41,55,.08) !important;
}

@media (max-width:760px){
  .block-container{padding-left:16px;padding-right:16px}
  .hero{padding:24px 20px;flex-direction:column}
  .hero-title{font-size:1.55rem !important}
  .hero-icon{font-size:2.1rem}
  .step-card{align-items:flex-start;flex-direction:column;gap:8px}
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────────────────────────────────────
def render_hero(title: str = "Session Analyser", subtitle: str = "Record, analyse, and improve with a full AI scorecard in seconds."):
    st.markdown(f"""<div class="hero">
      <div class="hero-copy">
        <div class="hero-kicker">Persona AI</div>
        <div class="hero-title">{title}</div>
        <div class="hero-sub">{subtitle}</div>
        <div class="hero-pills">
          <span class="hero-pill">Video</span>
          <span class="hero-pill">Audio</span>
          <span class="hero-pill">Text</span>
          <span class="hero-pill">Relevance</span>
        </div>
      </div>
      <div class="hero-icon">Persona AI</div>
    </div>
    """, unsafe_allow_html=True)


render_hero()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def _load_db_personas():
  """Load personas from the DB (cached for 10 minutes)."""
  return fetch_personas()


@st.cache_data(ttl=600, show_spinner=False)
def _load_db_activities(persona_id: int):
    return fetch_activities(persona_id)


@st.cache_data(ttl=600, show_spinner=False)
def _load_db_tasks(activity_id: int):
    return fetch_tasks(activity_id)


def score_cls(v: float) -> str:
  if v is None:
    return "score-na"
  if v >= 8:
    return "score-great"
  if v >= 6:
    return "score-good"
  if v >= 4:
    return "score-ok"
  return "score-poor"


def score_label(v: float) -> str:
  if v is None:
    return "Unavailable"
  if v >= 8:
    return "Strong"
  if v >= 6:
    return "Good"
  if v >= 4:
    return "Needs work"
  return "Low"


def metric_card(col, label, value):
  display_value = "N/A" if value is None else value
  col.markdown(
    f"""<div class="metric-card">
      <div class="metric-value {score_cls(value)}">{display_value}</div>
      <div class="metric-label">{label}</div></div>""",
    unsafe_allow_html=True,
  )


def render_step_card(step_label: str, title: str):
  st.markdown(
    f"""<div class="step-card">
      <div class="step-label">{step_label}</div>
      <div class="step-title"> {title}</div>
    </div>""",
    unsafe_allow_html=True,
  )


def _as_float(value, default=None):
  try:
    return float(value)
  except (TypeError, ValueError):
    return default


def build_audio_interpretation_lines(audio_metrics: dict, audio_signal: dict) -> list[str]:
  """Render audio coaching text from the current metric values."""
  lines = []

  wpm = _as_float(audio_metrics.get("wpm"))
  if wpm is not None:
    if wpm < 100:
      msg = f"Measured pace is {wpm:g} WPM, which is slow and may sound hesitant."
    elif wpm < 130:
      msg = f"Measured pace is {wpm:g} WPM, slightly slow but understandable."
    elif wpm <= 170:
      msg = f"Measured pace is {wpm:g} WPM, ideal for professional communication."
    elif wpm <= 190:
      msg = f"Measured pace is {wpm:g} WPM, slightly fast but still understandable."
    else:
      msg = f"Measured pace is {wpm:g} WPM, too fast and may affect clarity."
    lines.append(f"**WPM:** {msg}")

  filler_score = _as_float(audio_metrics.get("filler_score"))
  if filler_score is not None:
    filler_count = audio_metrics.get("filler_count")
    filler_pct = round(filler_score * 100, 1)
    count_text = f"{filler_count} filler pattern(s), " if filler_count is not None else ""
    if filler_score < 0.02:
      msg = f"Detected {count_text}{filler_pct}% of words, showing minimal hesitation."
    elif filler_score < 0.05:
      msg = f"Detected {count_text}{filler_pct}% of words, so hesitation is present but acceptable."
    elif filler_score < 0.1:
      msg = f"Detected {count_text}{filler_pct}% of words, which noticeably affects fluency."
    else:
      msg = f"Detected {count_text}{filler_pct}% of words, indicating high hesitation."
    lines.append(f"**Filler:** {msg}")

  pause_rate = _as_float(audio_metrics.get("pause_rate"))
  if pause_rate is not None:
    pause_pct = round(pause_rate * 100, 1)
    if pause_rate < 0.15:
      msg = f"Pause rate is {pause_pct}%, indicating smooth speech with minimal pauses."
    elif pause_rate < 0.3:
      msg = f"Pause rate is {pause_pct}%, so moderate pauses were observed."
    else:
      msg = f"Pause rate is {pause_pct}%, indicating frequent pauses and possible hesitation."
    lines.append(f"**Pause:** {msg}")

  energy = _as_float(audio_signal.get("voice_energy"))
  if energy is not None:
    if energy < 0.02:
      msg = f"Voice energy is {energy:.4f}, which is low and may sound weak or monotone."
    elif energy < 0.05:
      msg = f"Voice energy is {energy:.4f}, suggesting steady speech delivery."
    else:
      msg = f"Voice energy is {energy:.4f}, indicating enthusiastic speech."
    lines.append(f"**Energy:** {msg}")

  pitch = _as_float(audio_signal.get("pitch_variation"))
  if pitch is not None:
    if pitch < 20:
      msg = f"Pitch variation is {pitch:g}, so speech may sound monotone."
    elif pitch < 50:
      msg = f"Pitch variation is {pitch:g}, indicating natural speech."
    else:
      msg = f"Pitch variation is {pitch:g}, indicating expressive speech."
    lines.append(f"**Pitch:** {msg}")

  silence = _as_float(audio_signal.get("silence_ratio"))
  if silence is not None:
    silence_pct = round(silence * 100, 1)
    if silence < 0.15:
      msg = f"Silence ratio is {silence_pct}%, so speech flow is smooth with minimal pauses."
    elif silence < 0.30:
      msg = f"Silence ratio is {silence_pct}%, indicating moderate pauses."
    else:
      msg = f"Silence ratio is {silence_pct}%, indicating frequent pauses or hesitation."
    lines.append(f"**Silence:** {msg}")

  return lines


# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — Persona / Activity / Task
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("Step 0 — Persona / Activity / Task", expanded=True):
    st.markdown(
        "This section loads persona, activity, and task options from the database. "
        "Select a Persona first, then choose the related Activity and Task.",
    )
    st.markdown(
        '<div class="section-heading">Select your Persona, Activity and Task</div>',
        unsafe_allow_html=True,
    )
    db_error = None
    db_personas = []
    try:
        db_personas = _load_db_personas()
    except Exception as exc:
        db_error = str(exc)

    if db_error:
        st.error("Could not load persona data from database: " + db_error)

    if not db_personas:
        st.info("No persona options were found in the database.")
        selected_db_persona = None
    else:
        selected_db_persona = st.selectbox(
            "Persona",
            db_personas,
            format_func=lambda item: item["name"],
            key="db_persona",
            help="Choose a persona from the database.",
        )

    db_activities = []
    selected_db_activity = None
    if selected_db_persona:
        db_activities = _load_db_activities(selected_db_persona["id"])

    if db_activities:
        selected_db_activity = st.selectbox(
            "Activity",
            db_activities,
            format_func=lambda item: item["activity_name"],
            key="db_activity",
            help="Choose an activity related to the selected persona.",
        )
        if selected_db_activity and selected_db_activity.get("description"):
            st.caption(selected_db_activity["description"])
    else:
        st.info("No activities found for the selected persona.")

    db_tasks = []
    selected_db_task = None
    if selected_db_activity:
        db_tasks = _load_db_tasks(selected_db_activity["id"])

    if db_tasks:
        selected_db_task = st.selectbox(
            "Task",
            db_tasks,
            format_func=lambda item: item["task_name"],
            key="db_task",
            help="Choose a task related to the selected activity.",
        )
        if selected_db_task and selected_db_task.get("description"):
            st.caption(selected_db_task["description"])
    else:
        if selected_db_activity:
            st.info("No tasks found for the selected activity.")

    if selected_db_persona and selected_db_activity and selected_db_task:
        st.success(
            f"Selected scenario: {selected_db_persona['name']} → {selected_db_activity['activity_name']} "
            f"→ {selected_db_task['task_name']}"
        )
        st.session_state["selected_db_task"] = {
            "persona": selected_db_persona,
            "activity": selected_db_activity,
            "task": selected_db_task,
        }

    selected_performer = selected_db_persona["name"] if selected_db_persona else "Unknown Performer"
    selected_target = selected_db_activity["activity_name"] if selected_db_activity else "Unknown Target"


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Record
# ─────────────────────────────────────────────────────────────────────────────
render_step_card("Step 1", "Record Your Session")

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
# STEP 2 — Check status + Analyse
# ─────────────────────────────────────────────────────────────────────────────
render_step_card("Step 2", "Analyse Session")

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
        selected_scenario = {
            "persona": selected_db_persona,
            "activity": selected_db_activity,
            "task": selected_db_task,
        }

        results = run_batch_analysis(
            video_path=recording_path,
            prompt="",
            performer_role=selected_performer,
            target_role=selected_target,
            progress_fn=_prog,
            selected_scenario=selected_scenario,
        )
        prog.progress(1.0, text="Complete!")
        stxt.markdown("**Analysis complete!**")
        st.session_state["results"]       = results
        st.session_state["res_prompt"]    = "Scenario-based analysis"
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
    render_step_card("Step 4", "Your Session Scorecard")

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
    if scores.get("video_analysis_available") is False:
        st.info(
            "Video analysis was unavailable for this recording: "
            + (scores.get("video_analysis_status") or "no visual landmarks were detected.")
        )
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
                "Score": [
                    scores.get("visual_performance_score",0),
                    scores.get("audio_performance_score",0),
                    scores.get("text_performance_score",0),
                    scores.get("relevance_score",0),
                ],
            })
            chart = (alt.Chart(df).mark_bar(cornerRadiusTopLeft=6,cornerRadiusTopRight=6)
                .encode(
                    x=alt.X("Dimension:N", sort=None,
                            axis=alt.Axis(labelColor="#4b5563", labelFontSize=13, labelFontWeight="bold")),
                    y=alt.Y("Score:Q", scale=alt.Scale(domain=[0,10]),
                            axis=alt.Axis(labelColor="#8b949e")),
                    color=alt.Color("Score:Q", scale=alt.Scale(
                        domain=[0,5,8,10],
                        range=["#f43f5e","#f59e0b","#4f8ef7","#10b981"]), legend=None),
                    tooltip=["Dimension","Score"])
                .properties(height=240)
                .configure_view(strokeWidth=0, fill="#00000000")
                .configure_axis(grid=False, domain=False))
            st.altair_chart(chart, use_container_width=True)
        except Exception:
            for d,v in zip(["Video","Audio","Text","Relevance"],
                           [scores.get(k,0) for k in ["visual_performance_score","audio_performance_score",
                                                       "text_performance_score","relevance_score"]]):
                st.metric(d, f"{v}/10")

        with st.expander("Full Transcript", expanded=False):
            st.markdown(f"_{transcript or 'No transcript available.'}_")

        # Always categorise all 4 dimensions into strengths / needs work / gaps
        dim_scores = [
            ("Video body language",  "visual_performance_score"),
            ("Audio delivery",       "audio_performance_score"),
            ("Text quality",         "text_performance_score"),
            ("Prompt relevance",     "relevance_score"),
        ]
        strengths, mid, gaps = [], [], []
        for lbl, key in dim_scores:
            v = scores.get(key)
            if v is None:
                continue
            entry = f"{score_label(v)} {lbl} ({v}/10)"
            if v >= 7:
                strengths.append(entry)
            elif v < 5:
                gaps.append(entry)
            else:
                mid.append(entry)

        ga, gb = st.columns(2)
        with ga:
            st.markdown("#### Strengths")
            for s in (strengths or ["Keep practising to build strengths!"]):
                st.markdown(f"- {s}")
            if mid:
                st.markdown("#### Developing")
                for m in mid:
                    st.markdown(f"- {m}")
        with gb:
            st.markdown("#### Areas to Improve")
            for g in (gaps or ["No major gaps — great session!"]):
                st.markdown(f"- {g}")

    # ── Video ─────────────────────────────────────────────────────────────────
    with tab_vid:
        st.markdown("### Body Language Analysis")
        v1,v2,v3,v4 = st.columns(4)
        metric_card(v1,"Posture",       scores.get("posture_score",0))
        metric_card(v2,"Eye Contact",   scores.get("eye_contact_score",0))
        metric_card(v3,"Expression",    scores.get("expression_score",0))
        metric_card(v4,"Head Stability",scores.get("head_stability_score",0))

        if body_llm:
            overall_bl = body_llm.get("overall_body_language","")
            if overall_bl:
                st.markdown(f"<br>**Overall:** {overall_bl}", unsafe_allow_html=True)
            interp = body_llm.get("body_language_interpretation","")
            if interp:
                st.info(f"{interp}")

            # 3-column detailed analysis (always shown when data exists)
            ba, bb, bc = st.columns(3)
            posture_txt    = body_llm.get("posture_analysis","—")
            eyect_txt      = body_llm.get("eye_contact_analysis","—")
            expr_txt       = body_llm.get("expression_analysis","—")
            with ba:
                st.markdown("**Posture**")
                st.caption(posture_txt)
            with bb:
                st.markdown("**Eye Contact**")
                st.caption(eyect_txt)
            with bc:
                st.markdown("**Expression**")
                st.caption(expr_txt)

            sugg = body_llm.get("improvement_suggestions","")
            if sugg:
                st.warning(f"{sugg}")
        else:
            st.info("Body language interpretation unavailable for this recording.")

        st.markdown("#### Coaching Tips")
        for lbl, key, bad, good in [
            ("Posture",        "posture_score",        "Sit upright — shoulders back, spine tall.",          "Great posture!"),
            ("Eye Contact",    "eye_contact_score",    "Look at the camera lens, not the screen.",           "Excellent gaze!"),
            ("Expression",     "expression_score",     "Vary expressions — appear engaged and warm.",        "Good facial engagement!"),
            ("Head Stability", "head_stability_score", "Minimise excessive head movement.",                  "Steady and composed!"),
        ]:
            v = scores.get(key, 5)
            if v is None:
                st.info(f"{lbl}: visual analysis unavailable for this recording.")
            else:
                (st.warning if v < 6 else st.success)(f"{lbl}: {bad if v < 6 else good}")

    # ── Audio ─────────────────────────────────────────────────────────────────
    with tab_aud:
        st.markdown("### Audio & Speech Analysis")
        a1,a2,a3 = st.columns(3)
        metric_card(a1,"Clarity",       scores.get("speech_clarity_score",0))
        metric_card(a2,"Confidence",    scores.get("confidence_score",0))
        metric_card(a3,"Communication", scores.get("communication_score",0))

        st.markdown("<br>#### Speech Signal Metrics", unsafe_allow_html=True)
        sm1,sm2,sm3,sm4 = st.columns(4)
        sm1.metric("WPM",           audio_metrics.get("wpm","—"))
        sm2.metric("Filler Score",  audio_metrics.get("filler_score","—"))
        sm3.metric("Pause Rate",    audio_metrics.get("pause_rate","—"))
        sm4.metric("Silence Ratio", audio_signal.get("silence_ratio","—"))

        # Interpretation details always shown (no expander)
        interp_lines = build_audio_interpretation_lines(audio_metrics, audio_signal)
        for k, l in [("wpm_interpretation","WPM"), ("filler_interpretation","Filler"), ("pause_interpretation","Pause")]:
            v = audio_metrics.get(k,"")
            if v and v != "—":
                interp_lines.append(f"**{l}:** {v}")
        for k, l in [("energy_interpretation","Energy"), ("pitch_interpretation","Pitch"), ("silence_interpretation","Silence")]:
            v = audio_signal.get(k,"")
            if v and v != "—":
                interp_lines.append(f"**{l}:** {v}")
        interp_lines = interp_lines[:6]
        if interp_lines:
            with st.expander("Interpretation Details", expanded=True):
                for line in interp_lines:
                    st.markdown(f"- {line}")

        if audio_llm:
          st.markdown("#### LLM Feedback")
          for key, label in [
            ("tone_quality",          "Tone"),
            ("voice_quality_feedback","Voice"),
            ("engagement_level",      "Engagement"),
            ("communication_style",   "Style"),
            ("language_fluency",      "Fluency"),
            ("professionalism_level", "Professionalism"),
            ("filler_word_usage",     "Fillers"),
          ]:
            v = audio_llm.get(key,"")
            if v:
              st.markdown(f"**{label}:** {v}")
        else:
          st.info("No spoken content detected; audio LLM feedback is unavailable.")

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
          t_sc = scores.get("text_performance_score", 5)
          if t_sc is None:
            t_sc = 5
          try:
            t_sc = float(t_sc)
          except (TypeError, ValueError):
            t_sc = 5
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

        tag_map = {
            "relevant":          '<span class="tag-pill tag-relevant">✓ Relevant</span>',
            "partially_relevant":'<span class="tag-pill tag-partial">◑ Partially Relevant</span>',
            "irrelevant":        '<span class="tag-pill tag-irrelevant">✗ Irrelevant</span>',
        }
        # Fallback: wrap any unknown label in a styled pill too
        rel_display = tag_map.get(
            rel_label,
            f'<span class="tag-pill tag-irrelevant">{rel_label.replace("_"," ").title() if rel_label else "Unknown"}</span>'
        )
        st.markdown(f"**Status:** {rel_display}", unsafe_allow_html=True)

        with st.expander("Prompt evaluated against", expanded=False):
            st.info(enh_prompt)

        r_sc = scores.get("relevance_score", 5)
        if r_sc is None:
          r_sc = 5
        try:
          r_sc = float(r_sc)
        except (TypeError, ValueError):
          r_sc = 5
        if rel_reason:
          (st.success if r_sc>=8 else st.info if r_sc>=5 else st.warning)(f"**Evaluator:** {rel_reason}")

        # Deductions always visible (expanded) when score is low
        if rel_deducts:
            with st.expander("Deductions Applied", expanded=True):
                for d in rel_deducts:
                    st.markdown(f"- {d}")

        st.markdown("#### How to Improve Relevance")
        t_role = persona_framework.get_role(st.session_state.get("res_target",""))
        if r_sc >= 8:
            st.success("Highly relevant — keep anchoring to your audience's specific needs.")
        else:
            tips = []
            if r_sc < 4:
                tips += [
                    "**Re-read the prompt** — identify the core question before answering.",
                    "**Open with a direct statement** addressing what was asked.",
                    "**Avoid long preambles** — make your point within 2 sentences.",
                ]
            elif r_sc < 6:
                tips += [
                    "**Stay closer to the scenario** described in the prompt.",
                    "**Reduce tangents** — consciously redirect back to the topic.",
                ]
            else:
                tips += ["**Add specifics** — concrete examples or numbers."]

            if t_role:
                exps = t_role.requirements.get("expectations",[])
                if exps:
                    tips.append(f"**Speak to *{t_role.name}*'s expectations:**")
                    tips += [f"  - {e}" for e in exps]

            tips.append("**Use PREP:** Point → Reason → Example → Point (restate).")
            for tip in tips:
                st.markdown(f"- {tip}")

                  
