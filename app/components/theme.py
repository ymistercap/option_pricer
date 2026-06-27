"""Theme — starfield, CSS, Plotly layout, topbar."""

import streamlit as st
import streamlit.components.v1 as components

# ---------------------------------------------------------------------------
# Plotly layout helper
# ---------------------------------------------------------------------------

_PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="'JetBrains Mono', monospace", size=10, color="#8896a8"),
    margin=dict(t=24, r=16, b=40, l=52),
    xaxis=dict(gridcolor="#2a3040", zerolinecolor="#2a3040",
               tickfont=dict(size=9, color="#556070"), linecolor="#2a3040"),
    yaxis=dict(gridcolor="#2a3040", zerolinecolor="#2a3040",
               tickfont=dict(size=9, color="#556070"), linecolor="#2a3040"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)",
                font=dict(size=9, color="#8896a8")),
    hoverlabel=dict(bgcolor="#111418", bordercolor="#f7b731",
                    font=dict(family="'JetBrains Mono', monospace", size=10)),
    colorway=["#f7b731", "#00d4aa", "#4f8ef7", "#c084fc", "#e74c3c", "#2ecc71"],
)


def plotly_layout(**overrides):
    """Return a deep-merged Plotly layout dict."""
    out = {}
    for k, v in _PLOTLY_BASE.items():
        if isinstance(v, dict) and isinstance(overrides.get(k), dict):
            out[k] = {**v, **overrides.pop(k)}
        else:
            out[k] = v
    out.update(overrides)
    return out


# ---------------------------------------------------------------------------
# Starfield canvas — runs inside an iframe via components.html().
# Uses window.frameElement to forcefully position its own container
# as a fixed fullscreen backdrop (bypasses fragile CSS selectors).
# ---------------------------------------------------------------------------

_STARFIELD = r"""<!DOCTYPE html><html><head>
<style>*{margin:0;padding:0}html,body{overflow:hidden;background:#04060a}</style>
</head><body><canvas id="c"></canvas><script>
// Self-position: make iframe + parent fixed fullscreen
try{
  var fr=window.frameElement;
  if(fr){
    fr.style.cssText='position:fixed!important;top:0!important;left:0!important;width:100vw!important;height:100vh!important;z-index:0!important;pointer-events:none!important;border:none!important;';
    var pw=fr.parentElement;
    if(pw){pw.style.cssText='position:fixed!important;top:0!important;left:0!important;width:100vw!important;height:100vh!important;z-index:0!important;pointer-events:none!important;overflow:visible!important;';}
    // Walk up to clear any overflow:hidden ancestors (Streamlit nests deeply)
    var el=pw;
    for(var i=0;i<8&&el;i++){
      if(el.style){el.style.overflow='visible';}
      el=el.parentElement;
    }
  }
}catch(e){}

var c=document.getElementById("c"),x=c.getContext("2d"),W,H,f=0;
function rz(){W=innerWidth;H=innerHeight;c.width=W;c.height=H}
rz();addEventListener("resize",rz);

// Stellar color palette
var C=[[155,176,255],[202,215,255],[248,247,255],[255,244,234],[255,244,234],[255,244,234],[255,210,161],[255,204,111],[255,178,130]];

function ml(n,s0,s1,a0,a1,p){
  for(var o=[],i=0;i<n;i++){
    var cl=C[Math.random()*C.length|0];
    o.push({x:Math.random(),y:Math.random(),s:s0+Math.random()*(s1-s0),
      a:a0+Math.random()*(a1-a0),c:cl,ts:.005+Math.random()*.025,
      tp:Math.random()*6.28,ta:.15+Math.random()*.45,
      d:p*(.6+Math.random()*.8),p:p});
  }return o;
}

var L1=ml(450,.4,.9,.25,.55,1),L2=ml(180,.7,1.4,.45,.8,2.2),L3=ml(45,1.2,2.4,.7,1,4);

// Bright stars with diffraction spikes
var B=[];for(var i=0;i<8;i++){var cl=C[Math.random()*C.length|0];
  B.push({x:Math.random(),y:Math.random(),s:1.6+Math.random()*1.2,c:cl,tp:Math.random()*6.28,ts:.01+Math.random()*.02});}

// Nebulae
var N=[{x:.15,y:.25,r:380,h:220,a:.04},{x:.78,y:.65,r:460,h:280,a:.035},{x:.45,y:.85,r:320,h:200,a:.03}];

// Shooting star
var sh=null;
function ss(){var l=Math.random()<.5;sh={x:l?-50:W+50,y:Math.random()*H*.6,vx:(l?1:-1)*(8+Math.random()*4),vy:2+Math.random()*3,li:0,ml:60+Math.random()*30}}

function dl(S){
  for(var i=0;i<S.length;i++){
    var s=S[i],dr=(f*.00006*s.d)%1,
        xx=((s.x+dr)%1)*W,yy=s.y*H;
    if(xx<-5)xx+=W+10;if(xx>W+5)xx-=W+10;
    var tw=1+s.ta*Math.sin(f*s.ts+s.tp),
        al=Math.max(0,Math.min(1,s.a*tw));
    x.fillStyle="rgba("+s.c[0]+","+s.c[1]+","+s.c[2]+","+al+")";
    x.beginPath();x.arc(xx,yy,s.s,0,6.28);x.fill();
  }
}

function draw(){
  x.fillStyle="#04060a";x.fillRect(0,0,W,H);

  // Nebulae
  for(var i=0;i<N.length;i++){
    var n=N[i],px=n.x*W,py=n.y*H,
        g=x.createRadialGradient(px,py,0,px,py,n.r);
    g.addColorStop(0,"hsla("+n.h+",60%,55%,"+n.a+")");
    g.addColorStop(.6,"hsla("+n.h+",60%,45%,"+(n.a*.4)+")");
    g.addColorStop(1,"hsla("+n.h+",60%,40%,0)");
    x.fillStyle=g;x.fillRect(0,0,W,H);
  }

  // Galactic band hint
  var bg=x.createLinearGradient(0,H*.2,W,H*.8);
  bg.addColorStop(0,"rgba(140,160,200,0)");
  bg.addColorStop(.5,"rgba(180,180,210,.025)");
  bg.addColorStop(1,"rgba(140,160,200,0)");
  x.fillStyle=bg;x.fillRect(0,0,W,H);

  // Star layers
  dl(L1);dl(L2);dl(L3);

  // Bright stars with diffraction spikes
  for(var i=0;i<B.length;i++){
    var s=B[i],xx=s.x*W,yy=s.y*H,
        tw=.75+.25*Math.sin(f*s.ts+s.tp),
        r=s.c[0],g=s.c[1],b=s.c[2];
    var gw=x.createRadialGradient(xx,yy,0,xx,yy,s.s*14);
    gw.addColorStop(0,"rgba("+r+","+g+","+b+","+(0.35*tw)+")");
    gw.addColorStop(.4,"rgba("+r+","+g+","+b+","+(0.08*tw)+")");
    gw.addColorStop(1,"rgba("+r+","+g+","+b+",0)");
    x.fillStyle=gw;x.fillRect(xx-s.s*14,yy-s.s*14,s.s*28,s.s*28);
    // Spikes
    x.strokeStyle="rgba("+r+","+g+","+b+","+(0.5*tw)+")";
    x.lineWidth=.6;var sL=s.s*9*tw;
    x.beginPath();x.moveTo(xx-sL,yy);x.lineTo(xx+sL,yy);
    x.moveTo(xx,yy-sL);x.lineTo(xx,yy+sL);x.stroke();
    // Core
    x.fillStyle="rgba("+r+","+g+","+b+","+tw+")";
    x.beginPath();x.arc(xx,yy,s.s,0,6.28);x.fill();
  }

  // Shooting star
  if(!sh&&Math.random()<.0015)ss();
  if(sh){
    sh.x+=sh.vx;sh.y+=sh.vy;sh.li++;
    var t=sh.li/sh.ml,al=t<.2?t/.2:1-(t-.2)/.8;
    var tx=sh.x-sh.vx*8.75,ty=sh.y-sh.vy*8.75,
        tg=x.createLinearGradient(tx,ty,sh.x,sh.y);
    tg.addColorStop(0,"rgba(255,255,255,0)");
    tg.addColorStop(1,"rgba(255,250,230,"+(al*.85)+")");
    x.strokeStyle=tg;x.lineWidth=1.4;x.lineCap="round";
    x.beginPath();x.moveTo(tx,ty);x.lineTo(sh.x,sh.y);x.stroke();
    x.fillStyle="rgba(255,250,230,"+al+")";
    x.beginPath();x.arc(sh.x,sh.y,1.4,0,6.28);x.fill();
    if(sh.li>=sh.ml)sh=null;
  }

  // Vignette
  var v=x.createRadialGradient(W/2,H/2,H*.35,W/2,H/2,H*.95);
  v.addColorStop(0,"rgba(4,6,10,0)");
  v.addColorStop(1,"rgba(4,6,10,.55)");
  x.fillStyle=v;x.fillRect(0,0,W,H);

  f++;requestAnimationFrame(draw);
}
draw();
</script></body></html>"""


# ---------------------------------------------------------------------------
# CSS — Streamlit DOM targeting
# ---------------------------------------------------------------------------

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&family=Inter:wght@300;400;500;600;700&display=swap');

/* ========== ROOT / APP BACKGROUND ========== */
html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background-color: #0a0c0f !important;
    background: #0a0c0f !important;
    color: #e8ecf0 !important;
}
.main .block-container,
[data-testid="stMainBlockContainer"] {
    background: transparent !important;
    position: relative !important;
    z-index: 2 !important;
}
[data-testid="stMain"] {
    background: transparent !important;
}
[data-testid="stBottomBlockContainer"] {
    background: transparent !important;
}
[data-testid="stVerticalBlock"] {
    background: transparent !important;
}

/* ========== HIDE STREAMLIT CHROME ========== */
header[data-testid="stHeader"] {
    background: transparent !important;
    height: 0 !important;
    min-height: 0 !important;
    visibility: hidden !important;
}
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
#MainMenu { visibility: hidden !important; }
footer { visibility: hidden !important; }
[data-testid="manage-app-button"] { display: none !important; }

/* ========== SCANLINES OVERLAY ========== */
.stApp::after {
    content: '';
    pointer-events: none;
    position: fixed;
    inset: 0;
    z-index: 999;
    background: repeating-linear-gradient(
        0deg, transparent, transparent 2px,
        rgba(0,0,0,0.03) 2px, rgba(0,0,0,0.03) 4px
    );
}

/* ========== SIDEBAR ========== */
section[data-testid="stSidebar"] {
    background: #111418 !important;
    border-right: 1px solid #2a3040 !important;
    z-index: 10 !important;
    width: 260px !important;
}
section[data-testid="stSidebar"] > div,
[data-testid="stSidebarContent"],
[data-testid="stSidebarUserContent"] {
    background: transparent !important;
}
/* Amber accent line on left edge */
section[data-testid="stSidebar"]::after {
    content: '';
    position: absolute;
    top: 0; left: 0; bottom: 0; width: 2px;
    background: linear-gradient(180deg, transparent, #f7b731, transparent);
    opacity: 0.4;
    pointer-events: none;
}
/* Sidebar headings */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #f7b731 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
    font-size: 12px !important;
}
/* Sidebar text and labels */
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
    color: #556070 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 9px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}
/* Sidebar inputs */
section[data-testid="stSidebar"] input[type="number"],
section[data-testid="stSidebar"] .stNumberInput input,
section[data-testid="stSidebar"] [data-baseweb="input"] input {
    background: #181c22 !important;
    border: 1px solid #2a3040 !important;
    color: #f7b731 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    border-radius: 3px !important;
}
section[data-testid="stSidebar"] input:focus {
    border-color: #f7b731 !important;
    box-shadow: 0 0 0 2px rgba(247,183,49,0.15) !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #181c22 !important;
    border: 1px solid #2a3040 !important;
    color: #f7b731 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    border-radius: 3px !important;
}

/* SIDEBAR RADIO → NAVIGATION ITEMS */
section[data-testid="stSidebar"] .stRadio > div {
    gap: 0 !important;
}
section[data-testid="stSidebar"] .stRadio [role="radiogroup"] {
    gap: 0 !important;
}
section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    color: #8896a8 !important;
    padding: 8px 16px !important;
    border-left: 2px solid transparent !important;
    border-radius: 0 !important;
    transition: all 0.15s !important;
    background: transparent !important;
    margin: 0 !important;
    text-transform: none !important;
    letter-spacing: normal !important;
}
section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label:hover {
    color: #e8ecf0 !important;
    background: #181c22 !important;
}
/* Hide radio circles */
section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label div[data-testid="stMarkdownContainer"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
}
section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label div[data-testid="stMarkdownContainer"] p {
    text-transform: none !important;
    letter-spacing: normal !important;
    font-size: 11px !important;
    color: inherit !important;
}
/* Active nav item */
section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label[data-checked="true"],
section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label[data-selected="true"] {
    background: rgba(247,183,49,0.12) !important;
    color: #f7b731 !important;
    border-left-color: #f7b731 !important;
}
section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label[data-checked="true"] p,
section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label[data-selected="true"] p {
    color: #f7b731 !important;
}

/* Sidebar horizontal rule */
section[data-testid="stSidebar"] hr {
    border-color: #2a3040 !important;
    opacity: 0.5 !important;
}

/* SIDEBAR SLIDER */
section[data-testid="stSidebar"] [data-baseweb="slider"] {
    padding-top: 0 !important;
    padding-bottom: 4px !important;
}
section[data-testid="stSidebar"] [data-testid="stThumbValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    color: #f7b731 !important;
    font-size: 10px !important;
}

/* Sidebar radio for option type (horizontal) */
section[data-testid="stSidebar"] .stRadio[data-testid="stRadio"] > div[role="radiogroup"][aria-label="Option Type"] label {
    flex: 1 !important;
    text-align: center !important;
    justify-content: center !important;
    padding: 4px 0 !important;
    border-radius: 3px !important;
    border-left: none !important;
    border: 1px solid #2a3040 !important;
    text-transform: uppercase !important;
    font-size: 10px !important;
    letter-spacing: 0.06em !important;
}
section[data-testid="stSidebar"] .stRadio[data-testid="stRadio"] > div[role="radiogroup"][aria-label="Option Type"] label[data-checked="true"] {
    background: #f7b731 !important;
    color: #000 !important;
    border-color: #f7b731 !important;
}
section[data-testid="stSidebar"] .stRadio[data-testid="stRadio"] > div[role="radiogroup"][aria-label="Option Type"] label[data-checked="true"] p {
    color: #000 !important;
}

/* ========== MAIN CONTENT HEADINGS ========== */
.main h1, .main h2, .main h3,
[data-testid="stHeading"],
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    color: #e8ecf0 !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    font-weight: 600 !important;
}
.stMarkdown p, .stMarkdown li {
    color: #e8ecf0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
}

/* ========== METRICS — terminal-style cards ========== */
[data-testid="stMetric"],
[data-testid="stMetricValue"],
[data-testid="stMetricLabel"],
[data-testid="stMetricDelta"] {
    background: transparent !important;
}
[data-testid="stMetric"] {
    background: #181c22 !important;
    border: 1px solid #2a3040 !important;
    border-radius: 4px !important;
    padding: 14px 18px !important;
    overflow: hidden !important;
    position: relative !important;
}
/* Amber top accent line on metric cards */
[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #f7b731, transparent);
    opacity: 0.6;
}
[data-testid="stMetricLabel"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 9px !important;
    color: #556070 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] div {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 9px !important;
    color: #556070 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 22px !important;
    color: #f7b731 !important;
    font-weight: 500 !important;
}
[data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important;
}

/* ========== BUTTONS ========== */
.stButton > button,
button[kind="primary"],
[data-testid="stBaseButton-primary"],
[data-testid="baseButton-primary"] {
    background: #f7b731 !important;
    color: #000 !important;
    border: none !important;
    border-radius: 3px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    padding: 8px 20px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    transition: all 0.15s !important;
}
.stButton > button:hover,
button[kind="primary"]:hover {
    background: #ffd166 !important;
    color: #000 !important;
    border: none !important;
    box-shadow: 0 0 12px rgba(247,183,49,0.35) !important;
}
.stButton > button:active { transform: scale(0.98) !important; }
.stButton > button:focus {
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(247,183,49,0.3) !important;
}
/* Secondary/ghost buttons */
[data-testid="stBaseButton-secondary"] {
    background: transparent !important;
    color: #f7b731 !important;
    border: 1px solid #f7b731 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
}
[data-testid="stBaseButton-secondary"]:hover {
    background: rgba(247,183,49,0.12) !important;
}

/* ========== EXPANDERS ========== */
[data-testid="stExpander"] {
    background: #181c22 !important;
    border: 1px solid #2a3040 !important;
    border-radius: 3px !important;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] [data-testid="stExpanderToggleDetails"],
details summary {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important;
    color: #556070 !important;
    background: transparent !important;
}
details summary:hover { color: #8896a8 !important; }
[data-testid="stExpanderDetails"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    color: #8896a8 !important;
    line-height: 1.8 !important;
    background: transparent !important;
}
[data-testid="stExpanderDetails"] p {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    color: #8896a8 !important;
}

/* ========== TABS ========== */
.stTabs [data-baseweb="tab-list"],
[role="tablist"] {
    border-bottom: 1px solid #2a3040 !important;
    gap: 0 !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab"],
[role="tab"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    color: #556070 !important;
    padding: 8px 18px !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab"]:hover,
[role="tab"]:hover { color: #8896a8 !important; }
.stTabs [aria-selected="true"],
[role="tab"][aria-selected="true"] {
    color: #f7b731 !important;
    border-bottom-color: #f7b731 !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-highlight"] { background-color: #f7b731 !important; }
.stTabs [data-baseweb="tab-border"] { background-color: #2a3040 !important; }
[data-baseweb="tab-panel"],
[role="tabpanel"] { background: transparent !important; }

/* ========== INPUTS ========== */
.stSelectbox > div > div,
[data-baseweb="select"] > div,
.stTextInput input,
.stNumberInput input,
[data-baseweb="input"] input {
    background: #181c22 !important;
    border: 1px solid #2a3040 !important;
    color: #e8ecf0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    border-radius: 3px !important;
    font-size: 12px !important;
}
.stTextInput input:focus,
.stNumberInput input:focus {
    border-color: #f7b731 !important;
    box-shadow: 0 0 0 2px rgba(247,183,49,0.15) !important;
}
/* Dropdown menus */
[data-baseweb="popover"],
[data-baseweb="menu"] {
    background: #181c22 !important;
    border: 1px solid #2a3040 !important;
}
[data-baseweb="menu"] li { color: #e8ecf0 !important; }
[data-baseweb="menu"] li:hover { background: #1e2330 !important; }
/* Number input stepper buttons */
.stNumberInput button {
    background: #1e2330 !important;
    color: #8896a8 !important;
    border: 1px solid #2a3040 !important;
}
.stNumberInput button:hover {
    background: #2a3040 !important;
    color: #f7b731 !important;
}

/* ========== SLIDER ========== */
[data-baseweb="slider"] [role="slider"] {
    background: #f7b731 !important;
    border-color: #f7b731 !important;
    box-shadow: 0 0 6px rgba(247,183,49,0.35) !important;
    width: 14px !important;
    height: 14px !important;
}
[data-baseweb="slider"] [data-testid="stTickBar"] {
    background: #2a3040 !important;
}
/* Slider track */
[data-baseweb="slider"] > div > div > div[role="slider"] ~ div {
    background: #344055 !important;
}

/* ========== PLOTLY CHARTS ========== */
[data-testid="stPlotlyChart"],
.stPlotlyChart {
    background: #181c22 !important;
    border: 1px solid #2a3040 !important;
    border-radius: 4px !important;
    overflow: hidden !important;
    padding: 4px !important;
}

/* ========== DATAFRAMES / TABLES ========== */
[data-testid="stDataFrame"],
[data-testid="stTable"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
}
[data-testid="stDataFrame"] [data-testid="glideDataEditor"] {
    border: 1px solid #2a3040 !important;
    border-radius: 4px !important;
}

/* ========== CAPTION ========== */
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 9px !important;
    color: #556070 !important;
}

/* ========== ALERTS ========== */
[data-testid="stAlert"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    border-radius: 4px !important;
    border: 1px solid #2a3040 !important;
}

/* ========== SPINNER ========== */
.stSpinner > div > div { border-top-color: #f7b731 !important; }

/* ========== PROGRESS BAR ========== */
[data-testid="stProgress"] > div > div > div { background: #f7b731 !important; }

/* ========== WIDGET LABELS ========== */
[data-testid="stWidgetLabel"] p {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    color: #8896a8 !important;
}

/* ========== HORIZONTAL RULE ========== */
hr { border-color: #2a3040 !important; }

/* ========== SCROLLBAR ========== */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #111418; }
::-webkit-scrollbar-thumb { background: #344055; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #f7b731; }

/* ========== SELECT SLIDER ========== */
.stSelectSlider [data-baseweb="slider"] [role="slider"] {
    background: #f7b731 !important;
}

/* ========== TOPBAR ========== */
.topbar {
    background: rgba(17,20,24,0.92);
    border-bottom: 1px solid #2a3040;
    height: 44px;
    display: flex;
    align-items: center;
    padding: 0 24px;
    gap: 16px;
    margin: -1rem -1rem 1.5rem -1rem;
    position: relative;
    z-index: 5;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
}
.topbar-page {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #e8ecf0;
    font-weight: 500;
}
.topbar-badge {
    background: rgba(247,183,49,0.15);
    border: 1px solid #f7b731;
    color: #f7b731;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    padding: 2px 8px;
    border-radius: 2px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.topbar-spacer { flex: 1; }
.topbar-status {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #556070;
    display: flex;
    align-items: center;
    gap: 6px;
}
.status-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #2ecc71;
    box-shadow: 0 0 6px #2ecc71;
    animation: pulse 2s infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

/* ========== CHIP ========== */
.chip {
    display: inline-flex; align-items: center;
    padding: 2px 8px; border-radius: 2px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; letter-spacing: 0.06em;
    text-transform: uppercase; font-weight: 600;
}
.chip-green { background: rgba(46,204,113,0.15); color: #2ecc71; border: 1px solid rgba(46,204,113,0.3); }
.chip-red { background: rgba(231,76,60,0.15); color: #e74c3c; border: 1px solid rgba(231,76,60,0.3); }
.chip-amber { background: rgba(247,183,49,0.15); color: #f7b731; border: 1px solid rgba(247,183,49,0.3); }
.chip-blue { background: rgba(79,142,247,0.15); color: #4f8ef7; border: 1px solid rgba(79,142,247,0.3); }

/* ========== COLUMN GAP ========== */
[data-testid="stHorizontalBlock"] {
    gap: 16px !important;
}

/* ========== HIDE RADIO CIRCLES (sidebar nav + option type) ========== */
section[data-testid="stSidebar"] [data-baseweb="radio"] > div:first-child {
    display: none !important;
}

/* ========== CARD TITLE (amber square prefix) ========== */
.card-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #8896a8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.card-title::before {
    content: '';
    width: 8px;
    height: 8px;
    background: #f7b731;
    border-radius: 1px;
    flex-shrink: 0;
}

/* ========== SECTION TITLE (border-bottom) ========== */
.section-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #556070;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 12px;
    margin-top: 24px;
    padding-bottom: 6px;
    border-bottom: 1px solid #2a3040;
}

/* ========== STYLED TABLES ========== */
.tbl { width: 100%; border-collapse: collapse; font-family: 'JetBrains Mono', monospace; font-size: 11px; }
.tbl th { color: #556070; text-transform: uppercase; font-size: 9px; letter-spacing: 0.1em; padding: 8px 12px; border-bottom: 1px solid #2a3040; text-align: left; font-weight: 400; }
.tbl td { padding: 8px 12px; border-bottom: 1px solid #344055; color: #8896a8; }
.tbl tr:last-child td { border-bottom: none; }
.tbl .tbl-hi { color: #f7b731; }
.tbl .tbl-pos { color: #2ecc71; }
.tbl .tbl-neg { color: #e74c3c; }

/* ========== PARAM SUMMARY (inline after button) ========== */
.param-summary {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #556070;
    display: flex;
    align-items: center;
    height: 100%;
    padding-top: 6px;
}
</style>
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def inject_theme():
    """Inject starfield background and CSS theme. Call once at app startup."""
    components.html(_STARFIELD, height=0, scrolling=False)
    st.markdown(_CSS, unsafe_allow_html=True)


def card_title(text: str) -> None:
    """Render a card title with amber square prefix."""
    st.markdown(f'<div class="card-title">{text}</div>', unsafe_allow_html=True)


def section_title(text: str) -> None:
    """Render a section title with bottom border."""
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


def topbar_html(page_name: str, page_num: str, params: dict = None) -> str:
    """Return HTML for the top bar."""
    params_str = ""
    if params:
        params_str = (
            f'<span class="topbar-status" style="margin-left:16px">'
            f'S= {params.get("S", 100)} · K= {params.get("K", 100)} · '
            f'T= {params.get("T", 1.0)}y · σ= {params.get("sigma", 0.2)}'
            f'</span>'
        )
    return (
        '<div class="topbar">'
        f'<span class="topbar-page">{page_name}</span>'
        f'<span class="topbar-badge">{page_num} / 09</span>'
        '<span class="topbar-spacer"></span>'
        '<span class="topbar-status">'
        '<span class="status-dot"></span>Engine Active</span>'
        f'{params_str}'
        '</div>'
    )
