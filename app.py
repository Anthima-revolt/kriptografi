import io
import matplotlib.pyplot as plt

st.markdown("""
<style>

/* ====== GLOBAL ====== */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ===== BACKGROUND ANIMATION CONTAINER ===== */
body {
    background: radial-gradient(circle at top, #0B0F14, #020617);
    overflow-x: hidden;
}

/* ===== FLOATING PARTICLES ===== */
body::after {
    content: "";
    position: fixed;
    inset: 0;
    background-image:
        radial-gradient(#00E67655 1px, transparent 1px),
        radial-gradient(#00E67622 1px, transparent 1px);
    background-size: 120px 120px, 60px 60px;
    animation: floatParticles 40s linear infinite;
    pointer-events: none;
    z-index: 0;
}

/* ===== SCANLINE OVERLAY (ANIMATED) ===== */
body::before {
    content: "";
    position: fixed;
    inset: 0;
    background: repeating-linear-gradient(
        to bottom,
        rgba(0, 230, 118, 0.04),
        rgba(0, 230, 118, 0.04) 1px,
        transparent 1px,
        transparent 5px
    );
    animation: scanMove 8s linear infinite;
    pointer-events: none;
    z-index: 1;
}


/* ===== ANIMATIONS ===== */
@keyframes floatParticles {
    from {
        background-position: 0 0, 0 0;
    }
    to {
        background-position: 600px 1200px, -600px 600px;
    }
}

@keyframes scanMove {
    from {
        background-position: 0 0;
    }
    to {
        background-position: 0 200px;
    }
}

/* ===== FOREGROUND FIX ===== */
section[data-testid="stAppViewContainer"] {
    position: relative;
    z-index: 2;
}

/* ====== SIDEBAR ====== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B0F14, #020617);
    border-right: 1px solid #00E67633;
}

/* ====== CARD EFFECT ====== */
div[data-testid="stVerticalBlock"] > div {
    background: #0F172A;
    border-radius: 12px;
    padding: 1.2rem;
    border: 1px solid #00E67622;
    box-shadow: 0 0 20px rgba(0, 230, 118, 0.05);
}

/* ====== GLOW BUTTON ====== */
button {
    background: linear-gradient(90deg, #00E676, #00C853) !important;
    color: black !important;
    border-radius: 10px !important;
    font-weight: 600;
    box-shadow: 0 0 15px rgba(0, 230, 118, 0.4);
}

button:hover {
    box-shadow: 0 0 25px rgba(0, 230, 118, 0.8);
    transform: scale(1.03);
}

/* ====== INPUT ====== */
input, textarea {
    background-color: #020617 !important;
    color: #00E676 !important;
    border: 1px solid #00E67644 !important;
}

/* ====== MATRIX SCANLINE EFFECT ====== */
body::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: repeating-linear-gradient(
        to bottom,
        rgba(0, 230, 118, 0.03),
        rgba(0, 230, 118, 0.03) 1px,
        transparent 1px,
        transparent 4px
    );
    pointer-events: none;
    z-index: 9999;
}

/* ==============================
   CYBER MOVING LINES (BACKGROUND)
   ============================== */
body .cyber-lines {
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    background:
        linear-gradient(
            180deg,
            transparent 0%,
            rgba(0, 230, 118, 0.15) 50%,
            transparent 100%
    );
    background-size: 300px 300%;
    animation: cyberLinesMove 18s linear infinite;
    opacity: 0.6;
}

/* ==============================
   LINE ANIMATION
   ============================== */
@keyframes cyberLinesMoveVertical {
    from { background-position: 0 -300px; }
    to   { background-position: 0 1200px; }
}

/* ==============================
   HUD SCANNING LINE
   ============================== */
body .hud-scan {
    position: fixed;
    left: 0;
    top: -20%;
    width: 100%;
    height: 120px;
    pointer-events: none;
    z-index: 1;
    background: linear-gradient(
        to bottom,
        transparent,
        rgba(0, 230, 118, 0.15),
        transparent
    );
    animation: hudScanMove 7s ease-in-out infinite;
}

/* Animation */
@keyframes hudScanMove {
    0% {
        top: -20%;
        opacity: 0;
    }
    30% {
        opacity: 0.6;
    }
    50% {
        opacity: 0.9;
    }
    70% {
        opacity: 0.6;
    }
    100% {
        top: 120%;
        opacity: 0;
    }
}

/* ==============================
   DIGITAL ZIG-ZAG LINES
   ============================== */
body .zigzag-lines {
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    background:
        repeating-linear-gradient(
            135deg,
            rgba(0, 230, 118, 0.08) 0px,
            rgba(0, 230, 118, 0.08) 1px,
            transparent 1px,
            transparent 14px
        );
    animation: zigzagMove 30s linear infinite;
    opacity: 0.35;
}

/* Animation */
@keyframes zigzagMove {
    from {
        background-position: 0 0;
    }
    to {
        background-position: 800px 800px;
    }
}

</style>
<div class="cyber-lines"></div>
<div class="hud-scan"></div>
<div class="zigzag-lines"></div>
""", unsafe_allow_html=True)


# Import modul custom
# Pastikan file sbox_generator.py, sbox_test.py, dan aes_engine.py ada di folder yang sama
from sbox_generator import SBOXES, INV_SBOXES, AFFINE_128_DICT
