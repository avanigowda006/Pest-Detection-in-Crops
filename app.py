"""
app.py - Pest Detection in Crops
Streamlit Application with CNN Model
Usage: streamlit run app.py
"""
import os, numpy as np, streamlit as st
from PIL import Image
import tensorflow as tf

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "pest_model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "models", "class_names.txt")
HISTORY_IMG = os.path.join(BASE_DIR, "static", "training_history.png")
IMG_SIZE = (128, 128)

PEST_INFO = {
    "Healthy": {"icon": "🌱", "desc": "The crop leaf appears healthy with no visible signs of pest damage.", "severity": "None", "color": "#00d084", "risk": 0, "remedy": "Continue regular monitoring and good agricultural practices.", "prevention": "Ensure proper irrigation, fertilization, and crop rotation."},
    "Aphids": {"icon": "🦟", "desc": "Small sap-sucking insects clustering on new growth, causing curling and yellowing.", "severity": "Moderate", "color": "#f0b429", "risk": 45, "remedy": "Apply neem oil spray, introduce ladybugs, or use insecticidal soap.", "prevention": "Use reflective mulch, companion planting with marigolds."},
    "Armyworm": {"icon": "🐛", "desc": "Caterpillar pests feeding on leaves in large groups, causing severe defoliation.", "severity": "High", "color": "#e53e3e", "risk": 80, "remedy": "Apply Bacillus thuringiensis (Bt) spray, use pheromone traps.", "prevention": "Early planting, use of resistant varieties, biological control."},
    "Beetles": {"icon": "🪲", "desc": "Beetle species chewing holes in leaves, damaging stems and roots.", "severity": "Moderate-High", "color": "#ed8936", "risk": 65, "remedy": "Hand-pick beetles, use row covers, apply pyrethrin-based pesticides.", "prevention": "Crop rotation, removal of plant debris, use of trap crops."},
    "Grasshopper": {"icon": "🦗", "desc": "Chewing insects consuming large quantities of foliage, causing ragged holes.", "severity": "Moderate", "color": "#dd6b20", "risk": 50, "remedy": "Use Nosema locustae biological control, kaolin clay barriers.", "prevention": "Maintain field borders, encourage natural predators like birds."},
}

# ── Crop Identification Database ──────────────────────────────────────────
CROP_DB = [
    {"name": "Rice",      "icon": "🌾", "color": "#4ade80",
     "r": (50, 120),  "g": (130, 210), "b": (40, 100),
     "ratio_g_min": 1.2, "bright": (100, 180),
     "desc": "Long, narrow leaf with parallel veins — typical of paddy/rice."},
    {"name": "Wheat",     "icon": "🌾", "color": "#fbbf24",
     "r": (100, 180), "g": (130, 200), "b": (40, 100),
     "ratio_g_min": 0.9, "bright": (110, 190),
     "desc": "Elongated leaf blade with golden-green hue — common in wheat."},
    {"name": "Corn (Maize)", "icon": "🌽", "color": "#22c55e",
     "r": (40, 100),  "g": (140, 220), "b": (30, 80),
     "ratio_g_min": 1.5, "bright": (90, 170),
     "desc": "Broad, bright-green leaf with prominent midrib — characteristic of corn."},
    {"name": "Tomato",    "icon": "🍅", "color": "#ef4444",
     "r": (60, 160),  "g": (100, 170), "b": (40, 110),
     "ratio_g_min": 0.8, "bright": (80, 160),
     "desc": "Compound leaf with serrated edges and fuzzy texture — tomato plant."},
    {"name": "Cotton",    "icon": "🏵️", "color": "#a78bfa",
     "r": (50, 130),  "g": (110, 180), "b": (50, 120),
     "ratio_g_min": 1.0, "bright": (85, 165),
     "desc": "Palmate leaf with 3-5 lobes and smooth surface — cotton plant."},
    {"name": "Soybean",   "icon": "🫘", "color": "#84cc16",
     "r": (50, 110),  "g": (120, 190), "b": (35, 90),
     "ratio_g_min": 1.3, "bright": (80, 155),
     "desc": "Trifoliate leaf with oval leaflets — soybean plant."},
    {"name": "Potato",    "icon": "🥔", "color": "#78716c",
     "r": (55, 140),  "g": (100, 165), "b": (45, 100),
     "ratio_g_min": 0.9, "bright": (75, 150),
     "desc": "Pinnately compound leaf with dark-green surface — potato plant."},
    {"name": "Sugarcane", "icon": "🎋", "color": "#14b8a6",
     "r": (40, 100),  "g": (135, 210), "b": (45, 95),
     "ratio_g_min": 1.4, "bright": (95, 175),
     "desc": "Long, narrow blade with waxy surface — sugarcane leaf."},
]

def identify_crop(image):
    """Identify the crop type from a leaf image using colour-profile analysis."""
    arr = np.array(image.resize((64, 64)), dtype=np.float64)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    r_mean = arr[:, :, 0].mean()
    g_mean = arr[:, :, 1].mean()
    b_mean = arr[:, :, 2].mean()
    brightness = (r_mean + g_mean + b_mean) / 3
    g_ratio = g_mean / max(r_mean, 1)  # how dominant green is

    # Compute a colour-distance score for every crop
    best_crop = None
    best_score = -1
    for crop in CROP_DB:
        score = 0
        r_lo, r_hi = crop["r"]
        g_lo, g_hi = crop["g"]
        b_lo, b_hi = crop["b"]
        br_lo, br_hi = crop["bright"]

        # Channel-in-range check  (0-30 pts each)
        for val, lo, hi in [(r_mean, r_lo, r_hi), (g_mean, g_lo, g_hi), (b_mean, b_lo, b_hi)]:
            if lo <= val <= hi:
                # Closer to centre of range → higher score
                mid = (lo + hi) / 2
                dist = abs(val - mid) / max((hi - lo) / 2, 1)
                score += 30 * (1 - dist * 0.5)
            else:
                score -= 10

        # Brightness check  (0-15 pts)
        if br_lo <= brightness <= br_hi:
            score += 15

        # Green-ratio check  (0-15 pts)
        if g_ratio >= crop["ratio_g_min"]:
            score += 15

        if score > best_score:
            best_score = score
            best_crop = crop

    return best_crop

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

def load_class_names():
    if not os.path.exists(LABELS_PATH):
        return list(PEST_INFO.keys())
    with open(LABELS_PATH) as f:
        return [l.strip() for l in f if l.strip()]

def preprocess(image):
    img = image.resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    return np.expand_dims(arr, axis=0)

def is_crop_image(image):
    """Check if image has plant/leaf-like colors."""
    arr = np.array(image.resize((64, 64)))
    if arr.ndim == 2:
        return False
    r = arr[:, :, 0].mean()
    g = arr[:, :, 1].mean()
    b = arr[:, :, 2].mean()
    score = 0
    if g > r and g > b:
        score += 30
    if 40 < r < 180 and 30 < g < 160 and 20 < b < 120:
        score += 25
    if r > 100 and g > 100 and b < 120:
        score += 20
    if r < 100 and g < 100 and b < 100:
        score += 15
    if max(r, g, b) - min(r, g, b) < 150:
        score += 10
    return score >= 30

# Page Config
st.set_page_config(page_title="Pest Detection in Crops - CNN", page_icon="🌿", layout="wide")

# CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
.stApp { background: linear-gradient(135deg, #0b1120 0%, #111827 50%, #0d1321 100%); }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #111827 0%, #1a2332 100%); }
.hero { background: linear-gradient(135deg, #047857 0%, #10b981 50%, #34d399 100%); padding: 3rem 2rem; border-radius: 24px; text-align: center; margin-bottom: 2rem; box-shadow: 0 25px 50px -12px rgba(16,185,129,0.4); position: relative; overflow: hidden; }
.hero h1 { font-size: 2.8rem; font-weight: 800; color: white; margin: 0; letter-spacing: -1px; }
.hero p { color: rgba(255,255,255,0.9); font-size: 1.1rem; margin: 0.5rem 0 0; }
.glass-card { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 16px; padding: 1.8rem; margin: 1rem 0; }
.result-card { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 20px; padding: 2rem; margin: 1rem 0; border-left: 5px solid var(--accent); box-shadow: 0 8px 32px rgba(0,0,0,0.3); }
.ring-container { display: flex; align-items: center; justify-content: center; margin: 1.5rem 0; }
.ring { width: 140px; height: 140px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 2rem; font-weight: 800; color: white; }
.ring small { font-size: 0.7rem; font-weight: 400; opacity: 0.7; margin-top: 2px; }
.stat-box { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; padding: 1.2rem; text-align: center; }
.stat-box .label { font-size: 0.75rem; color: rgba(255,255,255,0.5); text-transform: uppercase; letter-spacing: 1px; }
.stat-box .value { font-size: 1.3rem; color: white; font-weight: 700; margin-top: 0.3rem; }
.risk-track { background: rgba(255,255,255,0.08); border-radius: 12px; height: 10px; overflow: hidden; margin: 0.5rem 0; }
.risk-fill { height: 100%; border-radius: 12px; }
.pill { display: inline-block; padding: 0.4rem 1rem; border-radius: 50px; font-size: 0.8rem; font-weight: 600; }
.detail-section { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 14px; padding: 1.2rem 1.5rem; margin: 0.6rem 0; }
.detail-section h4 { color: rgba(255,255,255,0.6); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; margin: 0 0 0.5rem; }
.detail-section p { color: rgba(255,255,255,0.85); margin: 0; font-size: 0.95rem; line-height: 1.6; }
.sidebar-title { color: #10b981; font-size: 1.4rem; font-weight: 700; }
.footer { text-align: center; padding: 2rem; margin-top: 3rem; border-top: 1px solid rgba(255,255,255,0.06); color: rgba(255,255,255,0.3); font-size: 0.8rem; }
.stMarkdown p, .stMarkdown li { color: rgba(255,255,255,0.75); }
h1,h2,h3,h4 { color: white !important; }
[data-testid="stFileUploader"] section { border: 2px dashed rgba(16,185,129,0.4) !important; border-radius: 16px; padding: 1.5rem; background: rgba(16,185,129,0.05); }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<p class="sidebar-title">🌿 Pest Detection</p>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Pest Detection in Crops** powered by CNN deep learning and image processing.")
    st.markdown("---")
    st.markdown("### Detectable Pests")
    for name, info in PEST_INFO.items():
        st.markdown(f'{info["icon"]} **{name}** — {info["severity"]}')
    st.markdown("---")
    st.markdown("### Supported Crops")
    for crop in CROP_DB:
        st.markdown(f'{crop["icon"]} **{crop["name"]}**')
    st.markdown("---")
    st.markdown("### Model Details")
    st.markdown("- **Architecture:** CNN (3 Conv Blocks)")
    st.markdown(f"- **Input:** {IMG_SIZE[0]}x{IMG_SIZE[1]} RGB")
    st.markdown(f"- **Classes:** {len(PEST_INFO)}")
    st.markdown("- **Framework:** TensorFlow / Keras")

# Hero
st.markdown("""
<div class="hero">
    <h1>🌿 Pest Detection in Crops</h1>
    <p>Using Image Processing &amp; Convolutional Neural Network (CNN)</p>
</div>
""", unsafe_allow_html=True)

# Load model
model = load_model()
class_names = load_class_names()

if model is None:
    st.error("**Model not found!** Run `python train.py` first.")
    st.stop()

# Stats bar
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="stat-box"><div class="label">Model</div><div class="value">CNN</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="stat-box"><div class="label">Classes</div><div class="value">{len(class_names)}</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="stat-box"><div class="label">Input Size</div><div class="value">{IMG_SIZE[0]}px</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="stat-box"><div class="label">Status</div><div class="value" style="color:#10b981;">Active</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Main layout
col_upload, col_result = st.columns([1, 1], gap="large")

# Run prediction early so we can show the label under the image
prediction_result = None
if "uploaded" not in dir():
    uploaded = None

with col_upload:
    st.markdown("### 📤 Upload Crop Image")
    uploaded = st.file_uploader("Select a crop leaf or pest image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")

        # Run prediction first so we can display name below image
        is_crop = is_crop_image(image)
        crop_detected = None
        if is_crop:
            with st.spinner("Analyzing..."):
                inp = preprocess(image)
                preds = model.predict(inp, verbose=0)[0]
                idx = int(np.argmax(preds))
                conf = float(preds[idx]) * 100
                cls = class_names[idx]
                p_info = PEST_INFO.get(cls, {"icon": "🌿", "desc": "Unknown", "severity": "Unknown", "color": "#888", "risk": 0, "remedy": "Consult expert.", "prevention": "N/A"})
                prediction_result = {"cls": cls, "conf": conf, "info": p_info, "preds": preds}
                crop_detected = identify_crop(image)

        # Show uploaded image
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # ── Crop Name + Pest Status Card (below image) ──
        if is_crop and prediction_result and crop_detected:
            p = prediction_result
            c = crop_detected
            pest_color = p["info"]["color"]

            if p["cls"] == "Healthy":
                status_line = "✅ No pest detected — Leaf is healthy"
                status_color = "#00d084"
            else:
                status_line = f'⚠️ Pest Detected: {p["info"]["icon"]} {p["cls"]}'
                status_color = pest_color

            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
                border: 1px solid rgba(255,255,255,0.12);
                border-radius: 16px;
                padding: 1.2rem 1.4rem;
                margin-top: 0.8rem;
            ">
                <!-- Crop Name -->
                <div style="display:flex; align-items:center; gap:0.7rem; margin-bottom:0.8rem;">
                    <span style="font-size:2.2rem;">{c['icon']}</span>
                    <div>
                        <p style="
                            color: {c['color']};
                            font-size: 1.4rem;
                            font-weight: 800;
                            margin: 0;
                            letter-spacing: 0.3px;
                        ">Crop: {c['name']}</p>
                        <p style="
                            color: rgba(255,255,255,0.45);
                            font-size: 0.78rem;
                            margin: 0.15rem 0 0;
                        ">{c['desc']}</p>
                    </div>
                </div>
                <!-- Divider -->
                <div style="height:1px; background:rgba(255,255,255,0.08); margin:0.6rem 0;"></div>
                <!-- Pest Status -->
                <div style="display:flex; align-items:center; justify-content:space-between;">
                    <p style="
                        color: {status_color};
                        font-size: 1.05rem;
                        font-weight: 700;
                        margin: 0;
                    ">{status_line}</p>
                    <span style="
                        background: {status_color}18;
                        color: {status_color};
                        padding: 0.3rem 0.8rem;
                        border-radius: 50px;
                        font-size: 0.78rem;
                        font-weight: 600;
                    ">{p['conf']:.1f}% conf</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif uploaded and not is_crop:
            st.markdown("""
            <div style="
                background: rgba(239,68,68,0.08);
                border: 2px solid #ef4444;
                border-radius: 14px;
                padding: 1rem 1.2rem;
                margin-top: 0.8rem;
                text-align: center;
            ">
                <span style="font-size: 1.6rem;">❌</span>
                <p style="
                    color: #ef4444;
                    font-size: 1.1rem;
                    font-weight: 700;
                    margin: 0.3rem 0 0;
                ">Not a Crop / Leaf Image</p>
            </div>
            """, unsafe_allow_html=True)

with col_result:
    if uploaded:
        st.markdown("### 🔬 Analysis Result")

        if not is_crop:
            st.markdown("""
            <div class="result-card" style="--accent:#ef4444;">
                <div style="display:flex; align-items:center; gap:1rem; margin-bottom:1rem;">
                    <span style="font-size:3rem;">❌</span>
                    <div>
                        <h2 style="margin:0; color:#ef4444!important; font-size:1.8rem;">Not a Crop Image</h2>
                        <span class="pill" style="background:#ef444422; color:#ef4444;">Invalid Input</span>
                    </div>
                </div>
                <p style="color:rgba(255,255,255,0.7); line-height:1.6;">
                    This does not appear to be a crop or plant leaf image.
                    Please upload a clear image of a <strong>crop leaf</strong>.
                </p>
                <div class="detail-section" style="margin-top:1rem;">
                    <h4>💡 Please Upload</h4>
                    <p>• Images of crop leaves (healthy or pest-affected)<br>
                    • Supported: Aphids, Armyworm, Beetles, Grasshopper<br>
                    • Do NOT upload: animals, people, vehicles, random objects</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif prediction_result:
            p = prediction_result
            info = p["info"]
            cls = p["cls"]
            conf = p["conf"]
            preds = p["preds"]
            accent = info["color"]

            # Crop info banner at top of results
            if crop_detected:
                c = crop_detected
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, {c['color']}15, {c['color']}05);
                    border: 1px solid {c['color']}40;
                    border-radius: 14px;
                    padding: 1rem 1.3rem;
                    margin-bottom: 1rem;
                    display: flex; align-items: center; gap: 0.8rem;
                ">
                    <span style="font-size:2rem;">{c['icon']}</span>
                    <div>
                        <p style="color:{c['color']}; font-size:1.1rem; font-weight:700; margin:0;">Identified Crop: {c['name']}</p>
                        <p style="color:rgba(255,255,255,0.45); font-size:0.75rem; margin:0.1rem 0 0;">{c['desc']}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Result card
            st.markdown(f"""
            <div class="result-card" style="--accent:{accent};">
                <div style="display:flex; align-items:center; gap:1rem; margin-bottom:1rem;">
                    <span style="font-size:3rem;">{info["icon"]}</span>
                    <div>
                        <h2 style="margin:0; color:{accent}!important; font-size:1.8rem;">{cls}</h2>
                        <span class="pill" style="background:{accent}22; color:{accent};">{info["severity"]} Severity</span>
                    </div>
                </div>
                <p style="color:rgba(255,255,255,0.7); line-height:1.6;">{info["desc"]}</p>
                <div class="ring-container">
                    <div class="ring" style="background: conic-gradient({accent} {conf}%, rgba(255,255,255,0.08) {conf}%);">
                        <div style="width:110px;height:110px;border-radius:50%;background:#111827;display:flex;align-items:center;justify-content:center;flex-direction:column;">
                            {conf:.1f}%<small>Confidence</small>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Risk bar
            risk_color = "#10b981" if info["risk"] < 30 else ("#f0b429" if info["risk"] < 60 else "#e53e3e")
            st.markdown(f"""
            <div class="detail-section">
                <h4>Risk Level</h4>
                <div class="risk-track"><div class="risk-fill" style="width:{info['risk']}%; background:linear-gradient(90deg,{accent},{risk_color});"></div></div>
                <p style="font-size:0.85rem; margin-top:0.3rem;">{info['risk']}% risk index</p>
            </div>
            """, unsafe_allow_html=True)

            # Remedy & Prevention
            d1, d2 = st.columns(2)
            with d1:
                st.markdown(f'<div class="detail-section"><h4>💊 Remedy</h4><p>{info["remedy"]}</p></div>', unsafe_allow_html=True)
            with d2:
                st.markdown(f'<div class="detail-section"><h4>🛡️ Prevention</h4><p>{info["prevention"]}</p></div>', unsafe_allow_html=True)

            # Confidence breakdown
            st.markdown("#### 📊 All Class Probabilities")
            for i, name in enumerate(class_names):
                pct = float(preds[i]) * 100
                st.progress(pct / 100, text=f"{name}: {pct:.1f}%")

    else:
        st.markdown("""
        <div class="glass-card">
            <h3 style="margin-top:0;">📌 How to Use</h3>
            <ol style="color:rgba(255,255,255,0.7); line-height:2;">
                <li>Upload a crop leaf or pest image on the left</li>
                <li>The CNN model will automatically analyze the image</li>
                <li>View pest identification, severity, and recommendations</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

# Training history
if os.path.exists(HISTORY_IMG):
    with st.expander("📈 Training History", expanded=False):
        st.image(HISTORY_IMG, caption="Training Accuracy & Loss", use_container_width=True)

# Footer
st.markdown("""
<div class="footer">
    🌿 Pest Detection in Crops — CNN &amp; Image Processing | Built with TensorFlow &amp; Streamlit
</div>
""", unsafe_allow_html=True)
