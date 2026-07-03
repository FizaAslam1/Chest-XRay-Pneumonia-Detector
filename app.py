import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Pneumonia Detector | AI X-Ray",
    page_icon="🏥",
    layout="centered"
)

# ==========================================
# CUSTOM CSS — BRIGHT & CLEAN
# ==========================================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f0f4f8, #e2e8f0);
    }
    
    h1, h2, h3, h4, p, label, div {
        color: #1a202c !important;
    }
    
    .stSidebar {
        background: linear-gradient(180deg, #1e293b, #334155);
    }
    .stSidebar h2, .stSidebar h3, .stSidebar p, .stSidebar label, .stSidebar div {
        color: #f1f5f9 !important;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #2563eb, #7c3aed);
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 1rem 2rem;
        font-size: 1.2rem;
        font-weight: 700;
        width: 100%;
        letter-spacing: 1px;
        box-shadow: 0 4px 15px rgba(37,99,235,0.4);
    }
    .stButton > button:hover {
        box-shadow: 0 8px 25px rgba(124,58,237,0.6);
    }
    
    .stFileUploader > div {
        background: white;
        border: 2px dashed #2563eb;
        border-radius: 15px;
        padding: 2rem;
    }
    
    .result-normal {
        background: linear-gradient(135deg, #dcfce7, #bbf7d0);
        border: 3px solid #16a34a;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
    }
    .result-normal h2 { color: #166534 !important; font-size: 2rem; }
    .result-normal h3 { color: #15803d !important; }
    .result-normal p { color: #14532d !important; font-size: 1.2rem; }
    
    .result-pneumonia {
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        border: 3px solid #dc2626;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
    }
    .result-pneumonia h2 { color: #991b1b !important; font-size: 2rem; }
    .result-pneumonia h3 { color: #b91c1c !important; }
    .result-pneumonia p { color: #7f1d1d !important; font-size: 1.2rem; }
    
    .title-text {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        color: #1e40af !important;
    }
    
    .subtitle-text {
        text-align: center;
        color: #475569 !important;
        font-size: 1.2rem;
    }
    
    .stInfo {
        background-color: #dbeafe !important;
        color: #1e40af !important;
    }
    .stSuccess {
        background-color: #dcfce7 !important;
        color: #166534 !important;
    }
    .stError {
        background-color: #fee2e2 !important;
        color: #991b1b !important;
    }
    
    .threshold-badge {
        display: inline-block;
        background: #e2e8f0;
        color: #475569 !important;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('pneumonia_detector.keras')

try:
    model = load_model()
    st.sidebar.success("✅ Model Loaded Successfully!")
except:
    st.sidebar.error("❌ Model not found! Upload 'pneumonia_detector.keras'")
    st.stop()

# ==========================================
# THRESHOLD SETTING
# ==========================================
THRESHOLD = 0.65  # Tune this for fewer false positives

# ==========================================
# HEADER
# ==========================================
st.markdown('<h1 class="title-text">🏥 Pneumonia Detection</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">AI-Powered Chest X-Ray Analysis</p>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# MODEL INFO (SIDEBAR)
# ==========================================
with st.sidebar:
    st.markdown("## 📊 Model Info")
    st.markdown("---")
    st.markdown("**Architecture:** VGG16 (Transfer Learning)")
    st.markdown("**Input Size:** 224 × 224 pixels")
    st.markdown("**Training Data:** 5,216 X-Rays")
    st.markdown("**Accuracy:** 84%")
    st.markdown("**Recall:** 98.5% 🏆")
    st.markdown("**Precision:** 80.3%")
    st.markdown("**Threshold:** 0.65")
    st.markdown("**Classes:** Normal / Pneumonia")
    st.markdown("---")
    st.markdown("## ⚠️ Disclaimer")
    st.warning("This is a screening tool, not a final medical diagnosis. Always consult a doctor!")
    st.markdown("---")
    st.markdown("## 🔧 Settings")
    st.markdown(f"Current Threshold: **{THRESHOLD}**")
    st.caption("Higher threshold = Fewer false alarms")
    st.caption("Lower threshold = Catch more pneumonia cases")

# ==========================================
# UPLOAD SECTION
# ==========================================
st.markdown("### 📤 Upload Chest X-Ray Image")
uploaded_file = st.file_uploader(
    "Choose an X-Ray image...", 
    type=["jpg", "jpeg", "png"],
    help="Upload a chest X-Ray image for pneumonia detection"
)

# ==========================================
# PREDICTION
# ==========================================
if uploaded_file is not None:
    # Display image
    col1, col2, col3 = st.columns([0.15, 0.7, 0.15])
    with col2:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption='Uploaded X-Ray', use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Predict button
    if st.button("🔍 Analyze X-Ray", use_container_width=True):
        with st.spinner("🧠 AI analyzing X-Ray..."):
            # Preprocess
            img_resized = img.resize((224, 224))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            prediction = model.predict(img_array, verbose=0)[0][0]
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if prediction < THRESHOLD:
                confidence = (1 - prediction) * 100
                st.markdown(f"""
                <div class="result-normal">
                    <h2>✅ NORMAL</h2>
                    <h3>No Pneumonia Detected</h3>
                    <p>Confidence: {confidence:.1f}%</p>
                    <span class="threshold-badge">Threshold: {THRESHOLD}</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Additional info
                st.info("💡 The lungs appear clear with no signs of infection. However, always consult a doctor for final diagnosis.")
                
            else:
                confidence = prediction * 100
                st.markdown(f"""
                <div class="result-pneumonia">
                    <h2>⚠️ PNEUMONIA DETECTED</h2>
                    <h3>Please Consult a Doctor Immediately</h3>
                    <p>Confidence: {confidence:.1f}%</p>
                    <span class="threshold-badge">Threshold: {THRESHOLD}</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Additional info
                st.warning("🚨 The AI has detected signs consistent with pneumonia. This is a screening result — please seek professional medical advice immediately!")
            
            # Raw prediction
            with st.expander("🔬 Technical Details"):
                st.write(f"Raw Prediction Score: {prediction:.4f}")
                st.write(f"Threshold Used: {THRESHOLD}")
                st.write(f"Prediction: {'Pneumonia' if prediction >= THRESHOLD else 'Normal'}")

# ==========================================
# HOW TO USE (WHEN NO FILE UPLOADED)
# ==========================================
if uploaded_file is None:
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([0.1, 0.8, 0.1])
    with col2:
        st.info("""
        ### 📋 How to Use:
        1. Upload a chest X-Ray image (JPG, JPEG, or PNG)
        2. Click "Analyze X-Ray"
        3. Get instant AI-powered results
        
        ⚡ The model uses VGG16 Transfer Learning trained on 5,000+ chest X-Rays
        """)

# ==========================================
# FOOTER
# ==========================================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#475569; padding:1rem; border-top:2px solid #e2e8f0;">
    <p><strong>🏥 Pneumonia Detection</strong> — Powered by VGG16 & Streamlit</p>
    <p style="font-size:0.8rem;">© 2026 Fiza Aslam | Data Scientist</p>
</div>
""", unsafe_allow_html=True)