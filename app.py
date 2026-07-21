import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page Config
st.set_page_config(
    page_title="Plant Doctor | Disease Detector",
    page_icon="🌱",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #0f172a; }
    .header-title { color: #22c55e; font-size: 2.8rem; font-weight: 800; text-align: center; }
    .header-subtitle { color: #94a3b8; font-size: 1.1rem; text-align: center; margin-bottom: 30px; }
    div[data-testid="stFileUploader"] { background-color: #1e293b; border: 2px dashed #334155; border-radius: 16px; padding: 20px; text-align: center; }
    .result-card { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-left: 6px solid #22c55e; border-radius: 12px; padding: 20px; margin-top: 15px; }
    .disease-title { color: #f8fafc; font-size: 1.8rem; font-weight: 700; }
    .treatment-box { background-color: #1a2e26; border: 1px solid #22c55e; border-radius: 10px; padding: 15px; margin-top: 15px; color: #cbd5e1; }
    </style>
""", unsafe_allow_html=True)

# Direct Local Model Load
@st.cache_resource
def load_resources():
    model_path = 'model.h5'
    
    if not os.path.exists(model_path):
        st.error("`model.h5` file nahi mili!")
        return None, None
            
    model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
    class_names = np.load('class_names.npy')
    return model, class_names

try:
    model, CLASS_NAMES = load_resources()
except Exception as e:
    st.error(f"Error loading model: {e}")

# Treatments Dictionary
TREATMENTS = {
    'Potato___Early_blight': {
        'status': 'Disease Detected ⚠️',
        'color': '#f59e0b',
        'desc': 'Early Blight is caused by the fungus Alternaria solani. It produces brown spots with concentric rings on leaves.',
        'remedy': '✂️ Remove infected leaves immediately.<br>🧪 Spray Copper-based Fungicide or Mancozeb every 7-10 days.<br>💧 Avoid overhead watering; keep leaves dry.'
    },
    'Potato___Late_blight': {
        'status': 'Critical Disease Detected 🚨',
        'color': '#ef4444',
        'desc': 'Late Blight is a destructive water mold (Phytophthora infestans) that causes dark, water-soaked leaf lesions.',
        'remedy': '⚠️ Apply systemic fungicide like Ridomil Gold or Chlorothalonil instantly.<br>🔥 Destroy severely affected plants so it doesn\'t spread.<br>🌬️ Ensure proper distance between plants for air circulation.'
    },
    'Potato___healthy': {
        'status': 'Healthy Plant 🌱',
        'color': '#22c55e',
        'desc': 'Your plant shows no signs of fungal or bacterial infection. Leaves are clean and healthy!',
        'remedy': '✅ Maintain regular watering and fertilizing schedule.<br>🔍 Inspect plants weekly for early detection of pests or spots.'
    }
}

# Sidebar Content (With Quick Guide Steps)
with st.sidebar:
    st.image("https://img.icons8.com/color/96/natural-user-interface1.png", width=70)
    st.title("🌱 Plant Doctor")
    st.markdown("AI-Powered Crop Diagnostics")
    st.divider()
    st.subheader("📋 Quick Guide")
    st.markdown("""
    1. Take a clear photo of the leaf.
    2. Upload the photo in JPG/PNG format.
    3. Get instant diagnosis & remedies!
    """)

# Main UI
st.markdown('<div class="header-title">Plant Disease Diagnostics</div>', unsafe_allow_html=True)
st.markdown('<div class="header-subtitle">Upload a potato leaf photo for real-time AI disease prediction & treatment tips</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📸 Upload Leaf Photo")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Potato Leaf", use_container_width=True)

with col2:
    if uploaded_file is not None and model is not None:
        st.subheader("🔍 Diagnostic Report")
        
        img = image.resize((256, 256))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner("AI is analyzing leaf patterns..."):
            predictions = model.predict(img_array)
            predicted_index = np.argmax(predictions[0])
            raw_class_name = CLASS_NAMES[predicted_index]
            confidence = round(100 * np.max(predictions[0]), 2)

        info = TREATMENTS.get(raw_class_name, {
            'status': 'Analysis Complete',
            'color': '#22c55e',
            'desc': '',
            'remedy': 'Consult an agricultural specialist.'
        })

        clean_name = raw_class_name.replace("Potato___", "").replace("_", " ").title()

        st.markdown(f"""
            <div class="result-card" style="border-left-color: {info['color']};">
                <span style="color: {info['color']}; font-weight: bold; font-size: 0.9rem;">{info['status'].upper()}</span>
                <div class="disease-title">{clean_name}</div>
                <p style="color: #94a3b8; font-size: 0.95rem; margin-top: 5px;">{info['desc']}</p>
            </div>
        """, unsafe_allow_html=True)

        st.write("")
        st.markdown(f"**Model Confidence:** `{confidence}%`")
        st.progress(float(confidence / 100))

        st.markdown(f"""
            <div class="treatment-box">
                <h4 style="color: #22c55e; margin-bottom: 8px;">💡 Recommended Actions & Treatment:</h4>
                <p style="margin: 0; line-height: 1.6;">{info['remedy']}</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.info("👈 Upload a leaf image on the left panel to see the diagnostic report here.")
