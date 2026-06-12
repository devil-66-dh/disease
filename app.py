import streamlit as st
import joblib
import pandas as pd

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Crop Recommendation",
    page_icon="🌱",
    layout="wide"
)

# ================= CSS =================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]{
    font-family:'Poppins',sans-serif;
}

/* Background */
.stApp{
background:
radial-gradient(circle at top left, rgba(34,197,94,0.25), transparent 30%),
radial-gradient(circle at bottom right, rgba(132,204,22,0.20), transparent 30%),
linear-gradient(
135deg,
#041c12 0%,
#0b3d20 30%,
#14532d 70%,
#052e16 100%
);
}

/* Hide Streamlit Branding */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* Glass Container */
.block-container{
padding-top:2rem;
}

/* Labels */
label{
color:white !important;
font-weight:600 !important;
}

/* Inputs */
.stNumberInput input{
background:white !important;
color:#14532d !important;
border-radius:15px !important;
border:2px solid #22c55e !important;
font-weight:600 !important;
}

/* Button */
.stButton button{
width:100%;
height:60px;
border:none;
border-radius:18px;
background:linear-gradient(135deg,#22c55e,#16a34a);
color:white;
font-size:20px;
font-weight:700;
}

.stButton button:hover{
transform:scale(1.02);
}

/* Subheaders */
h3{
color:white !important;
}

</style>
""", unsafe_allow_html=True)

# ================= HERO =================
st.markdown("""
<div style="
background:linear-gradient(135deg,#22c55e,#15803d);
padding:35px;
border-radius:25px;
text-align:center;
margin-bottom:25px;
box-shadow:0px 8px 30px rgba(0,0,0,0.25);
">
<h1 style="color:white;margin:0;">
🌾 Smart Crop Recommendation
</h1>
<p style="color:white;font-size:18px;margin-top:10px;">
AI Powered Farming Assistant for Better Yield & Smart Agriculture
</p>
</div>
""", unsafe_allow_html=True)

# ================= STATS =================
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🌱 Supported Crops", "22+")

with col2:
    st.metric("🎯 Model Accuracy", "99%")

with col3:
    st.metric("📊 Parameters Used", "7")

st.markdown("<br>", unsafe_allow_html=True)

# ================= CROP CARDS =================
st.markdown("### 🌾 Popular Crops")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("""
    <div style="background:rgba(255,255,255,0.1);
    padding:25px;border-radius:20px;text-align:center;color:white;">
    🌾<br><h4>Rice</h4>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div style="background:rgba(255,255,255,0.1);
    padding:25px;border-radius:20px;text-align:center;color:white;">
    🌽<br><h4>Maize</h4>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div style="background:rgba(255,255,255,0.1);
    padding:25px;border-radius:20px;text-align:center;color:white;">
    🌿<br><h4>Cotton</h4>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown("""
    <div style="background:rgba(255,255,255,0.1);
    padding:25px;border-radius:20px;text-align:center;color:white;">
    🌻<br><h4>Sunflower</h4>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ================= LOAD MODEL =================
model_data = joblib.load("crop_recommendation_model.joblib")

pipeline = model_data["pipeline"]
label_encoder = model_data["label_encoder"]

# ================= INPUTS =================
st.markdown("### 🧪 Soil & Weather Details")

col1, col2, col3 = st.columns(3)

with col1:
    N = st.number_input("Nitrogen (N)", 0, 200, 90)
    temperature = st.number_input("Temperature (°C)", value=20.8)

with col2:
    P = st.number_input("Phosphorus (P)", 0, 200, 42)
    humidity = st.number_input("Humidity (%)", value=82.0)

with col3:
    K = st.number_input("Potassium (K)", 0, 200, 43)
    ph = st.number_input("pH Value", value=6.5)

rainfall = st.number_input("Rainfall (mm)", value=202.0)

st.markdown("<br>", unsafe_allow_html=True)

# ================= PREDICT =================
if st.button("🚀 Recommend Best Crop"):

    sample = pd.DataFrame([{
        "N": N,
        "P": P,
        "K": K,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }])

    prediction = pipeline.predict(sample)
    crop = label_encoder.inverse_transform(prediction)[0]

    st.markdown(f"""
<div style="
background:rgba(255,255,255,0.04);
backdrop-filter:blur(15px);
border:1px solid rgba(255,255,255,0.1);
padding:30px;
border-radius:25px;
text-align:center;
margin-top:25px;
">
<h2 style="color:#86efac;">🌾 Recommended Crop</h2>
<h1 style="color:white;font-size:60px;">
{crop.upper()}
</h1>
</div>
""", unsafe_allow_html=True)
