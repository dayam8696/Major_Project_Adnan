import streamlit as st
import pandas as pd
import joblib

# Page config
st.set_page_config(page_title="Stress Level Predictor",page_icon="🧠", layout="centered")

# Enhanced CSS with vibrant colors and animations
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif;
        }
        
        .main {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 2.5rem;
            border-radius: 20px;
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.15);
            backdrop-filter: blur(4px);
            border: 1px solid rgba(255, 255, 255, 0.18);
        }
        
        .stButton>button {
            background: linear-gradient(45deg, #FF4E50 0%, #F9D423 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 14px 32px;
            font-size: 18px;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(255, 78, 80, 0.3);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(255, 78, 80, 0.4);
            background: linear-gradient(45deg, #F9D423 0%, #FF4E50 100%);
        }
        
        .stSelectbox>div>div {
            background-color: rgba(255, 255, 255, 0.8);
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            padding: 10px;
            transition: all 0.3s ease;
        }
        
        .stSelectbox>div>div:hover {
            border-color: #FF4E50;
            box-shadow: 0 0 0 2px rgba(255, 78, 80, 0.2);
        }
        
        .title {
            background: linear-gradient(45deg, #6a11cb 0%, #2575fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 48px;
            font-weight: 800;
            text-align: center;
            margin-bottom: 15px;
            letter-spacing: -0.5px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.05);
        }
        
        .subtitle {
            color: #4b5563;
            font-size: 20px;
            text-align: center;
            margin-bottom: 40px;
            font-weight: 400;
        }
        
        .footer {
            text-align: center;
            color: #9ca3af;
            font-size: 14px;
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid rgba(0,0,0,0.1);
        }
        
        .success-box {
            background: linear-gradient(135deg, rgba(46,213,115,0.2) 0%, rgba(32,178,170,0.2) 100%);
            color: #065f46;
            padding: 22px;
            border-radius: 15px;
            text-align: center;
            font-size: 24px;
            font-weight: 700;
            margin-top: 30px;
            border: 2px solid #2ed573;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.02); }
            100% { transform: scale(1); }
        }
        
        .question-card-container {
            margin-bottom: 25px;
        }
        
        .question-card {
            background: rgba(255, 255, 255, 0.9);
            border-left: 6px solid;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 15px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            transition: all 0.3s ease;
        }
        
        .question-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        }
        
        .question-title {
            font-size: 17px;
            font-weight: 500;
            margin-bottom: 8px;
        }
        
        /* Gradient borders for different questions */
        .question-card:nth-child(1) { border-color: #FF9A8B; }
        .question-card:nth-child(2) { border-color: #FF6B6B; }
        .question-card:nth-child(3) { border-color: #4ECDC4; }
        .question-card:nth-child(4) { border-color: #45B7D1; }
        .question-card:nth-child(5) { border-color: #A18CD1; }
        .question-card:nth-child(6) { border-color: #FFB347; }
        .question-card:nth-child(7) { border-color: #92FE9D; }
        .question-card:nth-child(8) { border-color: #FF8E53; }
        
        /* Floating animation for the brain icon */
        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
            100% { transform: translateY(0px); }
        }
        
        .floating-icon {
            animation: float 3s ease-in-out infinite;
            display: inline-block;
        }
        
        .response-header {
            color: #2563eb; 
            text-align: center; 
            margin-bottom: 25px;
            font-size: 22px;
            font-weight: 600;
        }
    </style>
""", unsafe_allow_html=True)

# Load model and encoders
try:
    rf_model = joblib.load('/Users/adnan/Desktop/MAJORPROJECT/stress_level_model.pkl')
    encoders = joblib.load('/Users/adnan/Desktop/MAJORPROJECT/label_encoders.pkl')
    st.success("✅ Model and encoders loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model or encoders: {e}")
    st.stop()

# Label mapping
label_mapping = {0: 'High', 1: 'Low', 2: 'Medium'}

# Valid input options
response_options = {
    'How often do you feel irritable or easily annoyed?': ['Rarely', 'Sometimes', 'often', 'always'],
    'Which of the following best describes your sleep recently?': [
        'I sleep well through the night',
        'I have trouble falling asleep',
        'I wake up frequently during the night',
        'I barely sleep at all'
    ],
    'How often do you experience palpitations or a racing heart (especially when anxious)?': [
        'Rarely', 'Sometimes', 'Often', 'Always'
    ],
    'Do you experience intense fear or phobia (e.g., of heights, crowds, darkness, etc.)?': [
        'No', 'Occasionally', 'Frequently', 'Daily'
    ],
    'How often do you find yourself overthinking or worrying excessively?': [
        'Rarely', 'Sometimes', 'Often', 'All the time'
    ],
    'Do you feel emotionally overwhelmed or tearful easily?': [
        'Rarely – only in extreme situations',
        'Sometimes – when under stress',
        'Often – with minor emotional triggers',
        'Very often – even without any clear reason'
    ],
    'Do you ever feel choking or have trouble breathing in open spaces (e.g., fields, public areas)?': [
        'No', 'Rarely', 'Sometimes', 'Often'
    ],
    'How often do you experience indigestion, stomach discomfort, or nausea when stressed?': [
        'Rarely – once in a while',
        'Sometimes – a few times a month',
        'Often – at least once a week',
        'Very often – multiple times a week or daily'
    ]
}

# UI Headings with floating icon
st.markdown('<div class="title"><span class="floating-icon">🧠</span> Stress Level Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Answer the questions below to predict your stress level (Low, Medium, or High).</div>', unsafe_allow_html=True)

# User input form
with st.form("stress_form"):
    st.markdown('<div class="response-header">💬 Please select your responses below:</div>', unsafe_allow_html=True)

    user_inputs = {}
    for i, (question, options) in enumerate(response_options.items(), 1):
        st.markdown(f"""
            <div class="question-card">
                <div class="question-title"><b>Q{i}:</b> {question}</div>
            </div>
        """, unsafe_allow_html=True)
        user_inputs[question] = st.selectbox("", options, key=f"q{i}", index=None, placeholder="Select your response...")

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        submitted = st.form_submit_button("🎯 Predict My Stress Level", use_container_width=True)
    if submitted:
        st.info("⏳ Form submitted! Processing...")

# Prediction logic
if submitted:
    input_data = pd.DataFrame([user_inputs])
    input_data_encoded = input_data.copy()
    for column in input_data_encoded.columns:
        if column in encoders:
            try:
                input_data_encoded[column] = encoders[column].transform(input_data_encoded[column])
            except ValueError:
                st.error(f"⚠️ Error: Unseen value in '{column}'. Please select valid responses.")
                st.stop()

    prediction = rf_model.predict(input_data_encoded)
    predicted_label = label_mapping[prediction[0]]
    
    # Different colored boxes for different stress levels
    if predicted_label == 'High':
        box_color = "linear-gradient(135deg, rgba(255,71,87,0.2) 0%, rgba(255,107,107,0.2) 100%)"
        border_color = "#ff4757"
    elif predicted_label == 'Medium':
        box_color = "linear-gradient(135deg, rgba(255,165,0,0.2) 0%, rgba(255,215,0,0.2) 100%)"
        border_color = "#FFA500"
    else:
        box_color = "linear-gradient(135deg, rgba(46,213,115,0.2) 0%, rgba(32,178,170,0.2) 100%)"
        border_color = "#2ed573"
    
    st.markdown(
        f'<div class="success-box" style="background: {box_color}; border-color: {border_color};">'
        f'🎉 Predicted Stress Level: <b style="color: {border_color};">{predicted_label}</b></div>', 
        unsafe_allow_html=True
    )

    # Add some celebration emojis
    if predicted_label == 'Low':
        st.balloons()

# Footer
st.markdown("""
    <div class="footer">
        Made with ❤️ by <b>Adnan Alvi</b><br>
        <div style="margin-top: 10px; font-size: 12px;">
            <span style="color: #FF4E50;">●</span> 
            <span style="color: #F9D423;">●</span> 
            <span style="color: #6a11cb;">●</span> 
            <span style="color: #2575fc;">●</span> 
            <span style="color: #2ed573;">●</span>
        </div>
    </div>
""", unsafe_allow_html=True)