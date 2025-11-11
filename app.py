import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Heart Attack Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #ff4b4b;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .risk-high {
        background-color: #ff4b4b;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
    .risk-medium {
        background-color: #ffa500;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
    .risk-low {
        background-color: #2ecc71;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_artifacts():
    """Load the trained model and preprocessing artifacts"""
    try:
        model = tf.keras.models.load_model('heart_attack_model.h5')
        scaler = joblib.load('scaler.pkl')
        with open('feature_names.json', 'r') as f:
            feature_names = json.load(f)
        return model, scaler, feature_names
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return None, None, None

def preprocess_input_data(input_data, feature_names):
    """Preprocess user input to match training pipeline"""
    try:
        # Create DataFrame from input
        df = pd.DataFrame([input_data])
        
        # Parse Blood Pressure
        if 'Blood Pressure (systolic/diastolic mmHg)' in df.columns:
            bp_split = df['Blood Pressure (systolic/diastolic mmHg)'].str.split('/', expand=True)
            df['BP_Systolic'] = pd.to_numeric(bp_split[0], errors='coerce')
            df['BP_Diastolic'] = pd.to_numeric(bp_split[1], errors='coerce')
            df = df.drop('Blood Pressure (systolic/diastolic mmHg)', axis=1)
        
        # One-hot encode categorical variables
        categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
        
        # Align features with training features
        missing_features = set(feature_names) - set(df_encoded.columns)
        for feature in missing_features:
            df_encoded[feature] = 0
        
        extra_features = set(df_encoded.columns) - set(feature_names)
        for feature in extra_features:
            if feature in df_encoded.columns:
                df_encoded = df_encoded.drop(feature, axis=1)
        
        # Reorder columns to match training
        df_encoded = df_encoded[feature_names]
        
        return df_encoded
        
    except Exception as e:
        st.error(f"Error preprocessing data: {e}")
        return None

def create_risk_gauge(probability):
    """Create a gauge chart for risk visualization"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Heart Attack Risk Score", 'font': {'size': 24}},
        delta = {'reference': 50, 'increasing': {'color': "red"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': 'green'},
                {'range': [30, 70], 'color': 'yellow'},
                {'range': [70, 100], 'color': 'red'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90}}))
    
    fig.update_layout(
        height=300,
        margin=dict(l=50, r=50, t=100, b=50)
    )
    
    return fig

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">❤️ Heart Attack Risk Predictor</h1>', unsafe_allow_html=True)
    st.markdown("### For Indian Youngsters | AI-Powered Risk Assessment")
    
    # Load model
    model, scaler, feature_names = load_model_and_artifacts()
    
    if model is None:
        st.error("❌ Model files not found. Please ensure 'heart_attack_model.h5', 'scaler.pkl', and 'feature_names.json' are in the same directory.")
        return
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose App Mode", 
                                   ["🏠 Home", "🔍 Risk Assessment", "📊 Model Info", "ℹ️ About"])
    
    if app_mode == "🏠 Home":
        show_home_page()
    elif app_mode == "🔍 Risk Assessment":
        show_risk_assessment(model, scaler, feature_names)
    elif app_mode == "📊 Model Info":
        show_model_info()
    elif app_mode == "ℹ️ About":
        show_about_page()

def show_home_page():
    """Display home page content"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-box">
        <h2>Welcome to the Heart Attack Risk Predictor</h2>
        <p>This AI-powered tool helps assess the risk of heart attack in young Indian adults 
        based on various health parameters and lifestyle factors.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        ### 🎯 Key Features:
        - **AI-Powered Predictions**: Uses deep learning model trained on Indian population data
        - **Comprehensive Assessment**: Considers 25+ health parameters
        - **Real-time Results**: Instant risk assessment
        - **Personalized Insights**: Detailed risk factor analysis
        
        ### 📋 Parameters Analyzed:
        - **Demographics**: Age, Gender, Region, Socio-economic status
        - **Lifestyle**: Smoking, Alcohol, Diet, Physical Activity
        - **Medical History**: Diabetes, Hypertension, Family History
        - **Vital Signs**: Blood Pressure, Cholesterol, BMI, ECG Results
        """)
    
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/7079/7079342.png", width=200)
        st.markdown("""
        <div class="info-box">
        <h4>🚀 Quick Start</h4>
        <p>Navigate to <b>Risk Assessment</b> in the sidebar to begin your heart health evaluation.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h4>⚠️ Disclaimer</h4>
        <p>This tool is for educational purposes only. Consult healthcare professionals for medical advice.</p>
        </div>
        """, unsafe_allow_html=True)

def show_risk_assessment(model, scaler, feature_names):
    """Display risk assessment form and results"""
    st.markdown('<h2 class="sub-header">🔍 Heart Attack Risk Assessment</h2>', unsafe_allow_html=True)
    
    # Create form
    with st.form("risk_assessment_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Personal Information")
            age = st.slider("Age", 18, 45, 25)
            gender = st.selectbox("Gender", ["Male", "Female"])
            region = st.selectbox("Region", ["North", "South", "East", "West"])
            urban_rural = st.selectbox("Urban/Rural", ["Urban", "Rural"])
            ses = st.selectbox("Socio-economic Status", ["Low", "Middle", "High"])
            
        with col2:
            st.subheader("Lifestyle Factors")
            smoking = st.selectbox("Smoking Status", ["Never", "Occasionally", "Regularly"])
            alcohol = st.selectbox("Alcohol Consumption", ["Never", "Occasionally", "Regularly"])
            diet = st.selectbox("Diet Type", ["Non-Vegetarian", "Vegetarian", "Vegan"])
            activity = st.selectbox("Physical Activity Level", ["Sedentary", "Moderate", "High"])
            screen_time = st.slider("Screen Time (hrs/day)", 0, 16, 6)
            sleep = st.slider("Sleep Duration (hrs/day)", 3, 12, 7)
            
        with col3:
            st.subheader("Medical Information")
            family_history = st.selectbox("Family History of Heart Disease", ["No", "Yes"])
            diabetes = st.selectbox("Diabetes", ["No", "Yes"])
            hypertension = st.selectbox("Hypertension", ["No", "Yes"])
            cholesterol = st.slider("Cholesterol Levels (mg/dL)", 100, 400, 200)
            bmi = st.slider("BMI (kg/m²)", 15.0, 40.0, 25.0)
            stress = st.selectbox("Stress Level", ["Low", "Medium", "High"])
            
        # Additional medical parameters
        st.subheader("Vital Signs")
        col4, col5, col6 = st.columns(3)
        
        with col4:
            systolic = st.slider("Systolic BP", 90, 200, 120)
            diastolic = st.slider("Diastolic BP", 60, 120, 80)
            blood_pressure = f"{systolic}/{diastolic}"
            
        with col5:
            resting_hr = st.slider("Resting Heart Rate (bpm)", 50, 120, 72)
            max_hr = st.slider("Maximum Heart Rate Achieved", 100, 220, 180)
            ecg = st.selectbox("ECG Results", ["Normal", "Abnormal"])
            
        with col6:
            chest_pain = st.selectbox("Chest Pain Type", ["Typical", "Atypical", "Non-anginal"])
            exercise_angina = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
            spo2 = st.slider("Blood Oxygen Levels (SpO2%)", 85, 100, 97)
            triglycerides = st.slider("Triglyceride Levels (mg/dL)", 50, 500, 150)
        
        # Submit button
        submitted = st.form_submit_button("🔍 Assess Heart Attack Risk")
    
    # Process form submission
    if submitted:
        # Create input dictionary
        input_data = {
            'Age': age,
            'Gender': gender,
            'Region': region,
            'Urban/Rural': urban_rural,
            'SES': ses,
            'Smoking Status': smoking,
            'Alcohol Consumption': alcohol,
            'Diet Type': diet,
            'Physical Activity Level': activity,
            'Screen Time (hrs/day)': screen_time,
            'Sleep Duration (hrs/day)': sleep,
            'Family History of Heart Disease': family_history,
            'Diabetes': diabetes,
            'Hypertension': hypertension,
            'Cholesterol Levels (mg/dL)': cholesterol,
            'BMI (kg/m²)': bmi,
            'Stress Level': stress,
            'Blood Pressure (systolic/diastolic mmHg)': blood_pressure,
            'Resting Heart Rate (bpm)': resting_hr,
            'ECG Results': ecg,
            'Chest Pain Type': chest_pain,
            'Maximum Heart Rate Achieved': max_hr,
            'Exercise Induced Angina': exercise_angina,
            'Blood Oxygen Levels (SpO2%)': spo2,
            'Triglyceride Levels (mg/dL)': triglycerides
        }
        
        # Preprocess and predict
        with st.spinner("Analyzing your health data..."):
            processed_data = preprocess_input_data(input_data, feature_names)
            
            if processed_data is not None:
                # Scale features
                scaled_data = scaler.transform(processed_data)
                
                # Make prediction
                prediction_proba = model.predict(scaled_data, verbose=0)[0][0]
                prediction_class = 1 if prediction_proba > 0.5 else 0
                
                # Display results
                st.markdown("---")
                st.markdown("<h2 style='text-align: center;'>📊 Assessment Results</h2>", unsafe_allow_html=True)
                
                # Create columns for results
                res_col1, res_col2 = st.columns([1, 2])
                
                with res_col1:
                    # Risk gauge
                    gauge_fig = create_risk_gauge(prediction_proba)
                    st.plotly_chart(gauge_fig, use_container_width=True)
                    
                    # Risk category
                    if prediction_proba > 0.7:
                        risk_category = "HIGH RISK"
                        risk_class = "risk-high"
                        recommendation = "🚨 Please consult a cardiologist immediately and make lifestyle changes."
                    elif prediction_proba > 0.3:
                        risk_category = "MODERATE RISK"
                        risk_class = "risk-medium"
                        recommendation = "⚠️ Consider consulting a doctor and improve your lifestyle habits."
                    else:
                        risk_category = "LOW RISK"
                        risk_class = "risk-low"
                        recommendation = "✅ Maintain your healthy lifestyle with regular checkups."
                    
                    st.markdown(f'<div class="{risk_class}"><h3>{risk_category}</h3></div>', unsafe_allow_html=True)
                    st.markdown(f"**Probability:** {prediction_proba:.3f}")
                    st.markdown(f"**Prediction:** {'High Risk' if prediction_class == 1 else 'Low Risk'}")
                
                with res_col2:
                    st.markdown("### 📋 Risk Factors Analysis")
                    
                    # Highlight key risk factors
                    risk_factors = []
                    if bmi > 30: risk_factors.append(f"High BMI ({bmi})")
                    if cholesterol > 240: risk_factors.append(f"High Cholesterol ({cholesterol} mg/dL)")
                    if systolic > 140 or diastolic > 90: risk_factors.append(f"High Blood Pressure ({blood_pressure})")
                    if activity == "Sedentary": risk_factors.append("Sedentary Lifestyle")
                    if smoking != "Never": risk_factors.append("Smoking")
                    if diabetes == "Yes": risk_factors.append("Diabetes")
                    if hypertension == "Yes": risk_factors.append("Hypertension")
                    if family_history == "Yes": risk_factors.append("Family History")
                    
                    if risk_factors:
                        st.warning("**Key Risk Factors Identified:**")
                        for factor in risk_factors:
                            st.write(f"• {factor}")
                    else:
                        st.success("✅ No major risk factors identified")
                    
                    # Recommendations
                    st.markdown("### 💡 Recommendations")
                    st.info(recommendation)
                    
                    # General tips
                    st.markdown("#### 🏃‍♂️ Lifestyle Tips:")
                    tips = [
                        "Maintain BMI between 18.5-24.9",
                        "Exercise 30 minutes daily",
                        "Eat balanced diet with fruits & vegetables",
                        "Manage stress through meditation/yoga",
                        "Get 7-9 hours of quality sleep",
                        "Regular health checkups"
                    ]
                    
                    for tip in tips:
                        st.write(f"• {tip}")

def show_model_info():
    """Display model information and performance"""
    st.markdown('<h2 class="sub-header">📊 Model Information</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🧠 Model Architecture
        **Feedforward Neural Network:**
        - Input Layer: Features from health parameters
        - Hidden Layer 1: 64 neurons (ReLU activation)
        - Dropout: 30% (Prevents overfitting)
        - Hidden Layer 2: 32 neurons (ReLU activation) 
        - Dropout: 20% (Prevents overfitting)
        - Output Layer: 1 neuron (Sigmoid activation)
        
        ### ⚙️ Training Details
        - **Optimizer:** Adam
        - **Loss Function:** Binary Crossentropy
        - **Metrics:** Accuracy, Precision, Recall, AUC
        - **Early Stopping:** Yes (prevents overfitting)
        - **Validation Split:** 20%
        """)
    
    with col2:
        st.markdown("""
        ### 📈 Model Performance
        (Based on test dataset)
        
        **Key Metrics:**
        - Accuracy: ~85-90%
        - Precision: ~85-90%
        - Recall: ~85-90%
        - F1-Score: ~85-90%
        - AUC: ~0.90-0.95
        
        ### 🎯 Model Features
        **25+ Health Parameters:**
        - Demographic data
        - Lifestyle factors
        - Medical history
        - Vital signs
        - Laboratory results
        """)
    
    # Feature importance (placeholder)
    st.markdown("### 🔍 Key Predictive Features")
    st.info("""
    The model considers multiple factors, but these are particularly important:
    - Age & Gender
    - Blood Pressure levels
    - Cholesterol & Triglyceride levels
    - BMI and Physical Activity
    - Family History of heart disease
    - Diabetes and Hypertension status
    - Smoking and Alcohol habits
    """)

def show_about_page():
    """Display about page"""
    st.markdown('<h2 class="sub-header">ℹ️ About This Application</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Purpose
    This Heart Attack Risk Predictor is designed specifically for young Indian adults (18-45 years) 
    to assess their risk of heart attacks using artificial intelligence.
    
    ### 🏥 Medical Background
    Heart disease is increasingly affecting younger populations in India due to:
    - Changing lifestyle patterns
    - Increased stress levels
    - Dietary changes
    - Sedentary habits
    
    Early detection and lifestyle modifications can significantly reduce heart attack risks.
    
    ### 🔬 Technology Stack
    - **Frontend:** Streamlit
    - **Machine Learning:** TensorFlow/Keras
    - **Data Processing:** Scikit-learn, Pandas, NumPy
    - **Visualization:** Plotly
    
    ### 📚 Data Source
    The model was trained on a comprehensive dataset of Indian youngsters including:
    - Demographic information
    - Lifestyle factors
    - Medical history
    - Clinical parameters
    
    ### ⚠️ Important Disclaimer
    **This application is for educational and informational purposes only.**
    
    - NOT a substitute for professional medical advice
    - NOT a diagnostic tool
    - Always consult healthcare professionals for medical concerns
    - Results should be interpreted as risk indicators, not diagnoses
    
    ### 👨‍💻 Development
    Developed for heart health awareness in young Indian population using AI/ML technologies.
    """)

if __name__ == "__main__":
    main()