import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns

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
    .risk-high {
        background-color: #ff4b4b;
        color: white;
        padding: 20px;
        border-radius: 10px;
        font-size: 1.5rem;
        text-align: center;
    }
    .risk-medium {
        background-color: #ffa500;
        color: white;
        padding: 20px;
        border-radius: 10px;
        font-size: 1.5rem;
        text-align: center;
    }
    .risk-low {
        background-color: #00cc66;
        color: white;
        padding: 20px;
        border-radius: 10px;
        font-size: 1.5rem;
        text-align: center;
    }
    .feature-importance {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    """Load model and preprocessing artifacts"""
    try:
        model = keras.models.load_model('heart_attack_model.h5')
        scaler = joblib.load('scaler.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        return model, scaler, feature_columns
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None

def create_feature_vector(input_data, feature_columns):
    """Create feature vector matching training data structure"""
    # Create a dataframe with all feature columns initialized to 0
    feature_df = pd.DataFrame(0, index=[0], columns=feature_columns)
    
    # Map input data to feature columns
    for key, value in input_data.items():
        if key in feature_df.columns:
            feature_df[key] = value
        else:
            # Handle one-hot encoded columns
            for col in feature_df.columns:
                if str(key) in col:
                    feature_df[col] = value
    
    return feature_df

def predict_heart_attack(model, scaler, feature_columns, patient_data):
    """Make prediction for a single patient"""
    try:
        # Create feature vector
        feature_vector = create_feature_vector(patient_data, feature_columns)
        
        # Scale features
        scaled_features = scaler.transform(feature_vector)
        
        # Make prediction
        prediction_proba = model.predict(scaled_features, verbose=0)
        probability = float(prediction_proba[0][0])
        
        return probability, feature_vector
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

def main():
    # Header
    st.markdown('<h1 class="main-header">❤️ Heart Attack Risk Predictor</h1>', unsafe_allow_html=True)
    st.markdown("### Early Detection for Young Adults (18-35 years)")
    
    # Load model
    model, scaler, feature_columns = load_artifacts()
    
    if model is None:
        st.error("Please make sure all model files (heart_attack_model.h5, scaler.pkl, feature_columns.pkl) are in the same directory as this app.")
        return
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📝 Patient Information")
        
        with st.form("patient_form"):
            # Personal Information
            st.subheader("Personal Details")
            age = st.slider("Age", 18, 35, 24)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            
            # Lifestyle Factors
            st.subheader("Lifestyle Factors")
            smoking = st.selectbox("Smoking Status", ["Never", "Occasionally", "Regularly"])
            alcohol = st.selectbox("Alcohol Consumption", ["Never", "Occasionally", "Regularly"])
            physical_activity = st.selectbox("Physical Activity Level", ["Sedentary", "Moderate", "High"])
            screen_time = st.slider("Screen Time (hours/day)", 0, 15, 6)
            sleep_duration = st.slider("Sleep Duration (hours/day)", 3, 12, 7)
            diet_type = st.selectbox("Diet Type", ["Vegetarian", "Non-Vegetarian", "Vegan"])
            stress_level = st.selectbox("Stress Level", ["Low", "Medium", "High"])
            
            # Medical History
            st.subheader("Medical History")
            family_history = st.selectbox("Family History of Heart Disease", ["No", "Yes"])
            diabetes = st.selectbox("Diabetes", ["No", "Yes"])
            hypertension = st.selectbox("Hypertension", ["No", "Yes"])
            
            # Clinical Measurements
            st.subheader("Clinical Measurements")
            cholesterol = st.slider("Cholesterol Levels (mg/dL)", 100, 300, 200)
            bmi = st.slider("BMI (kg/m²)", 15.0, 40.0, 25.0)
            systolic_bp = st.slider("Systolic Blood Pressure (mmHg)", 90, 180, 120)
            diastolic_bp = st.slider("Diastolic Blood Pressure (mmHg)", 60, 120, 80)
            resting_hr = st.slider("Resting Heart Rate (bpm)", 60, 120, 72)
            
            # Region and SES
            region = st.selectbox("Region", ["North", "South", "East", "West", "Central", "North-East"])
            urban_rural = st.selectbox("Urban/Rural", ["Urban", "Rural"])
            ses = st.selectbox("Socioeconomic Status", ["Low", "Middle", "High"])
            
            submitted = st.form_submit_button("🔍 Predict Heart Attack Risk", use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Prediction Results")
        
        if submitted:
            with st.spinner("Analyzing patient data..."):
                # Prepare input data
                input_data = {
                    'Age': age,
                    f'Gender_{gender}': 1,
                    f'Smoking Status_{smoking}': 1,
                    f'Alcohol Consumption_{alcohol}': 1,
                    f'Physical Activity Level_{physical_activity}': 1,
                    'Screen Time (hrs/day)': screen_time,
                    'Sleep Duration (hrs/day)': sleep_duration,
                    f'Diet Type_{diet_type}': 1,
                    f'Stress Level_{stress_level}': 1,
                    f'Family History of Heart Disease_{family_history}': 1,
                    f'Diabetes_{diabetes}': 1,
                    f'Hypertension_{hypertension}': 1,
                    'Cholesterol Levels (mg/dL)': cholesterol,
                    'BMI (kg/m²)': bmi,
                    f'Region_{region}': 1,
                    f'Urban/Rural_{urban_rural}': 1,
                    f'SES_{ses}': 1
                }
                
                # Add blood pressure (handle based on your actual feature names)
                input_data['Blood Pressure (systolic/diastolic mmHg)'] = f"{systolic_bp}/{diastolic_bp}"
                input_data['Resting Heart Rate (bpm)'] = resting_hr
                
                # Make prediction
                probability, feature_vector = predict_heart_attack(model, scaler, feature_columns, input_data)
                
                if probability is not None:
                    # Display results
                    st.markdown("---")
                    
                    # Risk level determination
                    if probability >= 0.7:
                        risk_level = "HIGH RISK"
                        risk_class = "risk-high"
                        recommendation = "🚨 Immediate medical consultation recommended. Please consult a cardiologist."
                        emoji = "🔴"
                    elif probability >= 0.4:
                        risk_level = "MEDIUM RISK"
                        risk_class = "risk-medium"
                        recommendation = "⚠️ Regular health monitoring advised. Consider lifestyle changes."
                        emoji = "🟡"
                    else:
                        risk_level = "LOW RISK"
                        risk_class = "risk-low"
                        recommendation = "✅ Maintain healthy lifestyle. Regular checkups recommended."
                        emoji = "🟢"
                    
                    # Display risk box
                    st.markdown(f'<div class="{risk_class}">{emoji} {risk_level} - {probability:.1%} Probability</div>', 
                              unsafe_allow_html=True)
                    
                    # Probability gauge
                    st.subheader("Risk Probability Gauge")
                    gauge_value = probability * 100
                    st.progress(int(gauge_value))
                    st.write(f"**{gauge_value:.1f}%** probability of heart attack risk")
                    
                    # Recommendations
                    st.subheader("💡 Recommendations")
                    st.info(recommendation)
                    
                    # Key risk factors
                    st.subheader("🔍 Key Risk Factors Identified")
                    
                    risk_factors = []
                    if diabetes == "Yes":
                        risk_factors.append("• Diabetes")
                    if hypertension == "Yes":
                        risk_factors.append("• Hypertension")
                    if family_history == "Yes":
                        risk_factors.append("• Family History of Heart Disease")
                    if smoking != "Never":
                        risk_factors.append(f"• Smoking: {smoking}")
                    if stress_level == "High":
                        risk_factors.append("• High Stress Level")
                    if bmi >= 30:
                        risk_factors.append("• High BMI (Obesity)")
                    if cholesterol >= 240:
                        risk_factors.append("• High Cholesterol")
                    if physical_activity == "Sedentary":
                        risk_factors.append("• Sedentary Lifestyle")
                    
                    if risk_factors:
                        for factor in risk_factors:
                            st.write(factor)
                    else:
                        st.write("No major risk factors identified")
                    
                    # Protective factors
                    st.subheader("🛡️ Protective Factors")
                    protective_factors = []
                    if physical_activity == "High":
                        protective_factors.append("• High Physical Activity")
                    if smoking == "Never":
                        protective_factors.append("• Non-smoker")
                    if diet_type == "Vegetarian":
                        protective_factors.append("• Vegetarian Diet")
                    if stress_level == "Low":
                        protective_factors.append("• Low Stress Level")
                    if bmi < 25:
                        protective_factors.append("• Healthy BMI")
                    
                    for factor in protective_factors:
                        st.success(factor)
        
        else:
            # Default view before prediction
            st.info("👆 Fill out the patient information form and click 'Predict Heart Attack Risk' to get started.")
            
            # Sample predictions for demonstration
            st.markdown("---")
            st.subheader("📈 Sample Risk Scenarios")
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric("Low Risk Profile", "15-30%", "Healthy")
                st.caption("Young, active, no family history")
            
            with col_b:
                st.metric("Medium Risk Profile", "31-69%", "Monitor") 
                st.caption("Some risk factors present")
            
            with col_c:
                st.metric("High Risk Profile", "70-95%", "Alert")
                st.caption("Multiple risk factors")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p><em>⚠️ Disclaimer: This tool is for educational purposes only. 
        Always consult healthcare professionals for medical advice.</em></p>
        <p>Built with ❤️ using Streamlit and TensorFlow</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()