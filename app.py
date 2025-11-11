import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(
    page_title="Heart Attack Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
        padding: 25px; 
        border-radius: 15px; 
        font-size: 1.8rem; 
        text-align: center;
        border: 3px solid #cc0000;
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
</style>
""", unsafe_allow_html=True)

def create_fallback_model():
    """Create a simple fallback model for demonstration"""
    model = Sequential([
        Dense(64, activation='relu', input_shape=(48,)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Create compatible scaler
    scaler = StandardScaler()
    scaler.mean_ = np.zeros(48)
    scaler.scale_ = np.ones(48)
    scaler.n_features_in_ = 48
    
    # Define feature columns
    feature_columns = [
        'Age', 'Screen Time (hrs/day)', 'Sleep Duration (hrs/day)', 
        'Cholesterol Levels (mg/dL)', 'BMI (kg/m²)', 'Resting Heart Rate (bpm)',
        'Systolic_BP', 'Diastolic_BP',
        'Gender_Female', 'Gender_Male', 'Gender_Other',
        'Smoking Status_Occasionally', 'Smoking Status_Never', 'Smoking Status_Regularly',
        'Alcohol Consumption_Occasionally', 'Alcohol Consumption_Never', 'Alcohol Consumption_Regularly',
        'Physical Activity Level_Sedentary', 'Physical Activity Level_Moderate', 'Physical Activity Level_High',
        'Diet Type_Vegetarian', 'Diet Type_Non-Vegetarian', 'Diet Type_Vegan',
        'Stress Level_Low', 'Stress Level_Medium', 'Stress Level_High',
        'Family History of Heart Disease_No', 'Family History of Heart Disease_Yes',
        'Diabetes_No', 'Diabetes_Yes',
        'Hypertension_No', 'Hypertension_Yes',
        'Region_North', 'Region_South', 'Region_East', 'Region_West', 'Region_Central', 'Region_North-East',
        'Urban/Rural_Urban', 'Urban/Rural_Rural',
        'SES_Low', 'SES_Middle', 'SES_High'
    ]
    
    return model, scaler, feature_columns

@st.cache_resource
def load_artifacts():
    """Load model and preprocessing artifacts with fallback"""
    try:
        if (os.path.exists('heart_attack_model.h5') and 
            os.path.getsize('heart_attack_model.h5') > 1000):
            model = keras.models.load_model('heart_attack_model.h5')
            scaler = joblib.load('scaler.pkl')
            feature_columns = joblib.load('feature_columns.pkl')
            st.success("✅ Original model loaded successfully!")
            return model, scaler, feature_columns
        else:
            st.warning("Using fallback model for demonstration")
            return create_fallback_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return create_fallback_model()

def prepare_patient_data_simple(form_data):
    """Simplified patient data preparation"""
    # Convert all numerical values to float
    patient_data = {}
    
    # Numerical features
    numerical_features = {
        'Age': form_data['age'],
        'Screen Time (hrs/day)': form_data['screen_time'],
        'Sleep Duration (hrs/day)': form_data['sleep_duration'],
        'Cholesterol Levels (mg/dL)': form_data['cholesterol'],
        'BMI (kg/m²)': form_data['bmi'],
        'Resting Heart Rate (bpm)': form_data['resting_hr'],
        'Systolic_BP': form_data['systolic_bp'],  # Already numerical from slider
        'Diastolic_BP': form_data['diastolic_bp']  # Already numerical from slider
    }
    
    for feature, value in numerical_features.items():
        patient_data[feature] = float(value)
    
    # Categorical features (one-hot encoding)
    categorical_mappings = {
        'gender': ['Female', 'Male', 'Other'],
        'smoking': ['Never', 'Occasionally', 'Regularly'],
        'alcohol': ['Never', 'Occasionally', 'Regularly'],
        'physical_activity': ['Sedentary', 'Moderate', 'High'],
        'diet_type': ['Vegetarian', 'Non-Vegetarian', 'Vegan'],
        'stress_level': ['Low', 'Medium', 'High'],
        'family_history': ['No', 'Yes'],
        'diabetes': ['No', 'Yes'],
        'hypertension': ['No', 'Yes'],
        'region': ['North', 'South', 'East', 'West', 'Central', 'North-East'],
        'urban_rural': ['Urban', 'Rural'],
        'ses': ['Low', 'Middle', 'High']
    }
    
    for field, options in categorical_mappings.items():
        value = form_data[field]
        for option in options:
            feature_name = f"{field.replace('_', ' ').title()}_{option}"
            patient_data[feature_name] = 1 if value == option else 0
    
    return patient_data

def create_feature_vector_safe(patient_data, feature_columns):
    """Safe feature vector creation"""
    feature_df = pd.DataFrame(0.0, index=[0], columns=feature_columns)
    
    mapped = 0
    for feature, value in patient_data.items():
        if feature in feature_df.columns:
            feature_df[feature] = float(value)
            mapped += 1
    
    st.info(f"📊 Mapped {mapped} out of {len(feature_columns)} features")
    return feature_df

def main():
    st.markdown('<h1 class="main-header">❤️ Heart Attack Risk Predictor</h1>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Early Detection for Young Adults (18-35 years)")
    
    model, scaler, feature_columns = load_artifacts()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📝 Patient Information")
        
        with st.form("patient_form"):
            st.subheader("Personal Details")
            col_a, col_b = st.columns(2)
            with col_a:
                age = st.slider("Age", 18, 35, 24)
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                smoking = st.selectbox("Smoking Status", ["Never", "Occasionally", "Regularly"])
                alcohol = st.selectbox("Alcohol Consumption", ["Never", "Occasionally", "Regularly"])
            with col_b:
                physical_activity = st.selectbox("Physical Activity Level", ["Sedentary", "Moderate", "High"])
                diet_type = st.selectbox("Diet Type", ["Vegetarian", "Non-Vegetarian", "Vegan"])
                stress_level = st.selectbox("Stress Level", ["Low", "Medium", "High"])
            
            st.subheader("Lifestyle Factors")
            col_life1, col_life2 = st.columns(2)
            with col_life1:
                screen_time = st.slider("Screen Time (hours/day)", 0, 15, 6)
            with col_life2:
                sleep_duration = st.slider("Sleep Duration (hours/day)", 3, 12, 7)
            
            st.subheader("Medical History")
            col_med1, col_med2 = st.columns(2)
            with col_med1:
                family_history = st.selectbox("Family History of Heart Disease", ["No", "Yes"])
                diabetes = st.selectbox("Diabetes", ["No", "Yes"])
            with col_med2:
                hypertension = st.selectbox("Hypertension", ["No", "Yes"])
            
            st.subheader("Clinical Measurements")
            col_clin1, col_clin2 = st.columns(2)
            with col_clin1:
                cholesterol = st.slider("Cholesterol Levels (mg/dL)", 100, 300, 200)
                bmi = st.slider("BMI (kg/m²)", 15.0, 40.0, 25.0)
            with col_clin2:
                systolic_bp = st.slider("Systolic BP (mmHg)", 90, 180, 120)
                diastolic_bp = st.slider("Diastolic BP (mmHg)", 60, 120, 80)
                resting_hr = st.slider("Resting Heart Rate (bpm)", 60, 120, 72)
            
            st.subheader("Additional Information")
            col_add1, col_add2 = st.columns(2)
            with col_add1:
                region = st.selectbox("Region", ["North", "South", "East", "West", "Central", "North-East"])
                urban_rural = st.selectbox("Urban/Rural", ["Urban", "Rural"])
            with col_add2:
                ses = st.selectbox("Socioeconomic Status", ["Low", "Middle", "High"])
            
            submitted = st.form_submit_button("🔍 Analyze Heart Attack Risk", use_container_width=True, type="primary")
            
            if submitted:
                st.session_state.form_data = {
                    'age': age, 'gender': gender, 'smoking': smoking, 'alcohol': alcohol,
                    'physical_activity': physical_activity, 'screen_time': screen_time,
                    'sleep_duration': sleep_duration, 'diet_type': diet_type, 'stress_level': stress_level,
                    'family_history': family_history, 'diabetes': diabetes, 'hypertension': hypertension,
                    'cholesterol': cholesterol, 'bmi': bmi, 'systolic_bp': systolic_bp,
                    'diastolic_bp': diastolic_bp, 'resting_hr': resting_hr, 'region': region,
                    'urban_rural': urban_rural, 'ses': ses
                }
    
    with col2:
        st.markdown("### 📊 Prediction Results")
        
        if 'form_data' in st.session_state and submitted:
            with st.spinner("🔬 Analyzing patient data..."):
                try:
                    # Prepare data
                    patient_data = prepare_patient_data_simple(st.session_state.form_data)
                    feature_vector = create_feature_vector_safe(patient_data, feature_columns)
                    
                    # Make prediction
                    scaled_features = scaler.transform(feature_vector)
                    prediction_proba = model.predict(scaled_features, verbose=0)
                    probability = float(prediction_proba[0][0])
                    
                    # Display results
                    st.markdown("---")
                    
                    if probability >= 0.7:
                        risk_level, risk_class, recommendation, emoji = "HIGH RISK", "risk-high", "🚨 Immediate medical consultation recommended!", "🔴"
                    elif probability >= 0.4:
                        risk_level, risk_class, recommendation, emoji = "MEDIUM RISK", "risk-medium", "⚠️ Regular health monitoring advised.", "🟡"
                    else:
                        risk_level, risk_class, recommendation, emoji = "LOW RISK", "risk-low", "✅ Maintain healthy lifestyle.", "🟢"
                    
                    st.markdown(f'<div class="{risk_class}">{emoji} {risk_level} - {probability:.1%} Probability</div>', unsafe_allow_html=True)
                    
                    st.subheader("Risk Probability Gauge")
                    st.progress(int(probability * 100))
                    st.write(f"**{probability * 100:.1f}%** probability of heart attack risk")
                    
                    st.info(recommendation)
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"❌ Prediction error: {str(e)}")
        
        elif not submitted:
            st.info("👆 Fill out the form and click 'Analyze Heart Attack Risk'")

if __name__ == "__main__":
    main()
