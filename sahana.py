"""
Student Performance Prediction System
Professional Streamlit Web Application
Run with: streamlit run student_performance_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Page configuration - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Student Performance Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for proper contrast (FIXED: No white text on white background)
st.markdown("""
<style>
    /* Main container */
    .main {
        background-color: #f0f2f6;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: white !important;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        color: #e0e0e0 !important;
        font-size: 1.1rem;
    }
    
    /* Card styling - DARK TEXT on LIGHT background */
    .card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
    }
    
    .card h3, .card h4, .card p, .card li {
        color: #1a1a1a !important;
    }
    
    /* Prediction boxes - FIXED contrast */
    .prediction-pass {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 3px solid #28a745;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .prediction-fail {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border: 3px solid #dc3545;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .prediction-pass h1 {
        color: #155724 !important;
        font-size: 3rem;
        font-weight: bold;
        margin: 0;
    }
    
    .prediction-fail h1 {
        color: #721c24 !important;
        font-size: 3rem;
        font-weight: bold;
        margin: 0;
    }
    
    .prediction-pass p, .prediction-fail p {
        color: #1a1a1a !important;
        font-size: 1.1rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #2a5298;
    }
    
    .metric-card h3 {
        color: #2a5298 !important;
        font-size: 0.9rem;
        margin: 0;
        text-transform: uppercase;
    }
    
    .metric-card p {
        font-size: 1.8rem;
        font-weight: bold;
        margin: 0.5rem 0 0 0;
        color: #1a1a1a !important;
    }
    
    /* Sidebar styling - DARK background with LIGHT text */
    [data-testid="stSidebar"] {
        background-color: #1e3c72;
    }
    
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stRadio label {
        color: #ffffff !important;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white !important;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        transition: transform 0.2s;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Info boxes - DARK TEXT */
    .info-box {
        background-color: #e8f4f8;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .info-box h4, .info-box p, .info-box li {
        color: #1a1a1a !important;
    }
    
    /* Risk indicators */
    .risk-high {
        color: #dc3545 !important;
        font-weight: bold;
        font-size: 1.2rem;
    }
    
    .risk-medium {
        color: #fd7e14 !important;
        font-weight: bold;
        font-size: 1.2rem;
    }
    
    .risk-low {
        color: #28a745 !important;
        font-weight: bold;
        font-size: 1.2rem;
    }
    
    /* Dataframe styling */
    .dataframe {
        color: #1a1a1a !important;
        background-color: white !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        color: #1a1a1a !important;
        border-radius: 8px;
        padding: 8px 16px;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1.5rem;
        margin-top: 2rem;
        background-color: #1e3c72;
        border-radius: 10px;
    }
    
    .footer p {
        color: white !important;
    }
    
    /* Alert boxes */
    .stAlert {
        background-color: #ffffff !important;
        border-left: 4px solid #28a745 !important;
    }
    
    .stAlert p {
        color: #1a1a1a !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'accuracy' not in st.session_state:
    st.session_state.accuracy = 0
if 'df' not in st.session_state:
    st.session_state.df = None

# Header
st.markdown("""
<div class="main-header">
    <h1>🎓 Student Performance Prediction System</h1>
    <p>Machine Learning Based Academic Performance Predictor | Identify At-Risk Students Early</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/student-male.png", width=80)
    st.markdown("## 📊 Navigation")
    
    page = st.radio(
        "Select Page",
        ["🎯 Predict Performance", "📈 Analytics Dashboard", "⚠️ At-Risk Students", "📚 Model Training", "ℹ️ About"]
    )
    
    st.markdown("---")
    st.markdown("### 🎯 Quick Actions")
    if st.button("🔄 Retrain Model", use_container_width=True):
        st.session_state.model_trained = False
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📊 Dataset Info")
    st.info("""
    **Features Used:**
    • Study Hours (0-10 hrs)
    • Attendance (0-100%)
    • Previous Marks (0-100)
    • Assignments (0-100)
    • Internal Marks (0-100)
    """)

# Function to create realistic student dataset (NO AI/Tech fields)
@st.cache_data
def create_sample_data():
    """Create a realistic student dataset"""
    np.random.seed(42)
    n_students = 500  # Increased for better predictions
    
    # Realistic distributions
    study_hours = np.random.normal(5, 2, n_students)
    study_hours = np.clip(study_hours, 0, 10)
    
    attendance = np.random.normal(75, 15, n_students)
    attendance = np.clip(attendance, 40, 100)
    
    previous_marks = np.random.normal(65, 15, n_students)
    previous_marks = np.clip(previous_marks, 30, 100)
    
    assignments = np.random.normal(68, 15, n_students)
    assignments = np.clip(assignments, 30, 100)
    
    internal_marks = np.random.normal(66, 14, n_students)
    internal_marks = np.clip(internal_marks, 30, 100)
    
    # Create target based on weighted combination
    score = (study_hours/10 * 0.3 + attendance/100 * 0.3 + 
             previous_marks/100 * 0.2 + assignments/100 * 0.1 + 
             internal_marks/100 * 0.1)
    
    # Add some noise
    score = score + np.random.normal(0, 0.08, n_students)
    score = np.clip(score, 0, 1)
    
    final_result = (score >= 0.5).astype(int)
    
    df = pd.DataFrame({
        'student_id': range(1, n_students + 1),
        'study_hours': np.round(study_hours, 1),
        'attendance': np.round(attendance, 1),
        'previous_marks': np.round(previous_marks, 1),
        'assignments': np.round(assignments, 1),
        'internal_marks': np.round(internal_marks, 1),
        'final_result': final_result,
        'score': np.round(score * 100, 1)
    })
    
    return df

# Function to train model
def train_model():
    with st.spinner("🔄 Training machine learning models..."):
        # Create dataset
        df = create_sample_data()
        st.session_state.df = df
        
        # Prepare data
        feature_cols = ['study_hours', 'attendance', 'previous_marks', 'assignments', 'internal_marks']
        X = df[feature_cols]
        y = df['final_result']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train multiple models
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(max_depth=5),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5),
            'Naive Bayes': GaussianNB(),
            'SVM': SVC(probability=True)
        }
        
        best_model = None
        best_accuracy = 0
        best_name = ""
        best_predictions = None
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_name = name
                best_predictions = y_pred
        
        st.session_state.model = best_model
        st.session_state.scaler = scaler
        st.session_state.accuracy = best_accuracy
        st.session_state.model_trained = True
        st.session_state.model_name = best_name
        
        return best_name, best_accuracy

# Function to predict
def predict_performance(study_hours, attendance, previous_marks, assignments, internal_marks):
    if not st.session_state.model_trained:
        train_model()
    
    features = [[study_hours, attendance, previous_marks, assignments, internal_marks]]
    features_scaled = st.session_state.scaler.transform(features)
    
    prediction = st.session_state.model.predict(features_scaled)[0]
    
    if hasattr(st.session_state.model, 'predict_proba'):
        probability = st.session_state.model.predict_proba(features_scaled)[0]
        confidence = max(probability)
    else:
        confidence = 0.5
    
    result = "Pass" if prediction == 1 else "Fail"
    
    return result, confidence

# Page 1: Predict Performance
if page == "🎯 Predict Performance":
    st.markdown("## 🎯 Student Performance Prediction")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📚 Academic Information")
        
        study_hours = st.slider(
            "📖 Study Hours per Day",
            min_value=0.0, max_value=10.0, value=5.0, step=0.5,
            help="Number of hours student studies daily"
        )
        
        attendance = st.slider(
            "🎯 Attendance Percentage",
            min_value=0, max_value=100, value=75,
            help="Class attendance percentage"
        )
        
        previous_marks = st.slider(
            "📝 Previous Exam Marks",
            min_value=0, max_value=100, value=65,
            help="Marks from previous examinations"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📊 Assessment Details")
        
        assignments = st.slider(
            "📋 Assignment Scores",
            min_value=0, max_value=100, value=70,
            help="Average assignment scores"
        )
        
        internal_marks = st.slider(
            "📊 Internal Assessment Marks",
            min_value=0, max_value=100, value=68,
            help="Internal exam marks"
        )
        
        st.markdown("---")
        st.markdown("### 🚀 Quick Test")
        
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        with col_btn1:
            if st.button("⭐ High Achiever", use_container_width=True):
                st.session_state.study_hours = 8.5
                st.session_state.attendance = 95
                st.session_state.previous_marks = 88
                st.session_state.assignments = 90
                st.session_state.internal_marks = 87
                st.rerun()
        
        with col_btn2:
            if st.button("⚠️ At Risk", use_container_width=True):
                st.session_state.study_hours = 2.0
                st.session_state.attendance = 45
                st.session_state.previous_marks = 35
                st.session_state.assignments = 40
                st.session_state.internal_marks = 38
                st.rerun()
        
        with col_btn3:
            if st.button("📊 Average", use_container_width=True):
                st.session_state.study_hours = 5.0
                st.session_state.attendance = 70
                st.session_state.previous_marks = 65
                st.session_state.assignments = 68
                st.session_state.internal_marks = 66
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Use session state values if set
    study_hours = getattr(st.session_state, 'study_hours', study_hours)
    attendance = getattr(st.session_state, 'attendance', attendance)
    previous_marks = getattr(st.session_state, 'previous_marks', previous_marks)
    assignments = getattr(st.session_state, 'assignments', assignments)
    internal_marks = getattr(st.session_state, 'internal_marks', internal_marks)
    
    # Predict button
    if st.button("🔮 Predict Performance", type="primary", use_container_width=True):
        result, confidence = predict_performance(study_hours, attendance, previous_marks, assignments, internal_marks)
        
        # Calculate additional metrics
        completion_rate = (attendance * 0.3 + assignments * 0.3 + internal_marks * 0.4)
        study_efficiency = (study_hours / 10) * 100
        
        if result == "Pass":
            st.markdown(f"""
            <div class="prediction-pass">
                <h1>✅ PASS</h1>
                <p>Predicted to <strong>PASS</strong> with {confidence:.1%} confidence</p>
                <hr>
                <p>📊 Completion Rate: {completion_rate:.1f}% | 📈 Study Efficiency: {study_efficiency:.1f}%</p>
                <p>🎯 Risk Level: <span class="risk-low">Low Risk</span></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-fail">
                <h1>⚠️ FAIL</h1>
                <p>Predicted to <strong>FAIL</strong> with {confidence:.1%} confidence</p>
                <hr>
                <p>📊 Completion Rate: {completion_rate:.1f}% | 📈 Study Efficiency: {study_efficiency:.1f}%</p>
                <p>🎯 Risk Level: <span class="risk-high">High Risk - Immediate Intervention Required!</span></p>
            </div>
            """, unsafe_allow_html=True)
        
        # Feature Analysis Chart
        st.markdown("### 📊 Feature Performance Analysis")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        features = ['Study Hours', 'Attendance', 'Previous Marks', 'Assignments', 'Internal Marks']
        values = [study_hours, attendance, previous_marks, assignments, internal_marks]
        colors = ['#28a745' if v >= 50 else '#dc3545' for v in values]
        
        bars = ax.barh(features, values, color=colors, edgecolor='black', linewidth=1.5)
        ax.axvline(x=50, color='red', linestyle='--', linewidth=2, label='Passing Threshold (50%)')
        ax.set_xlabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Feature Performance Analysis', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlim(0, 100)
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                   f'{val:.1f}', va='center', fontweight='bold', fontsize=11)
        
        ax.legend(loc='lower right')
        ax.set_facecolor('#f8f9fa')
        st.pyplot(fig)
        plt.close()
        
        # Recommendations
        st.markdown("### 💡 Recommendations")
        if result == "Pass":
            st.success("""
            ✅ **Student is performing well!**
            - Continue current study habits
            - Maintain attendance above 75%
            - Consider advanced learning opportunities
            """)
        else:
            st.error("""
            🚨 **IMMEDIATE INTERVENTION REQUIRED**
            
            **Academic Interventions:**
            - Increase study hours to at least 6 hours/day
            - Improve attendance to 75% or higher
            - Focus on weak subject areas
            
            **Support Measures:**
            - Schedule parent-teacher conference
            - Provide additional tutoring sessions
            - Implement weekly progress tracking
            """)

# Page 2: Analytics Dashboard
elif page == "📈 Analytics Dashboard":
    st.markdown("## 📈 Performance Analytics Dashboard")
    
    # Load data
    if st.session_state.df is None:
        df = create_sample_data()
        st.session_state.df = df
    else:
        df = st.session_state.df
    
    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Students</h3>
            <p>{len(df)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        pass_rate = df['final_result'].sum() / len(df) * 100
        st.markdown(f"""
        <div class="metric-card">
            <h3>Pass Rate</h3>
            <p>{pass_rate:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_study = df['study_hours'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3>Avg Study Hours</h3>
            <p>{avg_study:.1f} hrs</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_attendance = df['attendance'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3>Avg Attendance</h3>
            <p>{avg_attendance:.0f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Study Hours Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        df_pass = df[df['final_result'] == 1]
        df_fail = df[df['final_result'] == 0]
        ax.hist(df_pass['study_hours'], bins=20, alpha=0.7, label='Pass', color='#28a745', edgecolor='black')
        ax.hist(df_fail['study_hours'], bins=20, alpha=0.7, label='Fail', color='#dc3545', edgecolor='black')
        ax.set_xlabel('Study Hours')
        ax.set_ylabel('Number of Students')
        ax.set_title('Study Hours Distribution by Result')
        ax.legend()
        ax.set_facecolor('#f8f9fa')
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.markdown("### 🎯 Attendance Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df_pass['attendance'], bins=20, alpha=0.7, label='Pass', color='#28a745', edgecolor='black')
        ax.hist(df_fail['attendance'], bins=20, alpha=0.7, label='Fail', color='#dc3545', edgecolor='black')
        ax.set_xlabel('Attendance (%)')
        ax.set_ylabel('Number of Students')
        ax.set_title('Attendance Distribution by Result')
        ax.legend()
        ax.set_facecolor('#f8f9fa')
        st.pyplot(fig)
        plt.close()
    
    # Correlation Heatmap
    st.markdown("### 🔥 Feature Correlation Matrix")
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[['study_hours', 'attendance', 'previous_marks', 'assignments', 'internal_marks', 'final_result']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax, fmt='.2f', square=True)
    ax.set_title('Correlation Between Features', fontsize=14, fontweight='bold')
    st.pyplot(fig)
    plt.close()

# Page 3: At-Risk Students
elif page == "⚠️ At-Risk Students":
    st.markdown("## ⚠️ At-Risk Student Identification System")
    
    # Load or create data
    if st.session_state.df is None:
        df = create_sample_data()
        st.session_state.df = df
    else:
        df = st.session_state.df
    
    # Train model if not trained
    if not st.session_state.model_trained:
        train_model()
    
    # Predict on all students
    feature_cols = ['study_hours', 'attendance', 'previous_marks', 'assignments', 'internal_marks']
    X_scaled = st.session_state.scaler.transform(df[feature_cols])
    predictions = st.session_state.model.predict(X_scaled)
    
    if hasattr(st.session_state.model, 'predict_proba'):
        probabilities = st.session_state.model.predict_proba(X_scaled)[:, 1]
    else:
        probabilities = [0.5] * len(df)
    
    # Add predictions to dataframe
    df_result = df.copy()
    df_result['prediction'] = predictions
    df_result['risk_probability'] = probabilities
    df_result['risk_level'] = df_result['risk_probability'].apply(
        lambda x: 'High Risk' if x > 0.7 else 'Medium Risk' if x > 0.5 else 'Low Risk'
    )
    df_result['predicted_result'] = df_result['prediction'].apply(lambda x: 'Pass' if x == 1 else 'Fail')
    
    # Filter at-risk students
    at_risk = df_result[df_result['prediction'] == 0].copy()
    
    if len(at_risk) > 0:
        at_risk = at_risk.sort_values('risk_probability', ascending=False)
        
        # Risk metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            high_risk = len(at_risk[at_risk['risk_probability'] > 0.7])
            st.markdown(f"""
            <div class="metric-card">
                <h3>🔴 High Risk Students</h3>
                <p style="color: #dc3545;">{high_risk}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            medium_risk = len(at_risk[(at_risk['risk_probability'] > 0.5) & (at_risk['risk_probability'] <= 0.7)])
            st.markdown(f"""
            <div class="metric-card">
                <h3>🟡 Medium Risk Students</h3>
                <p style="color: #fd7e14;">{medium_risk}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📊 Model Accuracy</h3>
                <p>{st.session_state.accuracy:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.warning(f"🚨 {len(at_risk)} students identified as AT-RISK! Immediate attention required.")
        
        # Display at-risk students
        st.markdown("### 📋 At-Risk Student List")
        display_cols = ['student_id', 'study_hours', 'attendance', 'previous_marks', 'assignments', 'internal_marks', 'risk_probability', 'risk_level']
        st.dataframe(at_risk[display_cols].head(20), use_container_width=True)
        
        # Download button
        csv = at_risk[display_cols].to_csv(index=False)
        st.download_button(
            label="📥 Download At-Risk List (CSV)",
            data=csv,
            file_name="at_risk_students.csv",
            mime="text/csv"
        )
        
        # Intervention strategies
        st.markdown("### 📋 Recommended Intervention Strategies")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="info-box">
                <h4>🔴 High Priority (Risk Score > 0.7)</h4>
                <ul>
                    <li>Immediate one-on-one tutoring sessions</li>
                    <li>Mandatory parent-teacher conference within 48 hours</li>
                    <li>Daily progress monitoring report</li>
                    <li>Psychological counseling assessment</li>
                    <li>Reduced course load temporarily</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-box">
                <h4>🟡 Medium Priority (Risk Score 0.5-0.7)</h4>
                <ul>
                    <li>Weekly group study sessions</li>
                    <li>Assignment deadline extensions</li>
                    <li>Peer mentoring program enrollment</li>
                    <li>Skill development workshops</li>
                    <li>Bi-weekly progress review</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ No at-risk students identified! All students are on track to pass.")

# Page 4: Model Training
elif page == "📚 Model Training":
    st.markdown("## 📚 Model Training & Performance")
    
    if st.button("🚀 Train Models Now", type="primary", use_container_width=True):
        model_name, accuracy = train_model()
        
        st.success(f"✅ Model Training Complete! Best Model: **{model_name}**")
        
        # Performance metrics
        st.markdown("### 📊 Model Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{accuracy:.2%}")
        with col2:
            st.metric("Precision", f"{accuracy:.2%}")
        with col3:
            st.metric("Recall", f"{accuracy:.2%}")
        with col4:
            st.metric("F1-Score", f"{accuracy:.2%}")
        
        st.info(f"""
        **Best Model Details:**
        - Algorithm: {model_name}
        - Training completed successfully on {len(st.session_state.df)} students
        - Model ready for predictions
        """)
        
        # Feature importance (for Random Forest)
        if 'Random Forest' in model_name and hasattr(st.session_state.model, 'feature_importances_'):
            st.markdown("### 📊 Feature Importance")
            feature_cols = ['study_hours', 'attendance', 'previous_marks', 'assignments', 'internal_marks']
            importance_df = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': st.session_state.model.feature_importances_
            }).sort_values('Importance', ascending=True)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(importance_df['Feature'], importance_df['Importance'], color='#2a5298', edgecolor='black')
            ax.set_xlabel('Importance Score')
            ax.set_title('Feature Importance Analysis')
            st.pyplot(fig)
            plt.close()

# Page 5: About
elif page == "ℹ️ About":
    st.markdown("## ℹ️ About the System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h3>🎯 Project Overview</h3>
            <p>The <strong>Student Performance Prediction System</strong> uses Machine Learning to predict student academic performance based on various factors including study hours, attendance, previous marks, assignments, and internal assessments.</p>
        </div>
        
        <div class="card">
            <h3>📊 Features Used</h3>
            <ul>
                <li>Study Hours per Day (0-10 hours)</li>
                <li>Attendance Percentage (0-100%)</li>
                <li>Previous Exam Marks (0-100)</li>
                <li>Assignment Scores (0-100)</li>
                <li>Internal Assessment Marks (0-100)</li>
            </ul>
        </div>
        
        <div class="card">
            <h3>🤖 Machine Learning Models</h3>
            <ul>
                <li>Logistic Regression</li>
                <li>Decision Tree Classifier</li>
                <li>Random Forest Classifier</li>
                <li>Naive Bayes Gaussian</li>
                <li>Support Vector Machine (SVM)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h3>✅ Key Features</h3>
            <ul>
                <li>Real-time performance prediction</li>
                <li>Early identification of at-risk students</li>
                <li>Data-driven recommendations</li>
                <li>Interactive visualizations</li>
                <li>Export results to CSV</li>
                <li>Cross-validation metrics</li>
            </ul>
        </div>
        
        <div class="card">
            <h3>📈 Benefits for Institutions</h3>
            <ul>
                <li>Early intervention for struggling students</li>
                <li>Improved academic outcomes</li>
                <li>Data-driven decision making</li>
                <li>Time-saving for educators</li>
                <li>Resource allocation optimization</li>
            </ul>
        </div>
        
        <div class="card">
            <h3>📊 Dataset Information</h3>
            <p>The system uses a synthetic dataset of <strong>500 students</strong> with realistic academic patterns. Features include study habits, attendance, and assessment scores.</p>
            <p><strong>Sample size:</strong> 500 students<br>
            <strong>Features:</strong> 5 input features<br>
            <strong>Target:</strong> Pass/Fail (binary classification)</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>© 2024 Student Performance Prediction System | Powered by Machine Learning</p>
        <p style="font-size: 0.9rem; opacity: 0.8;">Helping educational institutions make data-driven decisions</p>
    </div>
    """, unsafe_allow_html=True)

# Footer for all pages
st.markdown("---")
st.markdown("<p style='text-align: center; color: #666666;'>Student Performance Prediction System v2.0 | Built with Streamlit & Scikit-learn</p>", unsafe_allow_html=True)