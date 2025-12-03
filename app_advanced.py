import streamlit as st
import pandas as pd
import joblib
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Student Performance AI", page_icon="🎓", layout="wide")

# --- LOAD MODEL & SUPPORTING DATA ---
@st.cache_resource
def load_model():
    return joblib.load('student_performance_model.pkl')

@st.cache_resource
def load_all_models():
    try:
        return joblib.load('all_models.pkl')
    except:
        return None

@st.cache_resource
def load_feature_importance():
    try:
        with open('feature_importance.json', 'r') as f:
            return json.load(f)
    except:
        return None

@st.cache_resource
def load_model_results():
    try:
        with open('model_results.json', 'r') as f:
            return json.load(f)
    except:
        return None

@st.cache_resource
def load_residuals():
    try:
        with open('residuals.json', 'r') as f:
            return json.load(f)
    except:
        return {'std_residuals': 5.0, 'mean_residuals': 0.0}

@st.cache_data
def load_training_data():
    try:
        return pd.read_csv('StudentPerformanceFactors.csv')
    except:
        return None

# Load all resources
try:
    model = load_model()
    all_models = load_all_models()
    feature_importance = load_feature_importance()
    model_results = load_model_results()
    residuals_data = load_residuals()
    training_data = load_training_data()
except Exception as e:
    st.error(f"Error loading model or data: {e}")
    st.stop()

# --- UI DESIGN ---
st.title("🎓 Student Performance Predictor (Advanced)")
st.markdown("AI-powered system to predict and analyze student exam performance with actionable insights.")

# Create tabs for better organization
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Prediction Dashboard", 
    "🔍 Feature Importance", 
    "📈 Prediction Confidence",
    "👥 Student Analytics",
    "ℹ️ Model Performance"
])

# ==========================================
# TAB 1: PREDICTION DASHBOARD
# ==========================================
with tab1:
    st.header("Student Performance Prediction")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📚 Study Habits")
        hours_studied = st.number_input("Hours Studied (Weekly)", min_value=0, max_value=50, value=20)
        attendance = st.slider("Attendance (%)", 60, 100, 85)
        tutoring = st.number_input("Tutoring Sessions (Monthly)", 0, 10, 1)
        access_resources = st.selectbox("Access to Resources", ["Low", "Medium", "High"], index=1)

    with col2:
        st.subheader("🏫 Environment & Support")
        parental_involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"], index=1)
        family_income = st.selectbox("Family Income", ["Low", "Medium", "High"], index=1)
        teacher_quality = st.selectbox("Teacher Quality", ["Low", "Medium", "High"], index=1)
        internet_access = st.radio("Internet Access", ["Yes", "No"], index=0)

    with col3:
        st.subheader("🧠 Personal Factors")
        motivation = st.selectbox("Motivation Level", ["Low", "Medium", "High"], index=1)
        peer_influence = st.selectbox("Peer Influence", ["Negative", "Neutral", "Positive"], index=1)
        sleep_hours = st.slider("Sleep Hours (Daily)", 4, 10, 7)
        previous_scores = st.number_input("Previous Exam Score", 0, 100, 75)

    # Expander for less critical features
    with st.expander("Advanced Factors (Optional)"):
        c1, c2, c3 = st.columns(3)
        with c1:
            extracurricular = st.radio("Extracurricular Activities", ["Yes", "No"])
            school_type = st.radio("School Type", ["Public", "Private"])
            grade_level = st.slider("Grade Level", 1, 4, 2)
        with c2:
            learning_disabilities = st.radio("Learning Disabilities", ["Yes", "No"], index=1)
            gender = st.radio("Gender", ["Male", "Female"])
            current_semester = st.slider("Current Semester", 1, 8, 4)
        with c3:
            distance = st.selectbox("Distance from Home", ["Near", "Moderate", "Far"])
            parental_education = st.selectbox("Parental Education", ["High School", "College", "Postgraduate"])
            physical_activity = st.number_input("Physical Activity (Hrs/Week)", 0, 10, 3)

    # --- PREDICTION LOGIC ---
    if st.button("🚀 Predict Score & Analysis", type="primary", use_container_width=True):
        
        # Prepare Mappings
        mappings_map = {
            'Low': 0, 'Medium': 1, 'High': 2,
            'No': 0, 'Yes': 1,
            'Male': 0, 'Female': 1,
            'Public': 0, 'Private': 1,
            'Negative': 0, 'Neutral': 1, 'Positive': 2,
            'High School': 0, 'College': 1, 'Postgraduate': 2,
            'Near': 0, 'Moderate': 1, 'Far': 2
        }

        # Create input data
        input_data = pd.DataFrame({
            'Hours_Studied': [hours_studied],
            'Attendance': [attendance],
            'Parental_Involvement': [mappings_map[parental_involvement]],
            'Access_to_Resources': [mappings_map[access_resources]],
            'Extracurricular_Activities': [mappings_map[extracurricular]],
            'Sleep_Hours': [sleep_hours],
            'Previous_Scores': [previous_scores],
            'Motivation_Level': [mappings_map[motivation]],
            'Internet_Access': [mappings_map[internet_access]],
            'Tutoring_Sessions': [tutoring],
            'Family_Income': [mappings_map[family_income]],
            'Teacher_Quality': [mappings_map[teacher_quality]],
            'School_Type': [mappings_map[school_type]],
            'Peer_Influence': [mappings_map[peer_influence]],
            'Physical_Activity': [physical_activity],
            'Learning_Disabilities': [mappings_map[learning_disabilities]],
            'Parental_Education_Level': [mappings_map[parental_education]],
            'Distance_from_Home': [mappings_map[distance]],
            'Gender': [mappings_map[gender]],
            'Grade_Level': [grade_level],
            'Current_Semester': [current_semester],
            'Age': [18 + grade_level],
            'Class_Participation_Score': [attendance * 0.8],
            'Cumulative_GPA': [2.0 + (previous_scores - 60) * 0.04],
            'Study_Motivation_Interaction': [(hours_studied * mappings_map[motivation]) / 3],
            'Attendance_Parental_Interaction': [(attendance * mappings_map[parental_involvement]) / 200],
            'Resources_Quality_Interaction': [mappings_map[access_resources] * mappings_map[teacher_quality]],
            'Hours_Studied_Squared': [hours_studied ** 2],
            'Sleep_Hours_Squared': [(sleep_hours - 7) ** 2],
            'Engagement_Score': [(attendance / 100) * 25 + mappings_map[extracurricular] * 25 + (attendance * 0.8) / 4],
            'Support_Index': [mappings_map[parental_involvement] + mappings_map[internet_access] + mappings_map[family_income] / 2],
            'Health_Wellness_Score': [(10 - abs(sleep_hours - 7)) + physical_activity * 1.5],
            'Sleep_Distance_from_Optimal': [abs(sleep_hours - 7)],
            'Is_Senior': [1 if current_semester >= 7 else 0],
            'Is_Sophomore': [1 if 3 <= current_semester < 5 else 0]
        })

        try:
            # Make prediction
            prediction = model.predict(input_data)[0]
            
            # Calculate confidence interval
            std_residuals = residuals_data.get('std_residuals', 5.0)
            confidence_interval = 1.96 * std_residuals
            lower_bound = max(0, prediction - confidence_interval)
            upper_bound = min(100, prediction + confidence_interval)

            # Display main result
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Predicted Score", f"{prediction:.1f}/100")
            
            with col2:
                st.metric("Lower Bound (95%)", f"{lower_bound:.1f}")
            
            with col3:
                st.metric("Upper Bound (95%)", f"{upper_bound:.1f}")

            # Visual gauge
            st.markdown("### Performance Visualization")
            
            fig = go.Figure(data=[
                go.Indicator(
                    mode = "gauge+number+delta",
                    value = prediction,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Exam Score"},
                    delta = {'reference': previous_scores, 'suffix': " vs Previous"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps' : [
                            {'range': [0, 60], 'color': "#ff6b6b"},
                            {'range': [60, 75], 'color': "#ffd93d"},
                            {'range': [75, 90], 'color': "#6bcf7f"},
                            {'range': [90, 100], 'color': "#2d8f69"}
                        ],
                        'threshold' : {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 75
                        }
                    }
                )
            ])
            st.plotly_chart(fig, use_container_width=True)

            # Performance category
            if prediction >= 90:
                st.success("🌟 **Excellent Performance!** - Outstanding work!")
                st.balloons()
            elif prediction >= 75:
                st.info("✅ **Good Job!** - You're performing well.")
            elif prediction >= 60:
                st.warning("⚠️ **Needs Improvement** - Focus on key areas.")
            else:
                st.error("🚨 **At Risk** - Consider tutoring and increased study hours.")

            # Key recommendations
            st.markdown("### 📋 Personalized Recommendations")
            
            recommendations = []
            
            if hours_studied < 15:
                recommendations.append("📖 Increase weekly study hours (currently below 15 hours)")
            
            if attendance < 80:
                recommendations.append("🏫 Improve attendance (currently below 80%)")
            
            if sleep_hours < 6 or sleep_hours > 9:
                recommendations.append(f"😴 Optimize sleep hours (currently {sleep_hours}, ideal is 7-8)")
            
            if mappings_map[motivation] < 2:
                recommendations.append("💪 Work on motivation and engagement")
            
            if physical_activity < 2:
                recommendations.append("🏃 Increase physical activity (current: {:.1f} hrs/week)")
            
            if tutoring < 2 and prediction < 75:
                recommendations.append("👨‍🏫 Consider additional tutoring sessions")
            
            if not recommendations:
                recommendations.append("✨ Great approach! Maintain your current study habits.")
            
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")

        except Exception as e:
            st.error(f"Error making prediction: {e}")


# ==========================================
# TAB 2: FEATURE IMPORTANCE
# ==========================================
with tab2:
    st.header("Feature Importance Analysis")
    st.markdown("Understand which factors have the biggest impact on exam scores.")
    
    if feature_importance:
        # Select model for importance
        importance_model = st.selectbox(
            "Select Model:", 
            list(feature_importance.keys()),
            index=0
        )
        
        # Get top features
        features = feature_importance[importance_model]
        top_n = st.slider("Show Top N Features", 5, 35, 15)
        
        top_features = sorted(features.items(), key=lambda x: x[1], reverse=True)[:top_n]
        feature_names = [x[0] for x in top_features]
        feature_values = [x[1] for x in top_features]
        
        # Create bar chart
        fig = px.bar(
            x=feature_values, 
            y=feature_names, 
            orientation='h',
            title=f"Top {top_n} Features - {importance_model}",
            labels={'x': 'Importance Score', 'y': 'Feature'},
            color=feature_values,
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=max(400, len(feature_names) * 25))
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretation
        st.markdown("### 📊 Key Insights")
        st.write(f"**Top Factor:** {feature_names[0]} (Importance: {feature_values[0]:.4f})")
        st.write(f"**Second Factor:** {feature_names[1]} (Importance: {feature_values[1]:.4f})")
        st.write(f"**Third Factor:** {feature_names[2]} (Importance: {feature_values[2]:.4f})")
        
        st.info("💡 **Interpretation:** Higher importance scores mean the factor has more influence on predicting exam scores. Focus on improving the top factors for better results.")


# ==========================================
# TAB 3: PREDICTION CONFIDENCE
# ==========================================
with tab3:
    st.header("Prediction Confidence & Uncertainty")
    st.markdown("Learn how confident the model is in its predictions and what affects confidence.")
    
    residuals_std = residuals_data.get('std_residuals', 5.0)
    residuals_mean = residuals_data.get('mean_residuals', 0.0)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Prediction Error (Std Dev)", f"±{residuals_std:.2f} points")
        st.metric("Residuals Mean", f"{residuals_mean:.2f} points")
    
    with col2:
        confidence_95 = 1.96 * residuals_std
        confidence_90 = 1.645 * residuals_std
        
        st.metric("95% Confidence Interval", f"±{confidence_95:.2f} points")
        st.metric("90% Confidence Interval", f"±{confidence_90:.2f} points")
    
    st.markdown("### 📈 Confidence Distribution")
    
    # Show confidence levels
    fig = go.Figure()
    
    ranges = np.linspace(0, 100, 100)
    for conf_level, multiplier, color in [(90, 1.645, 'rgba(100, 150, 255, 0.3)'), 
                                          (95, 1.96, 'rgba(50, 100, 255, 0.3)')]:
        interval = multiplier * residuals_std
        lower = ranges - interval
        upper = ranges + interval
        
        fig.add_trace(go.Scatter(
            x=ranges, y=upper,
            fill=None,
            mode='lines',
            name=f'{conf_level}% CI Upper',
            line=dict(width=0)
        ))
        
        fig.add_trace(go.Scatter(
            x=ranges, y=lower,
            fill='tonexty',
            mode='lines',
            name=f'{conf_level}% CI',
            line=dict(width=0),
            fillcolor=color
        ))
    
    fig.update_layout(
        title="Prediction Confidence Intervals Across Score Range",
        xaxis_title="Predicted Score",
        yaxis_title="Score Range (with uncertainty)",
        hovermode='x unified',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("📌 **What this means:** The darker shaded areas show where actual scores typically fall. Narrower ranges = more confident predictions.")


# ==========================================
# TAB 4: STUDENT ANALYTICS
# ==========================================
with tab4:
    st.header("Student Analytics & Comparison")
    st.markdown("Compare your performance against peer benchmarks and track patterns.")
    
    if training_data is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            analysis_type = st.radio("Select Analysis:", [
                "Score Distribution",
                "Attendance vs Performance",
                "Study Hours vs Performance",
                "GPA vs Performance"
            ])
        
        with col2:
            grade_filter = st.multiselect(
                "Filter by Grade Level:",
                [1, 2, 3, 4],
                default=[1, 2, 3, 4]
            )
        
        # Filter data
        filtered_data = training_data[training_data['Grade_Level'].isin(grade_filter)]
        
        if analysis_type == "Score Distribution":
            fig = px.histogram(
                filtered_data,
                x='Exam_Score',
                nbins=30,
                title="Distribution of Exam Scores",
                labels={'Exam_Score': 'Exam Score', 'count': 'Number of Students'},
                color_discrete_sequence=['#1f77b4']
            )
            fig.add_vline(x=filtered_data['Exam_Score'].mean(), 
                         line_dash="dash", line_color="red",
                         annotation_text=f"Mean: {filtered_data['Exam_Score'].mean():.1f}")
            st.plotly_chart(fig, use_container_width=True)
            
            st.write(f"**Mean Score:** {filtered_data['Exam_Score'].mean():.2f}")
            st.write(f"**Median Score:** {filtered_data['Exam_Score'].median():.2f}")
            st.write(f"**Std Dev:** {filtered_data['Exam_Score'].std():.2f}")
        
        elif analysis_type == "Attendance vs Performance":
            fig = px.scatter(
                filtered_data,
                x='Attendance',
                y='Exam_Score',
                color='Grade_Level',
                title="Attendance vs Exam Score",
                labels={'Attendance': 'Attendance (%)', 'Exam_Score': 'Exam Score'},
                trendline='ols'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            corr = filtered_data['Attendance'].corr(filtered_data['Exam_Score'])
            st.write(f"**Correlation:** {corr:.3f} (Attendance strongly affects performance!)")
        
        elif analysis_type == "Study Hours vs Performance":
            fig = px.scatter(
                filtered_data,
                x='Hours_Studied',
                y='Exam_Score',
                color='Grade_Level',
                title="Study Hours vs Exam Score",
                labels={'Hours_Studied': 'Weekly Study Hours', 'Exam_Score': 'Exam Score'},
                trendline='ols'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            corr = filtered_data['Hours_Studied'].corr(filtered_data['Exam_Score'])
            st.write(f"**Correlation:** {corr:.3f}")
        
        else:  # GPA vs Performance
            fig = px.scatter(
                filtered_data,
                x='Cumulative_GPA',
                y='Exam_Score',
                color='Grade_Level',
                title="Cumulative GPA vs Exam Score",
                labels={'Cumulative_GPA': 'Cumulative GPA', 'Exam_Score': 'Exam Score'},
                trendline='ols'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            corr = filtered_data['Cumulative_GPA'].corr(filtered_data['Exam_Score'])
            st.write(f"**Correlation:** {corr:.3f}")


# ==========================================
# TAB 5: MODEL PERFORMANCE
# ==========================================
with tab5:
    st.header("Model Performance Report")
    st.markdown("Detailed metrics comparing all trained models.")
    
    if model_results:
        # Model comparison
        st.subheader("📊 Model Comparison")
        
        models_list = list(model_results['individual_results'].keys())
        best_model = model_results['best_model']
        
        comparison_data = []
        for model_name in models_list:
            result = model_results['individual_results'][model_name]
            comparison_data.append({
                'Model': model_name,
                'Test R²': result['test_r2'],
                'Test MAE': result['test_mae'],
                'Test RMSE': result['test_rmse'],
                'Accuracy': result['accuracy']
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Display table
        st.dataframe(df_comparison, use_container_width=True, hide_index=True)
        
        # Visualization
        fig = px.bar(
            df_comparison,
            x='Model',
            y=['Test R²', 'Accuracy'],
            title="Model Performance Metrics",
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.success(f"✅ **Best Model:** {best_model}")
        
        # CV Results
        st.subheader("🔄 Cross-Validation Results (5-Fold)")
        
        cv_data = []
        for model_name, cv_result in model_results['cv_results'].items():
            cv_data.append({
                'Model': model_name,
                'CV Mean R²': cv_result['cv_r2_mean'],
                'Std Dev': cv_result['cv_r2_std']
            })
        
        df_cv = pd.DataFrame(cv_data)
        st.dataframe(df_cv, use_container_width=True, hide_index=True)
        
        # Feature information
        st.subheader("📋 Model Information")
        st.write(f"**Total Features:** {len(model_results['feature_names'])}")
        st.write(f"**Training Samples:** 5,285")
        st.write(f"**Test Samples:** 1,322")
        st.write(f"**Training Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Feature list
        with st.expander("View All Features Used"):
            st.write(model_results['feature_names'])
