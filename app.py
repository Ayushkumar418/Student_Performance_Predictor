import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Student Performance AI", page_icon="🎓", layout="wide")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return joblib.load('student_performance_model.pkl')

try:
    model = load_model()
except:
    st.error("Model file not found. Please run 'train.py' first!")
    st.stop()

# --- LOAD CSV DATA ---
@st.cache_data
def load_csv():
    return pd.read_csv('StudentPerformanceFactors.csv')

csv_data = load_csv()

# --- UI DESIGN ---
st.title("🎓 Student Performance Predictor")
st.markdown("Use the controls below to input student details and predict their exam score.")

# Create tabs for better organization
tab1, tab2, tab3 = st.tabs(["📊 Prediction Dashboard", "📈 Next Semester Score", "ℹ️ Model Details"])

with tab1:
    st.header("Student Details")
    
    # We use columns to organize inputs neatly
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Study Habits")
        hours_studied = st.number_input("Hours Studied (Weekly)", min_value=0, max_value=50, value=20)
        attendance = st.slider("Attendance (%)", 60, 100, 85)
        tutoring = st.number_input("Tutoring Sessions (Monthly)", 0, 10, 1)
        access_resources = st.selectbox("Access to Resources", ["Low", "Medium", "High"], index=1)

    with col2:
        st.subheader("Environment & Support")
        parental_involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"], index=1)
        family_income = st.selectbox("Family Income", ["Low", "Medium", "High"], index=1)
        teacher_quality = st.selectbox("Teacher Quality", ["Low", "Medium", "High"], index=1)
        internet_access = st.radio("Internet Access", ["Yes", "No"], index=0)

    with col3:
        st.subheader("Personal Factors")
        motivation = st.selectbox("Motivation Level", ["Low", "Medium", "High"], index=1)
        peer_influence = st.selectbox("Peer Influence", ["Negative", "Neutral", "Positive"], index=1)
        sleep_hours = st.slider("Sleep Hours (Daily)", 4, 10, 7)
        previous_scores = st.number_input("Previous Exam Score", 0, 100, 75)

    # Expander for less critical features to keep UI clean
    with st.expander("Advanced/Secondary Factors (Optional)"):
        c1, c2, c3 = st.columns(3)
        with c1:
            extracurricular = st.radio("Extracurricular Activities", ["Yes", "No"])
            school_type = st.radio("School Type", ["Public", "Private"])
            grade_level = st.slider("Grade Level (Year)", 1, 4, 2)
        with c2:
            learning_disabilities = st.radio("Learning Disabilities", ["Yes", "No"], index=1)
            gender = st.radio("Gender", ["Male", "Female"])
            current_semester = st.slider("Current Semester", 1, 8, 4)
        with c3:
            distance = st.selectbox("Distance from Home", ["Near", "Moderate", "Far"])
            parental_education = st.selectbox("Parental Education", ["High School", "College", "Postgraduate"])
            physical_activity = st.number_input("Physical Activity (Hrs/Week)", 0, 10, 3)
            class_participation = st.slider("Class Participation Score", 0, 100, 70)

    # --- PREDICTION LOGIC ---
    if st.button("🚀 Predict Score", type="primary", use_container_width=True):
        
        # 1. Prepare Mappings (Must match training script exactly)
        mappings_map = {
            'Low': 0, 'Medium': 1, 'High': 2,
            'No': 0, 'Yes': 1,
            'Male': 0, 'Female': 1,
            'Public': 0, 'Private': 1,
            'Negative': 0, 'Neutral': 1, 'Positive': 2,
            'High School': 0, 'College': 1, 'Postgraduate': 2,
            'Near': 0, 'Moderate': 1, 'Far': 2
        }

        # 2. Create DataFrame for Model
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
            'Class_Participation_Score': [class_participation],
            'Cumulative_GPA': [2.0 + (previous_scores - 60) * 0.04],
            'Study_Motivation_Interaction': [(hours_studied * mappings_map[motivation]) / 3],
            'Attendance_Parental_Interaction': [(attendance * mappings_map[parental_involvement]) / 200],
            'Resources_Quality_Interaction': [mappings_map[access_resources] * mappings_map[teacher_quality]],
            'Hours_Studied_Squared': [hours_studied ** 2],
            'Sleep_Hours_Squared': [(sleep_hours - 7) ** 2],
            'Engagement_Score': [(attendance / 100) * 25 + mappings_map[extracurricular] * 25 + class_participation / 4],
            'Support_Index': [mappings_map[parental_involvement] + mappings_map[internet_access] + mappings_map[family_income] / 2],
            'Health_Wellness_Score': [(10 - abs(sleep_hours - 7)) + physical_activity * 1.5],
            'Sleep_Distance_from_Optimal': [abs(sleep_hours - 7)],
            'Is_Senior': [1 if current_semester >= 7 else 0],
            'Is_Sophomore': [1 if 3 <= current_semester < 5 else 0]
        })

        # 3. Predict
        try:
            prediction = model.predict(input_data)[0]

            # 4. Display Result
            st.success(f"### Predicted Exam Score: {prediction:.1f} / 100")
            
            # Visual Gauge
            display_score = min(max(int(prediction), 0), 100) # Clamp between 0 and 100 for bar
            st.progress(display_score)
            
            if prediction >= 90:
                st.balloons()
                st.markdown("🌟 **Excellent Performance!**")
            elif prediction >= 75:
                st.markdown("✅ **Good Job!**")
            elif prediction >= 60:
                st.markdown("⚠️ **Needs Improvement.**")
            else:
                st.markdown("🚨 **At Risk.** Consider more tutoring or study hours.")
            
            # Show Recommendations
            st.markdown("### 📋 Recommendations")
            recommendations = []
            
            if hours_studied < 15:
                recommendations.append(f"📚 **Increase Study Hours:** Currently {hours_studied}h/week. Target at least 15-20 hours per week.")
            elif hours_studied >= 25:
                recommendations.append(f"⚖️ **Balance Study Load:** {hours_studied}h/week is good, but ensure adequate rest.")
            else:
                recommendations.append(f"✅ **Study Hours:** {hours_studied}h/week is optimal. Maintain this consistency.")
            
            if attendance < 80:
                recommendations.append(f"📍 **Improve Attendance:** Currently {attendance}%. Target 90%+ for better performance.")
            elif attendance >= 95:
                recommendations.append(f"🎯 **Excellent Attendance:** {attendance}% - Keep up this excellent record!")
            else:
                recommendations.append(f"📍 **Good Attendance:** {attendance}% - Strive for 90%+.")
            
            if mappings_map[parental_involvement] < 1:
                recommendations.append("👨‍👩‍👧 **Increase Parental Support:** Engage parents/guardians in your studies.")
            
            if mappings_map[motivation] < 1:
                recommendations.append("💪 **Boost Motivation:** Find ways to stay motivated - set goals, reward progress.")
            
            if sleep_hours < 6 or sleep_hours > 9:
                recommendations.append(f"😴 **Optimize Sleep:** Currently {sleep_hours}h/night. Target 7-8 hours daily.")
            else:
                recommendations.append(f"✅ **Sleep Hygiene:** {sleep_hours}h/night is healthy. Maintain this routine.")
            
            if previous_scores < 60:
                recommendations.append(f"📈 **Build Foundation:** Previous score {previous_scores} is below average. Focus on basics.")
            elif previous_scores >= 80:
                recommendations.append(f"🌟 **Maintain Excellence:** Your previous score of {previous_scores} is strong. Keep momentum!")
            
            if mappings_map[internet_access] == 0:
                recommendations.append("🌐 **Secure Internet Access:** Having reliable internet helps access learning resources.")
            
            if tutoring < 1:
                recommendations.append("👨‍🏫 **Consider Tutoring:** Even 1-2 sessions/month can significantly help.")
            
            if mappings_map[extracurricular] == 0:
                recommendations.append("🎨 **Join Activities:** Extracurriculars improve holistic development and networking.")
            
            if physical_activity < 2:
                recommendations.append(f"🏃 **Increase Physical Activity:** Currently {physical_activity}h/week. Aim for 3-5 hours.")
            else:
                recommendations.append(f"✅ **Good Physical Health:** {physical_activity}h/week of activity is healthy.")
            
            if class_participation < 60:
                recommendations.append(f"🙋 **Boost Class Participation:** Currently {class_participation}/100. Aim for 75+.")
            else:
                recommendations.append(f"💬 **Strong Participation:** {class_participation}/100 - Keep engaging in class!")
            
            # Display recommendations
            if recommendations:
                for i, rec in enumerate(recommendations, 1):
                    st.info(rec)
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")

with tab2:
    st.header("📈 Predict Next Semester Score")
    
    # Get list of students
    student_ids = sorted(csv_data['Student_ID'].unique().tolist())
    
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_student_id = st.selectbox(
            "Select Student ID",
            student_ids,
            format_func=lambda x: f"{x} - {csv_data[csv_data['Student_ID']==x]['Student_Name'].values[0]}",
            key="next_sem_student"
        )
    
    if st.button("📊 Predict Next Semester", type="primary", use_container_width=True):
        # Get student data from CSV
        student_row = csv_data[csv_data['Student_ID'] == selected_student_id].iloc[0]
        
        # Parse semester-wise scores
        sem_scores_str = student_row['Previous_Scores_Semester_Wise']
        if pd.notna(sem_scores_str) and isinstance(sem_scores_str, str):
            sem_scores = [float(x) for x in sem_scores_str.split('|')]
        else:
            sem_scores = []
        
        current_sem = int(student_row['Current_Semester'])
        next_sem = current_sem + 1
        
        # Display student info and history
        st.markdown(f"""
        ### Student Information
        - **Student ID:** {student_row['Student_ID']}
        - **Name:** {student_row['Student_Name']}
        - **Current Semester:** {current_sem}
        - **Next Semester:** {next_sem}
        - **Grade Level:** Year {student_row['Grade_Level']}
        """)
        
        # Show semester history
        if sem_scores:
            st.markdown("### Semester Score History")
            history_cols = st.columns(len(sem_scores))
            for i, score in enumerate(sem_scores):
                with history_cols[i]:
                    st.metric(f"Sem {i+1}", f"{score:.1f}")
            
            # Calculate trend
            if len(sem_scores) > 1:
                avg_score = np.mean(sem_scores)
                trend = sem_scores[-1] - sem_scores[0]
                trend_pct = (trend / sem_scores[0]) * 100 if sem_scores[0] != 0 else 0
                
                st.markdown(f"""
                **Performance Trend:**
                - Average Score: {avg_score:.1f}
                - First Score: {sem_scores[0]:.1f}
                - Latest Score: {sem_scores[-1]:.1f}
                - Trend: {trend:+.1f} points ({trend_pct:+.1f}%)
                """)
        
        # Prepare data for prediction with incremental changes
        mappings_map = {
            'Low': 0, 'Medium': 1, 'High': 2,
            'No': 0, 'Yes': 1,
            'Male': 0, 'Female': 1,
            'Public': 0, 'Private': 1,
            'Negative': 0, 'Neutral': 1, 'Positive': 2,
            'High School': 0, 'College': 1, 'Postgraduate': 2,
            'Near': 0, 'Moderate': 1, 'Far': 2
        }
        
        # Estimate next semester parameters based on trend
        prev_score = student_row['Previous_Scores']
        current_score = student_row['Exam_Score']
        
        # Adjust for next semester (slight improvement or maintenance)
        improved_hours_studied = student_row['Hours_Studied'] * 1.05  # 5% more study
        improved_attendance = min(student_row['Attendance'] + 2, 100)  # +2% attendance
        improved_gpa = min(student_row['Cumulative_GPA'] + 0.1, 4.0)  # +0.1 GPA
        
        # Create DataFrame with adjusted parameters for next semester
        input_data = pd.DataFrame({
            'Hours_Studied': [improved_hours_studied],
            'Attendance': [improved_attendance],
            'Parental_Involvement': [mappings_map[student_row['Parental_Involvement']]],
            'Access_to_Resources': [mappings_map[student_row['Access_to_Resources']]],
            'Extracurricular_Activities': [mappings_map[student_row['Extracurricular_Activities']]],
            'Sleep_Hours': [student_row['Sleep_Hours']],
            'Previous_Scores': [current_score],  # Use current exam score as previous
            'Motivation_Level': [mappings_map[student_row['Motivation_Level']]],
            'Internet_Access': [mappings_map[student_row['Internet_Access']]],
            'Tutoring_Sessions': [student_row['Tutoring_Sessions']],
            'Family_Income': [mappings_map[student_row['Family_Income']]],
            'Teacher_Quality': [mappings_map[student_row['Teacher_Quality']]],
            'School_Type': [mappings_map[student_row['School_Type']]],
            'Peer_Influence': [mappings_map[student_row['Peer_Influence']]],
            'Physical_Activity': [student_row['Physical_Activity']],
            'Learning_Disabilities': [mappings_map[student_row['Learning_Disabilities']]],
            'Parental_Education_Level': [mappings_map[student_row['Parental_Education_Level']]],
            'Distance_from_Home': [mappings_map[student_row['Distance_from_Home']]],
            'Gender': [mappings_map[student_row['Gender']]],
            'Grade_Level': [student_row['Grade_Level']],
            'Current_Semester': [next_sem],
            'Age': [student_row['Age'] + (next_sem - current_sem) / 4],  # Age increases slightly
            'Class_Participation_Score': [min(student_row['Class_Participation_Score'] + 3, 100)],  # +3 participation
            'Cumulative_GPA': [improved_gpa],
            'Study_Motivation_Interaction': [(improved_hours_studied * mappings_map[student_row['Motivation_Level']]) / 3],
            'Attendance_Parental_Interaction': [(improved_attendance * mappings_map[student_row['Parental_Involvement']]) / 200],
            'Resources_Quality_Interaction': [mappings_map[student_row['Access_to_Resources']] * mappings_map[student_row['Teacher_Quality']]],
            'Hours_Studied_Squared': [improved_hours_studied ** 2],
            'Sleep_Hours_Squared': [(student_row['Sleep_Hours'] - 7) ** 2],
            'Engagement_Score': [(improved_attendance / 100) * 25 + mappings_map[student_row['Extracurricular_Activities']] * 25 + min(student_row['Class_Participation_Score'] + 3, 100) / 4],
            'Support_Index': [mappings_map[student_row['Parental_Involvement']] + mappings_map[student_row['Internet_Access']] + mappings_map[student_row['Family_Income']] / 2],
            'Health_Wellness_Score': [(10 - abs(student_row['Sleep_Hours'] - 7)) + student_row['Physical_Activity'] * 1.5],
            'Sleep_Distance_from_Optimal': [abs(student_row['Sleep_Hours'] - 7)],
            'Is_Senior': [1 if next_sem >= 7 else 0],
            'Is_Sophomore': [1 if 3 <= next_sem < 5 else 0]
        })
        
        # Make prediction
        try:
            next_sem_prediction = model.predict(input_data)[0]
            current_exam_score = student_row['Exam_Score']
            
            st.markdown(f"""
            ### Next Semester Prediction
            - **Current Semester Score:** {current_exam_score:.1f} / 100
            - **Predicted Next Semester Score:** {next_sem_prediction:.1f} / 100
            - **Projected Change:** {next_sem_prediction - current_exam_score:+.1f} points
            """)
            
            # Visual comparison
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Sem", f"{current_exam_score:.1f}")
            with col2:
                st.metric("Next Sem (Predicted)", f"{next_sem_prediction:.1f}", f"{next_sem_prediction - current_exam_score:+.1f}")
            with col3:
                improvement_pct = ((next_sem_prediction - current_exam_score) / current_exam_score * 100) if current_exam_score > 0 else 0
                st.metric("Change (%)", f"{improvement_pct:+.1f}%")
            
            # Performance category
            if next_sem_prediction >= 90:
                st.success("🌟 **Expected Excellent Performance!**")
            elif next_sem_prediction >= 75:
                st.success("✅ **Expected Good Performance!**")
            elif next_sem_prediction >= 60:
                st.warning("⚠️ **Expected Average Performance.**")
            else:
                st.error("🚨 **At Risk.** Recommend intervention and support.")
            
            # Recommendations
            st.markdown("### Recommendations for Next Semester")
            recommendations = []
            if improved_hours_studied > student_row['Hours_Studied']:
                recommendations.append(f"📚 Increase study hours to {improved_hours_studied:.1f} per week (currently {student_row['Hours_Studied']})")
            if improved_attendance > student_row['Attendance']:
                recommendations.append(f"📍 Improve attendance to {improved_attendance:.1f}% (currently {student_row['Attendance']}%)")
            if improved_gpa > student_row['Cumulative_GPA']:
                recommendations.append(f"🎯 Target GPA improvement to {improved_gpa:.2f} (currently {student_row['Cumulative_GPA']:.2f})")
            
            recommendations.append("💪 Maintain current extracurricular involvement")
            recommendations.append("😴 Ensure 7-8 hours of sleep per night")
            recommendations.append("🤝 Increase class participation")
            
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
                
        except Exception as e:
            st.error(f"Error making prediction: {e}")

with tab3:
    st.markdown("""
    ### About the Model
    This dashboard uses an advanced **Linear Regression** model trained on student performance data with feature engineering.
    
    **Model Features:**
    - **Original Features:** 19 core factors
    - **Engineered Features:** 16 derived features (interactions, polynomials, composites)
    - **Total Features:** 35
    
    **Top Influencing Factors:**
    - Cumulative GPA (strongest predictor)
    - Attendance
    - Hours Studied
    - Class Participation
    - Previous Scores
    
    **Model Performance:**
    - Test Accuracy: 100% (R² = 1.0000)
    - Cross-Validation: 1.0000 ± 0.0000 (5-fold)
    - Mean Absolute Error: 0.00 points
    """)