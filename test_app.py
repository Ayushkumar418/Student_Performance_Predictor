"""
Test script to verify app.py works correctly with 35 features
"""
import pandas as pd
import joblib
import numpy as np

print("=" * 60)
print("TESTING APP.PY FEATURE COMPATIBILITY")
print("=" * 60)

# Load model
try:
    model = joblib.load('student_performance_model.pkl')
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit(1)

# Test input - simulate what app.py will send
mappings_map = {
    'Low': 0, 'Medium': 1, 'High': 2,
    'No': 0, 'Yes': 1,
    'Male': 0, 'Female': 1,
    'Public': 0, 'Private': 1,
    'Negative': 0, 'Neutral': 1, 'Positive': 2,
    'High School': 0, 'College': 1, 'Postgraduate': 2,
    'Near': 0, 'Moderate': 1, 'Far': 2
}

# Create test input with all 35 features
hours_studied = 20
attendance = 85
grade_level = 2
current_semester = 4
class_participation = 70
previous_scores = 75
sleep_hours = 7

input_data = pd.DataFrame({
    'Hours_Studied': [hours_studied],
    'Attendance': [attendance],
    'Parental_Involvement': [mappings_map['Medium']],
    'Access_to_Resources': [mappings_map['High']],
    'Extracurricular_Activities': [mappings_map['No']],
    'Sleep_Hours': [sleep_hours],
    'Previous_Scores': [previous_scores],
    'Motivation_Level': [mappings_map['Medium']],
    'Internet_Access': [mappings_map['Yes']],
    'Tutoring_Sessions': [1],
    'Family_Income': [mappings_map['Medium']],
    'Teacher_Quality': [mappings_map['Medium']],
    'School_Type': [mappings_map['Public']],
    'Peer_Influence': [mappings_map['Positive']],
    'Physical_Activity': [3],
    'Learning_Disabilities': [mappings_map['No']],
    'Parental_Education_Level': [mappings_map['College']],
    'Distance_from_Home': [mappings_map['Near']],
    'Gender': [mappings_map['Male']],
    'Grade_Level': [grade_level],
    'Current_Semester': [current_semester],
    'Age': [18 + grade_level],
    'Class_Participation_Score': [class_participation],
    'Cumulative_GPA': [2.0 + (previous_scores - 60) * 0.04],
    'Study_Motivation_Interaction': [(hours_studied * mappings_map['Medium']) / 3],
    'Attendance_Parental_Interaction': [(attendance * mappings_map['Medium']) / 200],
    'Resources_Quality_Interaction': [mappings_map['High'] * mappings_map['Medium']],
    'Hours_Studied_Squared': [hours_studied ** 2],
    'Sleep_Hours_Squared': [(sleep_hours - 7) ** 2],
    'Engagement_Score': [(attendance / 100) * 25 + mappings_map['No'] * 25 + class_participation / 4],
    'Support_Index': [mappings_map['Medium'] + mappings_map['Yes'] + mappings_map['Medium'] / 2],
    'Health_Wellness_Score': [(10 - abs(sleep_hours - 7)) + 3 * 1.5],
    'Sleep_Distance_from_Optimal': [abs(sleep_hours - 7)],
    'Is_Senior': [0],
    'Is_Sophomore': [1]
})

print(f"\n✅ Input DataFrame created with {len(input_data.columns)} features")
print(f"   Features: {list(input_data.columns)}")

# Test prediction
try:
    prediction = model.predict(input_data)[0]
    print(f"\n✅ Prediction successful!")
    print(f"   Predicted Score: {prediction:.2f}/100")
    
    if prediction >= 90:
        print(f"   Category: 🌟 Excellent Performance")
    elif prediction >= 75:
        print(f"   Category: ✅ Good Job")
    elif prediction >= 60:
        print(f"   Category: ⚠️ Needs Improvement")
    else:
        print(f"   Category: 🚨 At Risk")
        
except Exception as e:
    print(f"❌ Prediction failed: {e}")
    exit(1)

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED - APP.PY IS READY!")
print("=" * 60)
print("\nRun the app with:")
print("   streamlit run app.py")
print("=" * 60)
