"""
Model Analysis Script - Generates insights and visualizations
"""
import pandas as pd
import numpy as np
import json
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

print("Loading training data and models...")

# Load data
df = pd.read_csv('StudentPerformanceFactors.csv')

# Preprocessing (same as train_advanced.py)
columns_to_drop = ['Student_ID', 'Student_Name', 'Enrollment_Number', 'Academic_Year', 
                   'Admission_Date', 'Data_Entry_Date', 'Enrollment_Status', 
                   'Previous_Scores_Semester_Wise', 'Section']
df_clean = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Mappings
mappings = {
    'Parental_Involvement': {'Low': 0, 'Medium': 1, 'High': 2},
    'Access_to_Resources': {'Low': 0, 'Medium': 1, 'High': 2},
    'Motivation_Level': {'Low': 0, 'Medium': 1, 'High': 2},
    'Family_Income': {'Low': 0, 'Medium': 1, 'High': 2},
    'Teacher_Quality': {'Low': 0, 'Medium': 1, 'High': 2},
    'Parental_Education_Level': {'High School': 0, 'College': 1, 'Postgraduate': 2},
    'Distance_from_Home': {'Near': 0, 'Moderate': 1, 'Far': 2},
    'Extracurricular_Activities': {'No': 0, 'Yes': 1},
    'Internet_Access': {'No': 0, 'Yes': 1},
    'Learning_Disabilities': {'No': 0, 'Yes': 1},
    'School_Type': {'Public': 0, 'Private': 1},
    'Gender': {'Male': 0, 'Female': 1},
    'Peer_Influence': {'Negative': 0, 'Neutral': 1, 'Positive': 2}
}

df_processed = df_clean.copy()
for col, mapping in mappings.items():
    if col in df_processed.columns:
        df_processed[col] = df_processed[col].map(mapping)

# Load models and data
models = joblib.load('all_models.pkl')
model = joblib.load('student_performance_model.pkl')

# Generate correlation analysis
print("\nGenerating correlation analysis...")

numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
correlation_matrix = df_processed[numeric_cols].corr()

# Get top correlations with Exam_Score
top_correlations = correlation_matrix['Exam_Score'].sort_values(ascending=False)

print("\nTop 10 Features Correlated with Exam Score:")
for i, (feat, corr) in enumerate(top_correlations.head(11)[1:].items(), 1):
    print(f"  {i}. {feat}: {corr:.4f}")

# Generate grade-level analysis
print("\nGrade Level Performance Analysis:")
for grade in [1, 2, 3, 4]:
    grade_data = df[df['Grade_Level'] == grade]
    if len(grade_data) > 0:
        mean_score = grade_data['Exam_Score'].mean()
        std_score = grade_data['Exam_Score'].std()
        count = len(grade_data)
        print(f"  Year {grade}: Mean={mean_score:.2f}, Std={std_score:.2f}, N={count}")

# Generate demographics analysis
print("\nDemographics Analysis:")
gender_analysis = df.groupby('Gender')['Exam_Score'].agg(['mean', 'std', 'count'])
print("  By Gender:")
print(gender_analysis)

school_analysis = df.groupby('School_Type')['Exam_Score'].agg(['mean', 'std', 'count'])
print("\n  By School Type:")
print(school_analysis)

# Generate behavioral patterns
print("\nBehavioral Patterns:")
high_performers = df[df['Exam_Score'] >= 80]
low_performers = df[df['Exam_Score'] < 60]

print(f"  High Performers (Score >= 80): {len(high_performers)} students")
print(f"    Avg Hours Studied: {high_performers['Hours_Studied'].mean():.2f}")
print(f"    Avg Attendance: {high_performers['Attendance'].mean():.2f}%")
print(f"    Avg Sleep Hours: {high_performers['Sleep_Hours'].mean():.2f}")

print(f"\n  Low Performers (Score < 60): {len(low_performers)} students")
print(f"    Avg Hours Studied: {low_performers['Hours_Studied'].mean():.2f}")
print(f"    Avg Attendance: {low_performers['Attendance'].mean():.2f}%")
print(f"    Avg Sleep Hours: {low_performers['Sleep_Hours'].mean():.2f}")

# Save analysis summary
analysis_summary = {
    'total_students': len(df),
    'mean_score': float(df['Exam_Score'].mean()),
    'std_score': float(df['Exam_Score'].std()),
    'high_performers': len(high_performers),
    'low_performers': len(low_performers),
    'grade_levels': {
        f'year_{i}': {
            'count': int(len(df[df['Grade_Level'] == i])),
            'mean_score': float(df[df['Grade_Level'] == i]['Exam_Score'].mean())
        }
        for i in [1, 2, 3, 4]
    },
    'top_correlations': {feat: float(corr) for feat, corr in top_correlations.head(11)[1:].items()}
}

with open('analysis_summary.json', 'w') as f:
    json.dump(analysis_summary, f, indent=4)

print("\n✅ Analysis complete. Results saved to 'analysis_summary.json'")
