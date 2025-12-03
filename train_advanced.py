import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib
import os
import json
import warnings

warnings.filterwarnings('ignore')

# Try importing XGBoost, use without it if not available
try:
    import importlib
    xgb = importlib.import_module('xgboost')
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False
    print("⚠️ XGBoost not installed, skipping XGBoost model")

# ==========================================
# 1. LOAD DATA
# ==========================================
print("Loading dataset...")
file_path = 'StudentPerformanceFactors.csv'

if not os.path.exists(file_path):
    print(f"Error: '{file_path}' not found.")
    print("Make sure the CSV file is in the same folder as this script.")
    exit()

df = pd.read_csv(file_path)
print(f"Loaded {len(df)} records with {len(df.columns)} columns\n")

# ==========================================
# 2. PREPROCESSING & FEATURE ENGINEERING
# ==========================================
print("Preprocessing data...")

# Remove non-predictive columns (including semester history which we'll extract features from)
columns_to_drop = ['Student_ID', 'Student_Name', 'Enrollment_Number', 'Academic_Year', 
                   'Admission_Date', 'Data_Entry_Date', 'Enrollment_Status', 
                   'Previous_Scores_Semester_Wise', 'Section']
df_clean = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Impute missing values
for col in ['Teacher_Quality', 'Parental_Education_Level', 'Distance_from_Home']:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if df_clean[col].mode().size > 0 else 'Medium')

# Define Mappings
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
    'Peer_Influence': {'Negative': 0, 'Neutral': 1, 'Positive': 2},
    'Section': {'A': 0, 'B': 1, 'C': 2, 'D': 3}
}

df_processed = df_clean.copy()
for col, mapping in mappings.items():
    if col in df_processed.columns:
        df_processed[col] = df_processed[col].map(mapping)

# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================
print("Engineering features...")

# Create interaction features
df_processed['Study_Motivation_Interaction'] = (df_processed['Hours_Studied'] * 
                                                 df_processed['Motivation_Level']) / 3
df_processed['Attendance_Parental_Interaction'] = (df_processed['Attendance'] * 
                                                    df_processed['Parental_Involvement']) / 200
df_processed['Resources_Quality_Interaction'] = (df_processed['Access_to_Resources'] * 
                                                  df_processed['Teacher_Quality'])

# Create polynomial features for key variables
df_processed['Hours_Studied_Squared'] = df_processed['Hours_Studied'] ** 2
df_processed['Sleep_Hours_Squared'] = (df_processed['Sleep_Hours'] - 7) ** 2  # Deviation from optimal 7 hours

# Create derived features
df_processed['Engagement_Score'] = (
    (df_processed['Attendance'] / 100) * 25 +
    df_processed['Extracurricular_Activities'] * 25 +
    df_processed['Class_Participation_Score'] / 4
)

df_processed['Support_Index'] = (
    df_processed['Parental_Involvement'] * 1 +
    df_processed['Internet_Access'] * 1 +
    df_processed['Family_Income'] / 2
)

df_processed['Health_Wellness_Score'] = (
    (10 - abs(df_processed['Sleep_Hours'] - 7)) * 1 +
    df_processed['Physical_Activity'] * 1.5
)

# Normalize sleep hours to optimal 7 hours
df_processed['Sleep_Distance_from_Optimal'] = abs(df_processed['Sleep_Hours'] - 7)

# Add semester-based features if available
if 'Current_Semester' in df_processed.columns:
    df_processed['Is_Senior'] = (df_processed['Current_Semester'] >= 7).astype(int)
    df_processed['Is_Sophomore'] = ((df_processed['Current_Semester'] >= 3) & 
                                     (df_processed['Current_Semester'] < 5)).astype(int)

print(f"Features increased from 20 to {len(df_processed.columns) - 1} (excluding target)")

# ==========================================
# 4. PREPARE DATA FOR TRAINING
# ==========================================
X = df_processed.drop('Exam_Score', axis=1)
y = df_processed['Exam_Score']

feature_names = X.columns.tolist()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler for app usage
joblib.dump(scaler, 'scaler.pkl')

# ==========================================
# 5. TRAIN MULTIPLE MODELS
# ==========================================
print("\nTraining models with cross-validation...\n")

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, 
                                          max_depth=15, min_samples_split=5),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, 
                                                   learning_rate=0.1, max_depth=5)
}

if HAS_XGBOOST:
    models['XGBoost'] = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, 
                                   learning_rate=0.1, max_depth=5, verbosity=0)

results = {}
trained_models = {}

for model_name, model in models.items():
    print(f"Training {model_name}...")
    
    # Use scaled data for Linear Regression, original for tree-based models
    if model_name == 'Linear Regression':
        X_train_use = X_train_scaled
        X_test_use = X_test_scaled
    else:
        X_train_use = X_train
        X_test_use = X_test
    
    # Train
    model.fit(X_train_use, y_train)
    trained_models[model_name] = model
    
    # Predict
    y_pred_train = model.predict(X_train_use)
    y_pred_test = model.predict(X_test_use)
    
    # Evaluate
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    # Calculate accuracy percentage
    mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
    accuracy = max(0, 100 - mape)
    
    results[model_name] = {
        'train_r2': float(train_r2),
        'test_r2': float(test_r2),
        'train_mae': float(train_mae),
        'test_mae': float(test_mae),
        'train_rmse': float(train_rmse),
        'test_rmse': float(test_rmse),
        'accuracy': float(accuracy),
        'mape': float(mape)
    }
    
    print(f"  Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")
    print(f"  Train MAE: {train_mae:.2f} | Test MAE: {test_mae:.2f}")
    print(f"  Test RMSE: {test_rmse:.2f} | Accuracy: {accuracy:.2f}%\n")

# ==========================================
# 6. K-FOLD CROSS-VALIDATION
# ==========================================
print("\nPerforming 5-Fold Cross-Validation...\n")

cv_results = {}
for model_name, model in trained_models.items():
    if model_name == 'Linear Regression':
        X_use = X_train_scaled
    else:
        X_use = X_train
    
    cv_scores = cross_val_score(model, X_use, y_train, cv=5, scoring='r2', n_jobs=-1)
    cv_results[model_name] = {
        'cv_r2_mean': float(cv_scores.mean()),
        'cv_r2_std': float(cv_scores.std()),
        'cv_scores': cv_scores.tolist()
    }
    
    print(f"{model_name} CV R² Scores: {cv_scores}")
    print(f"  Mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")

# ==========================================
# 7. FEATURE IMPORTANCE ANALYSIS
# ==========================================
print("\nAnalyzing Feature Importance...\n")

feature_importance_dict = {}

# Random Forest Feature Importance
rf_model = trained_models['Random Forest']
rf_importance = rf_model.feature_importances_
feature_importance_dict['Random Forest'] = dict(zip(feature_names, 
                                                      [float(x) for x in rf_importance]))

# Gradient Boosting Feature Importance
gb_model = trained_models['Gradient Boosting']
gb_importance = gb_model.feature_importances_
feature_importance_dict['Gradient Boosting'] = dict(zip(feature_names, 
                                                         [float(x) for x in gb_importance]))

# XGBoost Feature Importance
if HAS_XGBOOST and 'XGBoost' in trained_models:
    xgb_model = trained_models['XGBoost']
    xgb_importance = xgb_model.feature_importances_
    feature_importance_dict['XGBoost'] = dict(zip(feature_names, 
                                                  [float(x) for x in xgb_importance]))

# Linear Regression Coefficients (absolute values as importance)
lr_model = trained_models['Linear Regression']
lr_importance = np.abs(lr_model.coef_)
feature_importance_dict['Linear Regression'] = dict(zip(feature_names, 
                                                         [float(x) for x in lr_importance]))

# Print top 10 features from Random Forest
print("Top 10 Most Important Features (Random Forest):")
sorted_features = sorted(feature_importance_dict['Random Forest'].items(), 
                        key=lambda x: x[1], reverse=True)
for i, (feat, imp) in enumerate(sorted_features[:10], 1):
    print(f"  {i}. {feat}: {imp:.4f}")

# ==========================================
# 8. SELECT BEST MODEL
# ==========================================
best_model_name = max(results, key=lambda x: results[x]['test_r2'])
best_model = trained_models[best_model_name]

print(f"\n{'='*50}")
print(f"BEST MODEL: {best_model_name}")
print(f"Test R²: {results[best_model_name]['test_r2']:.4f}")
print(f"Test MAE: {results[best_model_name]['test_mae']:.2f} points")
print(f"Accuracy: {results[best_model_name]['accuracy']:.2f}%")
print(f"{'='*50}\n")

# ==========================================
# 9. SAVE ARTIFACTS
# ==========================================
print("Saving models and artifacts...\n")

# Save best model
joblib.dump(best_model, 'student_performance_model.pkl')

# Save all models
joblib.dump(trained_models, 'all_models.pkl')

# Save results
with open('model_results.json', 'w') as f:
    json.dump({
        'individual_results': results,
        'best_model': best_model_name,
        'feature_names': feature_names,
        'cv_results': cv_results
    }, f, indent=4)

# Save feature importance
with open('feature_importance.json', 'w') as f:
    json.dump(feature_importance_dict, f, indent=4)

# Save residuals for confidence interval calculation
y_pred_best = best_model.predict(X_test_scaled if best_model_name == 'Linear Regression' else X_test)
residuals = (y_test - y_pred_best).tolist()

with open('residuals.json', 'w') as f:
    json.dump({
        'residuals': residuals,
        'std_residuals': float(np.std(residuals)),
        'mean_residuals': float(np.mean(residuals))
    }, f, indent=4)

print("✅ Best model saved as 'student_performance_model.pkl'")
print("✅ All models saved as 'all_models.pkl'")
print("✅ Results saved as 'model_results.json'")
print("✅ Feature importance saved as 'feature_importance.json'")
print("✅ Residuals saved as 'residuals.json'")

# ==========================================
# 10. GENERATE COMPREHENSIVE REPORT
# ==========================================
print("\n" + "="*60)
print("         COMPREHENSIVE MODEL PERFORMANCE REPORT         ")
print("="*60)

for model_name, result in results.items():
    marker = " ⭐ BEST" if model_name == best_model_name else ""
    print(f"\n{model_name}{marker}")
    print("-" * 60)
    print(f"  Test R²:        {result['test_r2']:.4f}")
    print(f"  Test MAE:       {result['test_mae']:.2f} points")
    print(f"  Test RMSE:      {result['test_rmse']:.2f} points")
    print(f"  Accuracy:       {result['accuracy']:.2f}%")
    
    if model_name in cv_results:
        cv = cv_results[model_name]
        print(f"  CV Mean R² (5-fold): {cv['cv_r2_mean']:.4f} ± {cv['cv_r2_std']:.4f}")

print("\n" + "="*60)
print(f"Feature Count: {len(feature_names)}")
print(f"Training Samples: {len(X_train)}")
print(f"Test Samples: {len(X_test)}")
print("="*60 + "\n")
