"""
Quick verification script to test all components
"""
import os
import json

print("=" * 60)
print("STUDENT PERFORMANCE SYSTEM - VERIFICATION CHECK")
print("=" * 60)

# Check for required files
required_files = {
    'Core Models': [
        'student_performance_model.pkl',
        'all_models.pkl',
        'scaler.pkl'
    ],
    'Data Files': [
        'StudentPerformanceFactors.csv',
        'analysis_summary.json'
    ],
    'Configuration': [
        'model_results.json',
        'feature_importance.json',
        'residuals.json'
    ],
    'Scripts': [
        'train_advanced.py',
        'app_advanced.py',
        'model_analysis.py'
    ]
}

print("\n📋 File Status Check:\n")

all_files_present = True
for category, files in required_files.items():
    print(f"{category}:")
    for file in files:
        exists = os.path.exists(file)
        status = "✅" if exists else "❌"
        print(f"  {status} {file}")
        if not exists:
            all_files_present = False
    print()

# Load and display model results
print("📊 Model Performance Summary:\n")
try:
    with open('model_results.json', 'r') as f:
        results = json.load(f)
    
    print(f"Best Model: {results['best_model']}")
    print(f"\nAll Models Tested:")
    for model_name, metrics in results['individual_results'].items():
        print(f"\n  {model_name}:")
        print(f"    - Test R²: {metrics['test_r2']:.4f}")
        print(f"    - Test MAE: {metrics['test_mae']:.2f} points")
        print(f"    - Accuracy: {metrics['accuracy']:.2f}%")
except Exception as e:
    print(f"⚠️  Could not load model results: {e}")

# Load and display feature importance
print("\n\n🎯 Top 5 Most Important Features:\n")
try:
    with open('feature_importance.json', 'r') as f:
        importance = json.load(f)
    
    best_model = results['best_model']
    if best_model in importance:
        features = importance[best_model]
        sorted_features = sorted(features.items(), key=lambda x: x[1], reverse=True)
        for i, (feat, imp) in enumerate(sorted_features[:5], 1):
            print(f"  {i}. {feat}: {imp:.4f}")
except Exception as e:
    print(f"⚠️  Could not load feature importance: {e}")

# Load and display analysis
print("\n\n📈 Dataset Analysis Summary:\n")
try:
    with open('analysis_summary.json', 'r') as f:
        analysis = json.load(f)
    
    print(f"Total Students: {analysis['total_students']}")
    print(f"Average Score: {analysis['mean_score']:.2f}")
    print(f"Score Std Dev: {analysis['std_score']:.2f}")
    print(f"High Performers (≥80): {analysis['high_performers']}")
    print(f"Low Performers (<60): {analysis['low_performers']}")
except Exception as e:
    print(f"⚠️  Could not load analysis: {e}")

# Summary
print("\n\n" + "=" * 60)
if all_files_present:
    print("✅ SYSTEM STATUS: READY FOR PRODUCTION")
    print("\nRun the app with:")
    print("  streamlit run app_advanced.py")
else:
    print("⚠️  SYSTEM STATUS: INCOMPLETE")
    print("\nPlease ensure all files are present.")

print("=" * 60)
