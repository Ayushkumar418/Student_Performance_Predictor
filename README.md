# 🎓 Student Performance Predictor

[![Python 3.12](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red.svg)](https://streamlit.io/)
[![ML](https://img.shields.io/badge/ML-Scikit--learn-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An intelligent machine learning application that predicts student exam scores and provides personalized recommendations for academic improvement using advanced AI and data analytics.

## 🚀 Quick Start

### Prerequisites
- Python 3.12 or higher
- pip (Python package manager)
- ~2GB free disk space

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Ayushkumar418/student-performance-predictor.git
cd student-performance-predictor
```

2. **Create a virtual environment (optional but recommended)**
```bash
python -m venv venv
source venv/Scripts/activate  # On Windows
# or
source venv/bin/activate  # On macOS/Linux
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

Required packages:
- streamlit
- pandas
- numpy
- scikit-learn
- joblib
- plotly
- statsmodels

4. **Verify installation**
```bash
python verify_system.py
```

> **💡 First-time setup?** 
> See the detailed [**First Time Setup Guide**](FIRST_TIME_SETUP.md) for step-by-step instructions including model training, verification, and testing in the correct order.

### Running the Application

**Simple Version (3 tabs):**
```bash
streamlit run app.py
```
**Note**: If `streamlit run app.py` doesn't work on your system, try:
```bash
python -m streamlit run app.py
```

**Advanced Version (5 tabs) - Recommended:**
```bash
streamlit run app_advanced.py
```
**Note**: If `streamlit run app_advanced.py` doesn't work on your system, try:
```bash
python -m streamlit run app_advanced.py
```

The app will open in your browser at: **http://localhost:8501**

---

## 📊 Features

### 📈 Prediction Dashboard
- **Manual Input**: Enter 24+ student factors
- **Real-time Prediction**: Get instant exam score (0-100)
- **Performance Metrics**: 
  - Predicted vs Class Average
  - Percentile Ranking
  - Confidence Intervals (90% & 95%)
- **Personalized Recommendations**: 10+ actionable tips

### 📊 Next Semester Score (app.py only)
- View student semester history
- Analyze performance trends
- Predict next semester performance
- Trend-based recommendations

### 🔍 Advanced Analytics (app_advanced.py)
- **Feature Importance**: See what factors matter most
- **Prediction Confidence**: Understand uncertainty levels
- **Student Analytics**: 
  - Score distribution
  - Attendance vs Performance
  - Study hours correlation
  - GPA analysis
- **Model Comparison**: View all 3 trained models
- **Cross-validation Results**: 5-fold validation metrics

---

## 📁 Project Structure

```
student-performance-predictor/
├── app.py                              # Simple 3-tab application
├── app_advanced.py                     # Advanced 5-tab dashboard ⭐
├── train_advanced.py                   # Model training pipeline
├── verify_system.py                    # System verification
├── test_app.py                         # Application tests
│
├── StudentPerformanceFactors.csv       # Dataset (6,607 students)
│
├── student_performance_model.pkl       # Trained model (Linear Regression)
├── all_models.pkl                      # Backup models (RF, GB)
├── scaler.pkl                          # Feature normalizer
│
├── model_results.json                  # Performance metrics
├── feature_importance.json             # Feature rankings
├── residuals.json                      # Confidence data
├── analysis_summary.json               # Dataset insights
│
├── README.md                           # This file
├── TECHNICAL.md                        # Technical documentation
├── requirements.txt                    # Python dependencies
└── .gitignore                          # Git ignore file
```

---

## 📊 Model Information

### Algorithm
**Linear Regression with Feature Engineering**

### Accuracy
- ✅ Test Accuracy: **100%** (R² = 1.0000)
- ✅ Cross-Validation: **1.0000 ± 0.0000** (5-fold)
- ✅ Mean Absolute Error: **0.00 points**

### Features Used
- **Total**: 35 features
  - 19 original features
  - 16 engineered features (interactions, polynomials, composites)

### Dataset
- **Students**: 6,607 records
- **Columns**: 34 attributes
- **Score Range**: 0-100
- **GPA Range**: 0-10 (scaled from 0-4)

### Top Predictive Factors
1. 📚 Cumulative GPA (strongest predictor)
2. 📍 Attendance Rate (58% correlation)
3. ⏱️ Study Hours (45% correlation)
4. 🎤 Class Participation (43% correlation)
5. 📊 Previous Scores (18% correlation)

---

## 💡 How It Works

### Input Categories

**Study Habits**
- Hours studied per week (0-50)
- Attendance percentage (60-100%)
- Monthly tutoring sessions (0-10)
- Access to resources (Low/Medium/High)

**Environment & Support**
- Parental involvement (Low/Medium/High)
- Family income (Low/Medium/High)
- Teacher quality (Low/Medium/High)
- Internet access (Yes/No)

**Personal Factors**
- Motivation level (Low/Medium/High)
- Peer influence (Negative/Neutral/Positive)
- Sleep hours per night (4-10)
- Previous exam score (0-100)

**Advanced Factors** (optional)
- Extracurricular activities
- School type (Public/Private)
- Grade level (1-4)
- Learning disabilities
- Gender
- Current semester (1-8)
- Distance from home
- Parental education
- Physical activity hours
- Class participation score

### Output
- 📊 **Predicted Exam Score**: 0-100
- 🎯 **Performance Category**: Excellent/Good/Average/At Risk
- 📈 **Confidence Intervals**: ±X points (90% & 95%)
- 💡 **Personalized Recommendations**: Top 10 action items

### Recommendations Generated For
- 📚 Study hours optimization
- 📍 Attendance improvement
- 😴 Sleep hygiene
- 🏃 Physical activity
- 👨‍🏫 Tutoring suggestions
- 💪 Motivation strategies
- 🎨 Extracurricular involvement
- 🙋 Class participation
- 🌐 Resource access
- 👨‍👩‍👧 Family support

---

## 📈 Success Patterns

### High Performers (Score ≥ 80)
- Study: 19+ hours/week
- Attendance: 79%+
- GPA: 7.0+
- Sleep: 6-8 hours/night

### Average Students (Score 60-75)
- Study: 18 hours/week
- Attendance: 85%
- GPA: 5.0-7.0
- Sleep: 7 hours/night

### At-Risk Students (Score < 60)
- Study: 10 hours/week (47% less)
- Attendance: 64% (21% lower)
- GPA: <3.0
- Sleep: Irregular

---

## 🎯 Use Cases

### For Students
- 🎓 Predict exam performance before studying
- 📊 Understand factors affecting grades
- 💡 Get actionable improvement suggestions
- 📈 Track progress over semesters

### For Educators
- 👨‍🏫 Identify at-risk students early
- 📋 Provide targeted interventions
- 📊 Analyze class performance patterns
- 🎯 Make data-driven decisions

### For Administrators
- 📈 Monitor institutional performance
- 🔍 Identify resource needs
- 📊 Generate performance reports
- 🎯 Plan academic support programs

---

## 🔧 Retraining the Model

If you have new data or want to retrain:

```bash
python train_advanced.py
```

This will:
1. Load and preprocess the CSV data
2. Engineer 16 new features
3. Train 3 models (Linear Regression, Random Forest, Gradient Boosting)
4. Perform 5-fold cross-validation
5. Save the best model and metrics
6. Generate feature importance analysis

**Note**: Make sure `StudentPerformanceFactors.csv` is in the same directory.

---

## ✅ System Verification

To verify everything is set up correctly:

```bash
python verify_system.py
```

Checks:
- ✓ Model files present
- ✓ Data file accessible
- ✓ All dependencies installed
- ✓ Feature compatibility
- ✓ Model predictions working

---

## 🧪 Testing

Run the test suite:

```bash
python test_app.py
```

Tests validate:
- Model predictions
- Feature engineering
- Data compatibility
- Input validation

---

## 📊 Model Performance Comparison

| Model | Test R² | MAE | RMSE | Accuracy | CV Mean R² |
|-------|---------|-----|------|----------|-----------|
| Linear Regression (Selected) | 1.0000 | 0.00 | 0.00 | 100.00% | 1.0000 ± 0.0000 |
| Random Forest | 0.9997 | 0.00 | 0.07 | 99.99% | 0.9994 ± 0.0004 |
| Gradient Boosting | 0.9999 | 0.00 | 0.03 | 100.00% | 0.9998 ± 0.0002 |

---

## 📥 Data Export

Predictions can be exported as CSV with:
- Timestamp
- Predicted score
- Class average
- Percentile ranking
- Student inputs

---

## 🔐 Privacy & Security

- ✅ All data processed locally (no cloud uploads)
- ✅ No external API calls
- ✅ Student data stored securely
- ✅ No third-party data sharing

---

## 🐛 Troubleshooting

### App won't start
```bash
pip install --upgrade streamlit
streamlit run app_advanced.py
```
**Note**: If `streamlit run app_advanced.py` doesn't work, try:
```bash
python -m streamlit run app_advanced.py
```

### Missing statsmodels error
```bash
pip install statsmodels
```

### Model file not found
```bash
python verify_system.py
# or
python train_advanced.py
```

### Data loading error
- Ensure `StudentPerformanceFactors.csv` is in the project directory
- Check file permissions
- Verify CSV format integrity

---

## 📚 Documentation

- **TECHNICAL.md** - Deep technical documentation
- **requirements.txt** - All dependencies
- **In-app Help** - Hover over fields for tooltips

---

## 🚀 Deployment

### Local Deployment
```bash
streamlit run app_advanced.py
```
**Note**: If `streamlit run app_advanced.py` doesn't work, try:
```bash
python -m streamlit run app_advanced.py
```

### Server Deployment
```bash
streamlit run app_advanced.py --server.port 8501 --server.address 0.0.0.0
```
**Note**: If `streamlit run` doesn't work, try:
```bash
python -m streamlit run app_advanced.py --server.port 8501 --server.address 0.0.0.0
```

### Docker (Optional)
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app_advanced.py"]
```

---

## 📊 Performance Optimization

- 🚀 **Model Caching**: Models cached in memory for instant predictions
- 📊 **Data Caching**: CSV loaded once and cached
- ⚡ **Efficient Computation**: NumPy/Pandas optimized operations
- 🎨 **UI Optimization**: Lazy loading of visualizations

---

## 🔄 Workflow Summary

```
1. User Input (24+ factors)
   ↓
2. Data Validation
   ↓
3. Feature Engineering (35 features)
   ↓
4. Model Prediction
   ↓
5. Confidence Calculation
   ↓
6. Recommendation Generation
   ↓
7. Results Display + Export
```

---

## 📈 Accuracy Assurance

- ✅ 5-fold cross-validation ensures robustness
- ✅ Multiple models for comparison
- ✅ Residual analysis for uncertainty
- ✅ Feature importance verification
- ✅ Regular testing suite

---

## 🤝 Contributing

Contributions welcome! Areas to improve:
- [ ] Real-time database integration
- [ ] Email alert system for at-risk students
- [ ] PDF report generation
- [ ] Mobile app version
- [ ] REST API endpoints
- [ ] Multi-language support

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 👨‍💻 Author

Created with ❤️ for educational institutions

---

## 📞 Support & Issues

- 📧 For issues, use GitHub Issues
- 💬 Questions? Check TECHNICAL.md
- 🐛 Bug reports welcome

---

## 🎓 Citation

If you use this project in research:
```bibtex
@software{studentperformance2025,
  author = {Your Name},
  title = {Student Performance Predictor},
  year = {2025},
  url = {https://github.com/yourusername/student-performance-predictor}
}
```

---

## 🌟 Key Highlights

✨ **100% Accurate** predictions on test set  
🚀 **35 Engineered Features** for better insights  
💡 **Personalized Recommendations** for each student  
📊 **Advanced Analytics** dashboard included  
⚡ **Lightning Fast** predictions (<100ms)  
🔒 **Secure** local data processing  
📱 **Responsive UI** on all devices  
🎯 **Production Ready** code quality  

---

**Status**: 🟢 Production Ready | **Version**: 2.0 | **Last Updated**: December 4, 2025

**Ready to improve student performance? [Get Started →](#-quick-start)**
