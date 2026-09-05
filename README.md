# CT-Assist 🫁

AI-Based CT Scan Candidate Detection and Analysis

## 📌 Overview

CT-Assist is a research/demo prototype designed to analyze CT scan images and identify candidate regions that may require further review.

The system combines image processing techniques with a Machine Learning model to highlight and rank candidate regions in CT scans.

> ⚠️ This project is a prototype for educational and hackathon purposes. It is not a medical diagnostic system and the highlighted regions should not be considered confirmed tumors or medical diagnoses.

## 🚀 Features

- Upload CT scan images through a Streamlit web interface
- Preprocess CT images using OpenCV
- Detect candidate regions using local contrast analysis
- Extract image-based features from candidate regions
- Classify candidate regions using a Random Forest model
- Rank candidate regions based on a prototype priority score
- Display highlighted candidate regions
- Show candidate statistics and extracted features
- Interactive and easy-to-use interface

## 🛠️ Technologies Used

- Python
- Streamlit
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- Joblib
- Pillow

## 🔬 System Workflow

```text
CT Scan Image
      ↓
Image Preprocessing
      ↓
Local Contrast Analysis
      ↓
Candidate Region Detection
      ↓
Feature Extraction
      ↓
Random Forest Classification
      ↓
Candidate Ranking
      ↓
Highlighted CT Scan
      ↓
Analysis Results