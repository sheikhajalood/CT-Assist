# 🫁 CT-Assist – Lung Candidate Detection

## 📌 Project Overview

**CT-Assist** is a machine-learning-based research prototype designed to assist in identifying and prioritizing potential candidate regions in lung CT images.

The system combines **image processing techniques, feature extraction, and a Random Forest machine-learning model** to analyze CT images and highlight regions that may require further examination.

The project provides a **Streamlit-based web interface** where users can upload a CT image and view the detected candidate regions, extracted features, and their corresponding prototype scores.

> ⚠️ **Important:** CT-Assist is a research/demo prototype and is **not a medical diagnostic system**. The scores produced by the current prototype should not be interpreted as clinical probabilities or a cancer diagnosis.

---

## 🎯 Problem Statement

Analysis of lung CT images can involve examining a large amount of visual information, making the identification and prioritization of potentially relevant regions a challenging task.

A computer-assisted system can help by automatically processing CT images, identifying candidate regions, extracting measurable image features, and prioritizing those regions for further analysis.

The problem addressed by CT-Assist is therefore:

> **To develop a computer-aided prototype that can process lung CT images, detect potential candidate regions, extract relevant image features, and prioritize the detected regions using a machine-learning model.**

The system is intended to support image analysis and candidate prioritization rather than replace clinical expertise.

---

## 💡 Proposed Solution

CT-Assist follows an image-processing and machine-learning pipeline:

```text
                    CT Image
                        │
                        ▼
              Image Preprocessing
                        │
                        ▼
               Local Contrast
                  Processing
                        │
                        ▼
                  Thresholding
                        │
                        ▼
              Morphological Filtering
                        │
                        ▼
              Candidate Detection
                        │
                        ▼
               Feature Extraction
                        │
                        ▼
              Random Forest Model
                        │
                        ▼
          Candidate Prioritization
                        │
                        ▼
             Results & Visualization
```

---

## 🔄 System Workflow

### 1. CT Image Upload

The user uploads a CT image through the Streamlit application.

Supported image formats:

* PNG
* JPG
* JPEG

---

### 2. Image Preprocessing

The uploaded image is converted into grayscale when required.

Gaussian filtering is then applied to reduce small-scale noise and produce a smoother representation of the image.

---

### 3. Local Contrast Processing

The system compares the original grayscale image with its blurred version.

This helps emphasize local intensity differences that may correspond to potential candidate regions.

---

### 4. Thresholding

A threshold is applied to the local contrast image to generate a binary candidate mask.

Regions satisfying the threshold condition are highlighted as possible candidate areas.

---

### 5. Morphological Filtering

Morphological operations are applied to clean the candidate mask and remove some small unwanted regions.

This helps produce more meaningful candidate regions before contour detection.

---

### 6. Candidate Detection

OpenCV contour detection is used to identify separate candidate regions.

Very small contours are filtered using a minimum-area threshold.

Each remaining contour represents a candidate region for further processing.

---

## 🔬 Feature Extraction

For every detected candidate region, the system calculates a set of image-based features.

| Feature            | Description                                                  |
| ------------------ | ------------------------------------------------------------ |
| **Area**           | Size of the detected candidate region                        |
| **Perimeter**      | Boundary length of the region                                |
| **Aspect Ratio**   | Ratio between the width and height of the bounding box       |
| **Extent**         | Ratio of contour area to bounding-box area                   |
| **Mean Intensity** | Average pixel intensity inside the region                    |
| **Intensity Std**  | Variation of pixel intensity inside the region               |
| **Circularity**    | Measure of how closely the region resembles a circular shape |

These features form the input to the machine-learning stage.

---

## 🤖 Machine Learning Model

The project uses a:

**Random Forest Classifier**

The model is configured using:

```text
n_estimators = 100
random_state = 42
```

The model uses the following seven features:

```text
Area
Perimeter
Aspect Ratio
Extent
Mean Intensity
Intensity Std
Circularity
```

The trained model is stored as:

```text
lung_candidate_model.pkl
```

The Streamlit application loads this model and uses it to generate predictions for the candidate regions detected in an uploaded image.

---

## 📊 Prototype Score

The application displays a **Prototype Score** for each candidate region.

The score is used to prioritize candidate regions within the current research prototype.

For example:

```text
Region 5 → Higher prototype score
Region 6 → Higher prototype score
Region 2 → Higher prototype score
```

Higher scores indicate that the current model gives that candidate a higher priority within the prototype.

> ⚠️ These scores are **not medical probabilities**, and they should not be interpreted as indicating that a region is cancerous.

---

## 🖥️ Application Interface

The project uses **Streamlit** to provide a simple web-based interface.

The application allows the user to:

* Upload a CT image
* View the uploaded image
* Analyze the image
* View the processed candidate mask
* View detected candidate regions
* View bounding boxes around candidates
* View extracted image features
* View Random Forest predictions
* View prototype scores
* Identify higher-priority candidate regions

---

## 📸 Results

After analysis, the application provides:

### Candidates Detected

The number of candidate regions identified by the image-processing pipeline.

### High-Priority Candidates

The number of candidate regions classified as the higher-priority class by the current Random Forest prototype.

### Candidate Mask

A processed binary image showing the regions identified during candidate detection.

### Bounding Boxes

Detected candidate regions are displayed with bounding boxes and region labels.

Example:

```text
       ┌───────────┐
       │    R1     │
       │ candidate │
       └───────────┘

       ┌───────────┐
       │    R2     │
       │ candidate │
       └───────────┘
```

### Candidate Table

The application displays information such as:

```text
Region
Prototype Score
Prediction
Area
Perimeter
Aspect Ratio
Extent
Mean Intensity
Intensity Std
Circularity
```

---

## 🗂️ Project Structure

```text
CT-Assist/
│
├── app.py
│
├── lung_candidate_model.pkl
│
└── README.md
```

### `app.py`

Contains the Streamlit application, image-processing pipeline, candidate detection, feature extraction, and Random Forest prediction.

### `lung_candidate_model.pkl`

Saved Random Forest model used for candidate prioritization.

### `README.md`

Project documentation.

---

## ⚙️ Technologies Used

### Programming Language

* Python

### Image Processing

* OpenCV
* NumPy
* Pillow

### Machine Learning

* Scikit-learn
* Random Forest

### Data Processing

* Pandas

### Model Storage

* Joblib

### Web Application

* Streamlit

### Development Environment

* Google Colab
* Visual Studio Code / local Python environment
* GitHub

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/CT-Assist.git
```

Move into the project directory:

```bash
cd CT-Assist
```

---

### 2. Install dependencies

Install the required Python libraries:

```bash
pip install streamlit opencv-python numpy pandas pillow joblib scikit-learn
```

---

### 3. Check the project files

Make sure the directory contains:

```text
CT-Assist/
│
├── app.py
├── lung_candidate_model.pkl
└── README.md
```

---

## ▶️ Running the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

The application will open in the browser.

The local Streamlit address is usually:

```text
http://localhost:8501
```

---

## 🧪 How to Use

### Step 1

Open the Streamlit application.

### Step 2

Upload a CT image using the **Upload CT Image** option.

### Step 3

Click:

```text
🔍 Analyze CT Scan
```

### Step 4

The system processes the image and detects candidate regions.

### Step 5

View:

* Candidate count
* Processed candidate mask
* Bounding boxes
* Candidate features
* Random Forest predictions
* Prototype scores

### Step 6

Use the displayed candidate ranking to identify regions requiring further analysis.

---

## 👥 Team Collaboration

The project is maintained using GitHub.

Team members can be added to the repository as collaborators.

After accepting the invitation, a team member can clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/CT-Assist.git
```

Then:

```bash
cd CT-Assist
```

Install the dependencies:

```bash
pip install streamlit opencv-python numpy pandas pillow joblib scikit-learn
```

Run the application:

```bash
streamlit run app.py
```

---

## 🔄 Updating the Project

Before working on the latest version:

```bash
git pull
```

After making changes:

```bash
git add .
git commit -m "Update CT-Assist"
git push
```

For team development, separate branches can also be used to avoid conflicts.

---

## 📈 Advantages

* Provides an automated candidate-region detection pipeline
* Reduces the need to manually inspect every image region
* Combines image processing with machine learning
* Provides interpretable image-based features
* Visualizes detected candidate regions
* Provides candidate prioritization through prototype scores
* Provides an easy-to-use Streamlit interface
* Can serve as a foundation for future research and development

---

## ⚠️ Current Limitations

The current implementation has several limitations.

1. The system is a **research/demo prototype**, not a clinical diagnostic tool.

2. Candidate detection is based on traditional image-processing techniques and may produce false-positive or missed regions.

3. The current Random Forest model is based on prototype-generated labels rather than clinically validated ground-truth annotations.

4. The current system does not perform clinical diagnosis.

5. The prototype has not been clinically validated.

6. Performance may vary depending on CT image quality, acquisition conditions, image format, and preprocessing requirements.

7. The current system works with individual uploaded images rather than complete clinical CT volumes.

---

## 🚀 Future Scope

The system can be further improved through:

### 1. Clinically Labelled Dataset

Training and evaluation can be performed using larger datasets containing reliable expert annotations.

### 2. Advanced Segmentation

Lung and candidate-region segmentation can be improved using more advanced computer-vision or deep-learning methods.

### 3. Deep Learning

CNN-based architectures and other modern medical-image models could be explored for improved candidate detection and classification.

### 4. 3D CT Analysis

Instead of analyzing individual 2D slices, future versions could process complete 3D CT volumes.

### 5. Model Evaluation

Future versions should include proper evaluation using metrics such as:

* Accuracy
* Precision
* Recall
* F1-score
* Sensitivity
* Specificity
* ROC-AUC

### 6. Clinical Validation

Any system intended for real medical use would require extensive validation with appropriate clinical datasets and expert assessment.

---

## 🔐 Disclaimer

CT-Assist is developed as an **academic/research prototype**.

It is intended for experimentation and demonstration of image processing and machine-learning techniques.

**It must not be used for medical diagnosis, treatment decisions, or clinical decision-making.**

The prototype scores do not represent clinical probabilities.

---

## 📚 Conclusion

CT-Assist demonstrates how traditional image processing and machine learning can be combined to create a computer-assisted pipeline for lung CT candidate detection.

The system processes a CT image, detects potential candidate regions, extracts quantitative image features, and uses a Random Forest model to prioritize the detected candidates.

The current prototype provides a foundation that can be extended with clinically labelled data, improved segmentation, advanced machine-learning methods, 3D CT analysis, and rigorous validation.
