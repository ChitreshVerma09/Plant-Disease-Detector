# 🌱 Plant Disease Diagnostic System

An AI-powered web application built with **Streamlit** and **TensorFlow** to detect diseases in potato plants using leaf image analysis. It provides real-time predictions along with confidence scores and actionable agricultural remedies.

---

## ✨ Features

* 📸 **Instant Image Upload:** Upload leaf photos in JPG, JPEG, or PNG format.
* 🧠 **Deep Learning Diagnostics:** Trained Convolutional Neural Network (CNN) for accurate prediction.
* 💡 **Treatment Guidance:** Provides remedies and immediate action steps for detected diseases.
* ⚡ **Fast & Responsive UI:** Clean interactive interface designed using Streamlit.

---

## 🛠️ Tech Stack

* **Frontend/UI:** Streamlit
* **Deep Learning:** TensorFlow / Keras
* **Data Handling:** NumPy, Pillow
* **Language:** Python 3.10+

---

## 🚀 Local Setup & Run

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/ChitreshVerma09/Plant-Disease-Detector.git](https://github.com/ChitreshVerma09/Plant-Disease-Detector.git)
   cd Plant-Disease-Detector

  1) Install dependencies:
   Bash
   pip install -r requirements.txt

  2) Run the Streamlit Web App:
    Bash
    python -m streamlit run app.py


📂 Project Structure-
Plaintext
├── app.py               # Streamlit Frontend & Model Inference Logic
├── train_model.py       # CNN Architecture & Model Training Pipeline
├── class_names.npy      # Encoded Class Names Array
├── requirements.txt     # Python Dependencies
└── README.md            # Project Documentation
