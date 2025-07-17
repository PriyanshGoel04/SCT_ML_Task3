# 🐱🐶 Task 3 – Cat vs Dog Classification using SVM

This project is part of my Machine Learning Internship at **SkillCraft Technology**.

---

## 🧠 Objective

Build a **Support Vector Machine (SVM)** classifier to distinguish between **cat** and **dog** images using a small, manually curated dataset.

The model is trained using image pixel data after resizing and flattening.

---

## 🧰 Tools & Libraries Used

- **Python**
- **OpenCV (`cv2`)** – for image reading and resizing
- **NumPy** – for array and numerical operations
- **Scikit-learn** – for SVM, model training, and evaluation

---

## 📂 Project Files

- `svm_cat_dog_classifier.py` – Main script that:
  - Loads and preprocesses image data
  - Trains an SVM model
  - Evaluates and prints accuracy & classification report

- `README.md` – Project overview and explanation

---

## 📁 Dataset Structure

A manually collected image dataset organized like this:

CATS & DOGS/
├── CATS/
│ ├── Cat1.jpg
│ ├── Cat2.jpg
│ └── ...
└── DOGS/
├── Dog1.jpg
├── Dog2.jpg
└── ...

## 🎯 Accuracy: 0.50

## Classification Report:

precision recall f1-score support
    CATS       1.00      1.00      1.00         1
    DOGS       0.00      0.00      0.00         1

## ✅ Status

✔️ Completed
📅 Submitted: July 2025
📁 Repo: SCT_ML_Task3    
