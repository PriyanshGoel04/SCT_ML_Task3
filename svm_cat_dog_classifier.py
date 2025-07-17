import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Path to dataset
data_dir = r"C:\Users\Priyansh\Desktop\CLG\CATS & DOGS"
categories = ["CATS", "DOGS"]

# Data holders
X = []  # images (flattened)
y = []  # labels

# Load and process images
for label, category in enumerate(categories):
    folder_path = os.path.join(data_dir, category)
    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (64, 64))  # Resize to 64x64
            X.append(img.flatten())         # Flatten image
            y.append(label)                 # 0 for Cat, 1 for Dog

# Convert to arrays
X = np.array(X)
y = np.array(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=categories))
