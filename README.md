## 🌌 Galaxy Classifier CNN

This project implements a Convolutional Neural Network (CNN) to classify galaxy images into three categories:

* **E** – Elliptical
* **S** – Spiral
* **SB** – Barred Spiral

Trained on astronomical image data resized to **96x96**, the model achieves high accuracy and is suitable for morphology-based galaxy classification.

---

### 📈 Model Performance

* **Training Accuracy:** 83.56%
* **Test Accuracy:** 79%
* **Input Size:** 96x96×3
* **Model Type:** CNN (Keras/TensorFlow)

---

### 📄 Description
The model learns visual galaxy features and distinguishes structural differences between elliptical, spiral, and barred spiral galaxies. It processes raw image data into a format suitable for deep learning, and outputs class predictions with competitive accuracy.

This project demonstrates practical use of CNNs in astronomy and deep learning image classification, and can be used for academic, educational, or research purposes.

---

### 🧪 Features

* Custom image preprocessing using `image_dataset_from_directory`
* Supports prediction on new images
* Dataset compatible structure and scalable design
* Includes model saving/loading functionality

---

### 📁 Files Overview

* `model/` – Saved model files
* `notebooks/` – Training & evaluation notebook
* `predictor/` – Script to classify new images
* `requirements.txt` – Python dependencies
