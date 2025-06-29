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

### 🧪 Features

* Custom image preprocessing using `image_dataset_from_directory`
* Supports prediction on new images
* Dataset compatible structure and scalable design
* Includes model saving/loading functionality

---

### 🚀 How to Use

```bash
# Clone and install
git clone https://github.com/your-username/galaxy-classification.git
cd galaxy-classification
pip install -r requirements.txt


To predict a new image:

```bash
python predictor/predict_single_image.py --image path/to/image.jpg
```

---

### 📁 Files Overview

* `model/` – Saved model files
* `notebooks/` – Training & evaluation notebook
* `predictor/` – Script to classify new images
* `requirements.txt` – Python dependencies
