# 🏛️ Large-Scale Landmark Classification using VGG19

> A deep learning project focused on large-scale image classification across **7,444 landmark classes** using a custom **VGG19 architecture trained from scratch** on a real-world landmark dataset.

This project explores:

* high-dimensional image classification,
* large-scale deep learning workflows,
* custom image data pipelines,
* and scalable CNN training using TensorFlow/Keras.

---

# ⭐ Key Highlights

| Metric                 | Value                    |
| ---------------------- | ------------------------ |
| ✅ Landmark Classes     | **7,444**                |
| ✅ Trainable Parameters | **28.36 Million**        |
| ✅ Dataset Pipeline     | Custom batch generator   |
| ✅ Framework            | **TensorFlow / Keras**   |
| ✅ Architecture         | **VGG19 (from scratch)** |
| ✅ Image Processing     | **PIL + NumPy + Pandas** |

### 🔑 Core Concepts

`Computer Vision` • `CNN` • `VGG19` • `Deep Learning` • `Large-Scale Classification` • `TensorFlow` • `Custom Data Generators` • `Label Encoding` • `Batch Processing` • `Model Training`

---

# 🎯 Project Objective

The objective of this project was to build a landmark recognition system capable of handling an extremely large output space consisting of thousands of unique landmark categories.

The project focuses on:

* implementing VGG19 from random initialization,
* handling high-dimensional multi-class classification,
* building efficient image loading pipelines,
* and understanding the challenges of sparse real-world datasets.

While accuracy on this sparse, high-class dataset is naturally constrained, the project’s primary value lies in demonstrating large-scale data pipeline engineering, custom deep learning workflows, and baseline model experimentation — all essential components of real-world AI systems.

---

# 🧠 Dataset & Data Pipeline

## Dataset Source

* Google Landmarks Dataset (subset)
* Extracted from archived image files (`images_000.tar`)
* Landmark labels mapped via CSV metadata (`train.csv`)

---

## Real-World Data Challenges Solved

### ✅ Nested Directory Navigation

Images stored in multi-level paths:

```text
0/0/0/image.jpg
```

### ✅ Missing Image Handling

Implemented checks to skip corrupted or missing files safely.

### ✅ CSV-to-Image Mapping

Matched metadata entries with physically extracted images (8,266 valid pairs).

### ✅ Memory-Efficient Loading

Built a custom batch generator to stream images dynamically instead of loading the entire dataset into memory.

---

# 🧱 Model Architecture

```python
VGG19 (weights=None, include_top=False)

Architecture:
- 5 convolutional blocks
- BatchNormalization (added after block5_conv4)
- Flatten()
- Dense(256, activation='relu')
- Dropout(0.5)
- Dense(7444, activation='softmax')
```

---

## Why VGG19?

VGG19 was selected to:

* understand deep CNN behavior from scratch,
* experiment with large parameter spaces,
* and establish a baseline architecture before applying transfer learning.

The model contains:

* deep convolutional feature extraction,
* fully connected classification layers,
* and regularization using dropout and batch normalization.

---

# ⚙️ Training Configuration

| Parameter        | Value                    |
| ---------------- | ------------------------ |
| Optimizer        | RMSprop                  |
| Learning Rate    | 0.0001                   |
| Loss Function    | Categorical Crossentropy |
| Batch Size       | 32                       |
| Epochs           | 5                        |
| Validation Split | 20%                      |

---

## Label Processing

* `LabelEncoder`
* One-hot encoding for 7,444 classes

---

## Training Strategy

* Manual batch iteration using `train_on_batch`
* Dynamic image preprocessing (resize 224×224, normalize)
* Real-time batch loading from nested folders
* Generator-based streaming pipeline

---

# 📊 Technical Insights & Observations

This project highlighted several important deep learning engineering concepts.

## 🔹 Large Output Spaces are Difficult

Training across 7,444 classes creates sparse gradients and challenging optimization dynamics.

## 🔹 Data Distribution Matters

Many landmark classes contained very few examples, leading to strong class imbalance.

## 🔹 Transfer Learning is Powerful

Experiments demonstrated why pretrained architectures are often preferred in practical computer vision systems.

## 🔹 Efficient Data Pipelines are Critical

Custom generators enabled scalable processing of image datasets without exhausting system memory.

---

# 🏆 Challenges & Solutions

| Challenge                        | Solution                                |
| -------------------------------- | --------------------------------------- |
| Extremely high number of classes | One-hot encoding + custom batch loading |
| Nested image directories         | Dynamic path resolution                 |
| Missing/corrupted files          | Graceful error handling                 |
| Memory constraints               | Generator-based streaming               |
| Deep model instability           | Batch normalization + dropout           |

---

# 🛠️ Tech Stack

| Category         | Tools & Technologies             |
| ---------------- | -------------------------------- |
| Framework        | TensorFlow 2.x, Keras            |
| Architecture     | VGG19 (from scratch)             |
| Data Processing  | Pandas, NumPy, PIL               |
| Preprocessing    | Resize + normalization           |
| Environment      | Jupyter Notebook                 |
| Dataset Handling | CSV parsing + tarfile extraction |
| Version Control  | Git, GitHub                      |

---

# ⚙️ Quick Start

## Prerequisites

* Python 3.8+
* TensorFlow 2.x
* Jupyter Notebook

---

## Installation & Run

```bash
# Clone repository
git clone https://github.com/PriyanshiSingh19/Landmark-detection.git

cd Landmark-detection

# Install dependencies
pip install tensorflow pandas numpy matplotlib pillow scikit-learn

# Launch notebook
jupyter notebook Landmark-detection.ipynb
```

Run all notebook cells. The notebook will:

* extract images from `images_000.tar`,
* match CSV labels to extracted images,
* build and train VGG19 from scratch,
* and save the model as `trained_model.h5`.

---

## Note

This repository consists of a single Jupyter notebook containing the complete pipeline:

* data extraction,
* preprocessing,
* model definition,
* training,
* and evaluation.

---

# 📁 Project Structure

```text
Landmark-detection/
│
├── Landmark-detection.ipynb   # Complete project (code + output)
├── train.csv                  # Original labels (required)
├── images_000.tar             # Image archive (required)
└── trained_model.h5           # Saved model (generated after training)
```

---

# 🚀 Future Improvements

## 🔹 Model Enhancements

* Replace VGG19 with MobileNetV2 or EfficientNet
* Apply transfer learning using ImageNet weights
* Add learning rate scheduling
* Increase training epochs with early stopping

---

## 🔹 Dataset Improvements

* Reduce class sparsity
* Expand images per landmark
* Apply advanced augmentation

---

## 🔹 Deployment Extensions

* FastAPI inference API
* TensorFlow Lite export
* Real-time webcam predictions
* Docker containerization

---

# 📝 Resume-Ready Bullet Points

### Large-Scale Landmark Classification

Built a VGG19-based image classification system handling **7,444 landmark categories** using TensorFlow/Keras and custom data generators for scalable image streaming.

### Deep Learning Pipeline Engineering

Implemented CSV-to-image mapping, dynamic batch loading, nested directory traversal, and memory-efficient preprocessing for large-scale landmark datasets.

### High-Dimensional Classification

Applied label encoding and one-hot vectorization for thousands of classes while training a **28M-parameter CNN architecture** from scratch.

### Model Analysis & Optimization

Analyzed training limitations caused by class imbalance and sparse samples, documenting practical improvements involving transfer learning and scalable architectures.

---

# 👩‍💻 About This Project

This project demonstrates practical understanding of:

* Deep Learning
* CNN Architectures
* Large-Scale Image Classification
* Data Pipeline Engineering
* TensorFlow/Keras Workflows
* Model Training & Optimization

It reflects real-world challenges involved in training computer vision systems on large and imperfect datasets.

---

Built with 🧠 TensorFlow, deep CNN architectures, and curiosity for scalable computer vision systems.
