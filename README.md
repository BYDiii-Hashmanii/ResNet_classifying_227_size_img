# ♻️ Garbage Image Classifier — CNN & ResNet50 on 227×227 Images

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-CNN%20%7C%20ResNet50-red?logo=keras&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-green?logo=opencv&logoColor=white)
![Classes](https://img.shields.io/badge/Classes-6%20Waste%20Types-blueviolet)
![Input](https://img.shields.io/badge/Input-227×227%20RGB-informational)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A complete image classification pipeline for **waste/garbage sorting** using two deep learning architectures: a **custom 4-block CNN (LeNet-style)** and a **hand-built ResNet50 from scratch**. Images of 6 waste categories are preprocessed, serialized to CSV, and fed into both models for training and evaluation.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset & Classes](#-dataset--classes)
- [Image Preprocessing Pipeline](#-image-preprocessing-pipeline)
- [Model 1 — Custom CNN (LeNet-Style)](#-model-1--custom-cnn-lenet-style)
- [Model 2 — ResNet50 (From Scratch)](#-model-2--resnet50-from-scratch)
- [Training Configuration](#-training-configuration)
- [Project Structure](#-project-structure)
- [Requirements](#-requirements)
- [Usage](#-usage)
- [Key Design Decisions](#-key-design-decisions)

---

## 🧠 Overview

This project tackles **automated garbage classification** — a critical step toward smart recycling systems. The pipeline:

1. **Reads** raw `.jpg` images from categorized folders on Google Drive
2. **Resizes** each image to `227×227` and **flattens** pixel data into CSV rows
3. **Loads** the CSV and reshapes into `(N, 110, 110, 3)` or `(N, 227, 227, 3)` tensors
4. **Trains** two architectures — a lightweight CNN and a full ResNet50 — on 6 waste categories
5. **Evaluates** both models on a held-out 20% test split

---

## 📂 Dataset & Classes

### Class Mapping

| Label ID | Category |
|----------|----------|
| 0 | `cardboard` |
| 1 | `glass` |
| 2 | `metal` |
| 3 | `paper` |
| 4 | `plastic` |
| 5 | `trash` |

### Dataset Stats

| Property | Value |
|---|---|
| Total Samples | ~2,526 images |
| Train Split | 80% (~2,020 samples) |
| Test Split | 20% (~506 samples) |
| Image Resolution | 227 × 227 × 3 (RGB) |
| CSV Size | `Bewerage_227.csv` — one row per image |
| Storage | Google Drive (`Garbage classification/`) |

### Folder Structure (Google Drive)

```
MyDrive/
├── Garbage classification/
│   ├── cardboard/   *.jpg
│   ├── glass/       *.jpg
│   ├── metal/       *.jpg
│   ├── paper/       *.jpg
│   ├── plastic/     *.jpg
│   └── trash/       *.jpg
├── Bewerage_110.csv       ← 110×110 flattened CSV
└── Bewerage_227.csv       ← 227×227 flattened CSV
```

---

## 🔄 Image Preprocessing Pipeline

```
Raw .jpg files (per-class folders)
           │
           ▼
    cv2.imread()          ← Load in BGR
           │
           ▼
  cv2.resize(227, 227)    ← Standardize spatial dims
           │
           ▼
  image.flatten()         ← (227×227×3) = 154,587 values
           │
           ▼
  Prepend class label     ← label_dic[class_name]
           │
           ▼
  Write row to CSV        ← Bewerage_227.csv
           │
           ▼
  pd.read_csv()           ← Reload for training
           │
           ▼
  drop('0') / astype / /255   ← Normalize to [0, 1]
           │
           ▼
  reshape(N, 110, 110, 3)     ← For CNN model
  reshape(N, 227, 227, 3)     ← For ResNet50 model
           │
           ▼
  train_test_split (80/20)
           │
           ▼
  to_categorical (6 classes)  ← One-hot encode labels
```

---

## 🏗️ Model 1 — Custom CNN (LeNet-Style)

A 4-block convolutional network operating on `(110, 110, 3)` inputs.

```
Input: (110, 110, 3)
        │
        ▼
┌──────────────────────────┐
│  Conv2D(32, 3×3, valid)  │
│  BatchNormalization       │
│  Activation('relu')       │
│  MaxPooling2D(2×2)        │
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│  Conv2D(64, 3×3, valid)  │
│  BatchNormalization       │
│  Activation('relu')       │
│  MaxPooling2D(2×2)        │
└────────────┬─────────────┘
             ▼
┌───────────────────────────┐
│  Conv2D(128, 3×3, valid)  │
│  BatchNormalization        │
│  Activation('relu')        │
│  MaxPooling2D(2×2)         │
└────────────┬──────────────┘
             ▼
┌───────────────────────────┐
│  Conv2D(256, 3×3, valid)  │
│  BatchNormalization        │
│  Activation('relu')        │
│  MaxPooling2D(2×2)         │
└────────────┬──────────────┘
             ▼
          Flatten()
             ▼
      Dense(128, relu)
             ▼
      Dense(6, softmax)
             │
             ▼
     Output: 6 classes
```

**Optimizer:** SGD  
**Loss:** Categorical Crossentropy  
**Epochs:** 20  
**Input Shape:** `(110, 110, 3)`

---

## 🏗️ Model 2 — ResNet50 (From Scratch)

A full **ResNet50** implementation built manually using TensorFlow/Keras functional API — no pretrained weights, trained end-to-end on the garbage dataset.

### Building Blocks

#### Identity Block
Used when input/output dimensions match — adds the shortcut directly:

```
X ──────────────────────────────────────────┐
│                                           │
Conv2D(F1, 1×1) → BN → ReLU               │ (shortcut)
Conv2D(F2, f×f, same) → BN → ReLU         │
Conv2D(F3, 1×1) → BN                       │
│                                           │
└───────────── Add ◄────────────────────────┘
                │
              ReLU
```

#### Convolutional Block
Used when dimensions change — projects the shortcut with a 1×1 conv:

```
X ──────────────────────────────────────────────┐
│                                               │
Conv2D(F1, 1×1, s) → BN → ReLU               Conv2D(F3, 1×1, s) → BN
Conv2D(F2, f×f, same) → BN → ReLU             (shortcut projection)
Conv2D(F3, 1×1) → BN                           │
│                                               │
└──────────────────── Add ◄────────────────────┘
                       │
                     ReLU
```

### Full ResNet50 Architecture

```
Input: (227, 227, 3)
        │
        ▼
ZeroPadding2D(3,3)
Conv2D(64, 7×7, stride=2) → BN → ReLU
MaxPooling2D(3×3, stride=2)
        │
        ▼
Stage 2: ConvBlock([64,64,256], s=1)
         IdentityBlock × 2
        │
        ▼
Stage 3: ConvBlock([128,128,512], s=2)
         IdentityBlock × 3
        │
        ▼
Stage 4: ConvBlock([256,256,1024], s=2)
         IdentityBlock × 5
        │
        ▼
Stage 5: ConvBlock([512,512,2048], s=2)
         IdentityBlock × 2
        │
        ▼
GlobalAveragePooling2D()
        │
        ▼
Dense(6, softmax)
        │
        ▼
Output: 6 classes
```

**Input Shape:** `(227, 227, 3)`  
**Total Residual Blocks:** 16 (mix of identity + convolutional)

---

## ⚙️ Training Configuration

| Setting | CNN (LeNet) | ResNet50 |
|---|---|---|
| Input Size | 110 × 110 × 3 | 227 × 227 × 3 |
| Optimizer | SGD | — |
| Loss | Categorical Crossentropy | Categorical Crossentropy |
| Epochs | 20 | — |
| Batch Size | Default (Keras) | — |
| Classes | 6 | 6 |
| Train Split | 80% | 80% |
| Test Split | 20% | 20% |
| Normalization | `/255` | `/255` |

---

## 📁 Project Structure

```
garbage-image-classifier/
│
├── 📓 Handling_227_images.ipynb    # Full pipeline: preprocessing + CNN + ResNet50
│
├── 📁 Garbage classification/      # Raw images (Google Drive)
│   ├── cardboard/
│   ├── glass/
│   ├── metal/
│   ├── paper/
│   ├── plastic/
│   └── trash/
│
├── 📄 Bewerage_110.csv             # Flattened 110×110 image data
├── 📄 Bewerage_227.csv             # Flattened 227×227 image data
└── README.md
```

---

## ⚙️ Requirements

```txt
tensorflow>=2.x
keras
opencv-python
numpy
pandas
scikit-learn
matplotlib
```

Install dependencies:

```bash
pip install tensorflow opencv-python numpy pandas scikit-learn matplotlib
```

For Google Colab:

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## 🚀 Usage

### Step 1 — Build the CSV Dataset

```python
import os, cv2, csv

label_dic = {'cardboard':0, 'glass':1, 'metal':2, 'paper':3, 'plastic':4, 'trash':5}

with open('Bewerage_227.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for cls, label in label_dic.items():
        path = f'Garbage classification/{cls}'
        for r, d, files in os.walk(path):
            for filename in files:
                if filename.endswith('.jpg'):
                    image = cv2.imread(os.path.join(r, filename))
                    image = cv2.resize(image, (227, 227))
                    row = [label] + list(image.flatten())
                    writer.writerow(row)
```

### Step 2 — Train the CNN (LeNet-Style)

```python
import pandas as pd
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

data = pd.read_csv('Bewerage_110.csv', low_memory=False)
x = data.drop(columns=['0']).astype('float') / 255
y = data['0']

x = x.to_numpy().reshape(-1, 110, 110, 3)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
y_train = to_categorical(y_train, num_classes=6)
y_test  = to_categorical(y_test,  num_classes=6)

le_net_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
le_net_model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, verbose=1)
```

### Step 3 — Train ResNet50

```python
data_227 = pd.read_csv('Bewerage_227.csv', low_memory=False)
x = data_227.drop(columns=['0']).astype('float') / 255
y = data_227['0']
x = x.to_numpy().reshape(-1, 227, 227, 3)

model = ResNet50_227(input_shape=(227, 227, 3), classes=6)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20)
```

---

## 🔬 Key Design Decisions

| Decision | Reason |
|---|---|
| **CSV serialization** | Converts image folders into a single portable file, compatible with pandas-based pipelines |
| **Two resolutions (110 & 227)** | 110×110 for the lightweight CNN; 227×227 matches the original AlexNet/ResNet input standard |
| **Pixel `/255` normalization** | Scales RGB values to `[0, 1]` — standard for CNN convergence stability |
| **ResNet50 from scratch** | Demonstrates understanding of residual connections without relying on pretrained weights |
| **BatchNormalization in every block** | Stabilizes gradient flow across deep layers |
| **SGD for CNN, Adam for ResNet50** | SGD is simpler for shallow nets; Adam adapts learning rates for deep residual networks |

---

## ⚠️ Notes & Recommendations

- **CSV size warning**: A 227×227×3 image flattens to ~154K values per row. With 2,500+ images, the CSV can exceed 1–2 GB — consider HDF5 or TFRecord for larger datasets
- **Class imbalance**: If category folder sizes differ significantly, consider weighted loss or oversampling
- **Data augmentation**: Adding random flips, rotations, and brightness jitter can significantly boost generalization
- **Model checkpointing**: Add `ModelCheckpoint` to save the best epoch weights during training

---

## 💡 Gig Title Suggestions

Here are specific, high-converting titles for your freelance gig:

> **🥇 Best Pick:**
> `I will build a custom CNN or ResNet50 image classifier using TensorFlow and Keras`

> **Alternative Options:**
> - `I will train a deep learning model to classify images into multiple categories`
> - `I will build and train a garbage or product image classification model from scratch`
> - `I will implement ResNet50 from scratch for your custom image dataset`
> - `I will create an end-to-end image classification pipeline with CNN and deep learning`

---

## 📄 License

This project is licensed under the **MIT License**.

---

> Built with  Bydiii ❤️ using TensorFlow, Keras, OpenCV, and Python on Google Colab
