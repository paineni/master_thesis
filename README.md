# Master Thesis: Enhancing Object Detection in Large Images through Efficient Processing and Evaluation  
![Python](https://img.shields.io/badge/Python-3.10.10-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red?style=for-the-badge&logo=pytorch)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange?style=for-the-badge&logo=mlflow)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8-brightgreen?style=for-the-badge&logo=ultralytics)
![FiftyOne](https://img.shields.io/badge/FiftyOne-Dataset--Visualization-purple?style=for-the-badge&logo=fiftyone)
![SAHI](https://img.shields.io/badge/SAHI-Sliced%20Inference-lightblue?style=for-the-badge)
![License](https://img.shields.io/badge/License-Public-green?style=for-the-badge)
![Open Source Love](https://img.shields.io/badge/%E2%9D%A4%EF%B8%8F-Open%20Source-pink?style=for-the-badge)

## Research Overview 🧑‍🔬  
This thesis explores innovative approaches to improve object detection in large images, particularly for applications in **remote sensing** and **quality control**. The research focuses on balancing **high image resolution** with **computational efficiency**, addressing challenges such as hardware limitations, processing time, and detection accuracy. The study evaluates three methods: a **baseline approach**, **SAHI-enhanced YOLOv8**, and a **unified YOLOv8-VGG-16 model**, with a specific focus on cocoa bean quality control.

---

## Research Problem ❓  
When applying neural networks to tasks like remote sensing or quality control, there is a critical need to balance **image resolution**, **computational resources**, and the **level of detail** required for accurate object detection. Traditional methods often involve resizing or cropping images, which can lead to information loss or inefficiencies. The central research question is:

**"How can we improve object detection in large images while optimizing hardware usage, processing time, and model performance?"**

---

## Purpose and Significance 🌍  
This research is significant because it addresses a key challenge in computer vision: **detecting objects in high-resolution images** while managing **computational constraints**. The findings aim to enhance object detection accuracy and efficiency, making it applicable to industries like **remote sensing**, **agriculture**, and **quality control**, where large images are common.

---

## Scope and Limitations 📊  
The research focuses on:
- Comparing **downscaling**, **tiling**, and **unified detection-classification** methods for handling large images.
- Evaluating trade-offs in **memory consumption**, **processing time**, and **model performance**.
- Using a dataset of cocoa bean images to benchmark the methods.

**Limitations**:
- High computational requirements for some methods (e.g., SAHI-enhanced YOLOv8).
- Challenges in detecting small or split objects across image slices.

---

## Methodology 🛠️  
The research employs a **comparative analysis** of three approaches:
1. **Baseline Method**: A two-step approach using YOLOv8 for object detection and VGG-16 for classification.
2. **SAHI-Enhanced YOLOv8 (Method 1)**: Integrates Slicing Aided Hyper Inference (SAHI) with YOLOv8 to improve detection accuracy, especially for small objects.
3. **Unified YOLOv8 and VGG-16 (Method 2)**: Combines YOLOv8 and VGG-16 into a single training loop for joint detection and classification.

---

## Results Summary 📈

### 1. **Detection Accuracy**
- **Unified YOLOv8 and VGG-16 (Method 2)**: **AP of 79%** for detecting valid cocoa objects.
- **SAHI-Enhanced YOLOv8 (Method 1)**: **AP of 23.4%** for detecting invalid objects.
- **Baseline Method**: **AP of 77.8%** for valid cocoa objects, but only **5.8%** for invalid ones.

### 2. **Mean Average Precision (mAP)**
- **SAHI-Enhanced YOLOv8**: **50.1% mAP** (Best).
- **Unified YOLOv8 and VGG-16**: **41.4% mAP**.
- **Baseline Method**: **41.8% mAP**.

### 3. **Training Efficiency**  
- **Memory Usage**:
  - **SAHI-Enhanced YOLOv8**: **5.67 GB**.
  - **Unified YOLOv8 and VGG-16**: **6.75 GB**.
  - **Baseline Method**: **6.36 GB**.
  
- **Training Time**:
  - **Baseline Method**: **0.6 hours** (fastest).
  - **SAHI-Enhanced YOLOv8**: **9.2 hours** (slowest).
  - **Unified YOLOv8 and VGG-16**: **7 hours**.

### 4. **Inference Performance**  
- **Inference Time**:
  - **Baseline Method**: **1.69 seconds/image**.
  - **SAHI-Enhanced YOLOv8**: **4.7 seconds/image**.
  - **Unified YOLOv8 and VGG-16**: **1.94 seconds/image**.
  
- **Memory Consumption**:
  - **SAHI-Enhanced YOLOv8**: **3.58 GB**.
  - **Baseline Method**: **1.28 GB**.
  - **Unified YOLOv8 and VGG-16**: **2.35 GB**.

### 5. **Challenges and Limitations**
- **Baseline Method**: Requires retraining both YOLOv8 and VGG-16 if one model is modified.
- **SAHI-Enhanced YOLOv8**: Difficulty detecting objects split across slices and slower inference times.
- **Unified YOLOv8 and VGG-16**: High memory usage and complex training process.

---

## Conclusion 🎯  
- **Unified YOLOv8 and VGG-16**: Best for high-accuracy tasks with sufficient computational resources.
- **SAHI-Enhanced YOLOv8**: Great for detecting a range of objects, but resource-intensive.
- **Baseline Method**: Most suitable for edge devices due to low memory usage and fast inference times.

---

## Future Work 🚀
- Optimize **Non-Maximum Suppression (NMS)** threshold for better accuracy.
- Experiment with different batch sizes for VGG-16 to improve training efficiency.
- Explore architectural integration between YOLOv8 and VGG-16.
- Test different model combinations for complex object detection tasks.

---

# 🚀 How to Run This Project


---

## 🏆 Unified YOLOv8 and VGG16

🔹 **Step 1:** Create a new environment:
   ```bash
   conda env create -f env.yml
   ```  
🔹 **Step 2:** Clone the repository:
   ```bash
   git clone https://github.com/paineni/master_thesis.git
   cd master_thesis
   pip install -e .
   ```  
🔹 **Step 3:** Make sure your data is in YOLOv8 format and then run:
   ```bash
   python yolo_train.py
   ```

---

## 🔥 SAHI-Enhanced YOLOv8 Model

🔹 **Step 1:** Create a new environment:
   ```bash
   conda env create -f env.yml
   ```  
🔹 **Step 2:** Install the related dependencies:
   ```bash
   pip install ultralytics sahi
   ```  
🔹 **Step 3:** Follow the instructions in the [SAHI repository](https://github.com/obss/sahi) to preprocess the data into slices.  
🔹 **Step 4:** Train the SAHI-enhanced YOLOv8 model by modifying the path in `yolo_train.py` and running:
   ```bash
   python yolo_train.py
   ```

---

## 🎯 Baseline Approach (2-Step Approach)

🔹 **Step 1:** Create a new environment:
   ```bash
   conda env create -f env.yml
   ```  
🔹 **Step 2:** Install the related dependencies:
   ```bash
   pip install ultralytics
   ```  
🔹 **Step 3:** Extract patches using `patches_extraction.ipynb`, then train YOLOv8 and VGG16 separately:
   ```bash
   python yolo_train.py
   python vgg16.py
   ```

---

## 🎯 Inferencing

🔹 Inference can be performed using either:
- `joint_inferencing.py` 📌
- `data_handling.ipynb` 🛠️

🚀 **Happy Coding!** 🎉


