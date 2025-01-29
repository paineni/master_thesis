# Master Thesis: Enhancing Object Detection in Large Images through Efficient Processing and Evaluation

## Research Overview

This thesis explores innovative approaches to improve object detection in large images, particularly for applications in **remote sensing** and **quality control**. The research focuses on balancing **high image resolution** with **computational efficiency**, addressing challenges such as hardware limitations, processing time, and detection accuracy. The study evaluates three methods: a **baseline approach**, **SAHI-enhanced YOLOv8**, and a **unified YOLOv8-VGG-16 model**, with a specific focus on cocoa bean quality control.

---

## Research Problem

When applying neural networks to tasks like remote sensing or quality control, there is a critical need to balance **image resolution**, **computational resources**, and the **level of detail** required for accurate object detection. Traditional methods often involve resizing or cropping images, which can lead to information loss or inefficiencies. The central research question is:

**"How can we improve object detection in large images while optimizing hardware usage, processing time, and model performance?"**

---

## Purpose and Significance

This research is significant because it addresses a key challenge in computer vision: **detecting objects in high-resolution images** while managing **computational constraints**. The findings aim to enhance object detection accuracy and efficiency, making it applicable to industries like **remote sensing**, **agriculture**, and **quality control**, where large images are common.

---

## Scope and Limitations

The research focuses on:
- Comparing **downscaling**, **tiling**, and **unified detection-classification** methods for handling large images.
- Evaluating trade-offs in **memory consumption**, **processing time**, and **model performance**.
- Using a dataset of cocoa bean images to benchmark the methods.

Limitations include:
- High computational requirements for some methods (e.g., SAHI-enhanced YOLOv8).
- Challenges in detecting small or split objects across image slices.

---

## Methodology

The research employs a **comparative analysis** of three approaches:
1. **Baseline Method**: A two-step approach using YOLOv8 for object detection and VGG-16 for classification.
2. **SAHI-Enhanced YOLOv8 (Method 1)**: Integrates Slicing Aided Hyper Inference (SAHI) with YOLOv8 to improve detection accuracy, especially for small objects.
3. **Unified YOLOv8 and VGG-16 (Method 2)**: Combines YOLOv8 and VGG-16 into a single training loop for joint detection and classification.

---

## Results Summary

### 1. **Detection Accuracy**
- **Unified YOLOv8 and VGG-16 (Method 2)** achieved the highest accuracy for detecting valid cocoa objects with an **AP of 79%**.
- **SAHI-Enhanced YOLOv8 (Method 1)** showed the best performance in detecting invalid objects, with an **AP of 23.4%**.
- The **Baseline Method** had moderate performance, with an **AP of 77.8%** for cocoa objects but struggled with invalid objects (AP of 5.8%).

### 2. **Mean Average Precision (mAP)**
- **SAHI-Enhanced YOLOv8** achieved the highest **mAP of 50.1%**.
- The **Unified YOLOv8 and VGG-16** method had a slightly lower **mAP of 41.4%**, while the **Baseline Method** achieved a **mAP of 41.8%**.

### 3. **Training Efficiency**
- **Memory Usage**:
  - **SAHI-Enhanced YOLOv8**: **5.67 GB** (most memory-efficient).
  - **Unified YOLOv8 and VGG-16**: **6.75 GB** (highest memory usage).
  - **Baseline Method**: **6.36 GB**.
  
- **Training Time**:
  - **Baseline Method**: **0.6 hours** (fastest).
  - **SAHI-Enhanced YOLOv8**: **9.2 hours** (slowest due to slicing and stitching).
  - **Unified YOLOv8 and VGG-16**: **7 hours**.

### 4. **Inference Performance**
- **Inference Time**:
  - **Baseline Method**: **1.69 seconds per image** (fastest).
  - **SAHI-Enhanced YOLOv8**: **4.7 seconds per image** (slowest).
  - **Unified YOLOv8 and VGG-16**: **1.94 seconds per image**.
  
- **Memory Consumption**:
  - **SAHI-Enhanced YOLOv8**: **3.58 GB** (highest memory usage).
  - **Baseline Method**: **1.28 GB** (most memory-efficient).
  - **Unified YOLOv8 and VGG-16**: **2.35 GB**.

### 5. **Challenges and Limitations**
- **Baseline Method**: Dependency between the detector and classifier requires retraining both models if one is modified.
- **SAHI-Enhanced YOLOv8**: Struggles with objects split across multiple slices, leading to reduced accuracy. It also has high computational overhead and slower inference times.
- **Unified YOLOv8 and VGG-16**: High memory usage and complex training process, making it less suitable for edge devices.

---

## Conclusion

- **Unified YOLOv8 and VGG-16** is ideal for high-accuracy tasks with sufficient computational resources.
- **SAHI-Enhanced YOLOv8** is robust for detecting a wide range of objects but is resource-intensive.
- The **Baseline Method** is the most suitable for edge devices due to its low memory usage and fast inference times.

---

## Future Work

- Optimize the **Non-Maximum Suppression (NMS)** threshold for better classification accuracy.
- Experiment with different batch sizes for VGG-16 to improve training efficiency.
- Explore architectural integration to enhance the synergy between YOLOv8 and VGG-16.
- Test different model combinations to find the optimal configuration for complex object detection tasks.

---

## How to Use This Repository

1. Clone the repository:
   ```bash
   git clone https://github.com/paineni/master_thesis.git


