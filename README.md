
# Home Assignment 5 – CS5720 Neural Network and Deep Learning

**Student Name:** Akhil Dondapati
**Course:** CS5720 – Neural Network and Deep Learning  

**University:** University of Central Missouri  
**Semester:** Spring 2025  

---

## 📘 Overview

This repository includes solutions for two programming tasks from **Home Assignment 5**:

1. **Basic GAN Implementation using PyTorch**  
2. **Data Poisoning Simulation using scikit-learn**

Each task includes source code, model explanation, sample outputs, and visualizations as required by the assignment.

---

## 🧪 Question 3 – Basic GAN (MNIST)

This task implements a **Generative Adversarial Network (GAN)** to generate handwritten digits from the MNIST dataset using PyTorch.

### 🧠 Architecture

- **Generator**:
  - Input: 100-dim random noise
  - Layers: Linear → ReLU → Linear → Tanh
  - Output: Flattened 28x28 fake digit

- **Discriminator**:
  - Input: Flattened 28x28 image
  - Layers: Linear → LeakyReLU → Linear → Sigmoid
  - Output: Probability (real or fake)

### 🏃 How to Run
```bash
pip install torch torchvision
python gan_mnist.py

# Data Poisoning Simulation – Sentiment Classifier (Question 4)

**Course:** CS5720 Neural Network and Deep Learning  
**Assignment:** Home Assignment 5 – Question 4  

**University:** University of Central Missouri  
**Semester:** Spring 2025  

---

## 📝 Overview

This project demonstrates a **data poisoning attack** on a basic sentiment classification model using logistic regression. It shows how modifying just a few training samples related to a specific keyword ("UC Berkeley") can cause the model to perform worse, highlighting the importance of secure and clean training data in machine learning.

---

## 💡 Objective

- Train a sentiment classifier using clean labeled text data.
- Poison the data by flipping labels of specific samples mentioning "UC Berkeley".
- Compare the accuracy and confusion matrices before and after poisoning.

---

## 📄 Dataset

A custom dataset of 20 movie reviews:
- 10 positive reviews (label = 1)
- 10 negative reviews (label = 0)
- 3 reviews that mention “UC Berkeley”

---

## 🧠 Methodology

1. **Clean Training:** Train a `LogisticRegression` classifier using clean labels.
2. **Label Flipping:** Identify reviews with "UC Berkeley" and flip their sentiment labels.
3. **Retrain Model:** Train the classifier again with poisoned data.
4. **Evaluation:** Measure accuracy and plot confusion matrices before and after poisoning.

---

## 🛠️ Requirements

Install the necessary Python libraries:
```bash
pip install pandas scikit-learn matplotlib seaborn

