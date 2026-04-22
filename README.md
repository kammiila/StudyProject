# Customer Segmentation using KMeans

## Overview
This project implements **customer segmentation** using an unsupervised machine learning algorithm — **KMeans clustering**.

The goal is to group customers based on their behavior and characteristics without predefined labels.

---

## Objectives
- Analyze customer data
- Identify distinct customer groups (clusters)
- Build a clustering model using KMeans
- Provide predictions via CLI and web interface

---

## 🛠️ Technologies Used
- Python 3
- pandas
- numpy
- scikit-learn
- matplotlib
- flask

---

# 📂 Project Structure

StudyProject/
│
├── train.py # Model training
├── test.py # CLI testing
├── app.py # Web application (Flask)
│
├── customer_data.csv # Input dataset
├── final_dataset.csv # Processed dataset
│
├── static/
│ └── cluster_plot.png # Cluster visualization
│
└── templates/
└── index.html # Web UI

---

## ⚙️ Installation

Clone the repository or download the project, then install dependencies:

```bash
pip3 install -r requirements.txt
