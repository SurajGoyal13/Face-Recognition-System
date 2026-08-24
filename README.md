# Face Recognition System

A computer vision and machine learning based face recognition system that detects faces in images and identifies known individuals using deep face embeddings and similarity matching.

## Features

- Face detection using OpenCV YuNet
- Face feature extraction using OpenCV SFace
- Recognition of multiple known identities
- Unknown-person rejection
- Multiple-face detection in a single image
- Flask-based web application
- Confidence/similarity score display
- Image upload interface
- Machine learning based identity matching

## Technology Stack

- Python
- OpenCV
- YuNet
- SFace
- NumPy
- Scikit-learn
- Flask
- Joblib
- HTML/CSS

## System Workflow

```text
Input Image
     ↓
Face Detection
     ↓
Face Alignment / Preprocessing
     ↓
SFace Feature Extraction
     ↓
Similarity Comparison
     ↓
Known Person / Unknown Person
     ↓
Flask Web Interface