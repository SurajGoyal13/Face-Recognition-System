# 👤 Face Recognition System

A computer vision-based face recognition system built with **Python, OpenCV, and Flask**. The application detects faces using **YuNet**, extracts deep face embeddings using **SFace**, and identifies known individuals by comparing face embeddings against a stored gallery using **cosine similarity**.

The project also includes a separate SVM-based training script for experimentation with traditional image-based classification. The Flask recognition application itself uses the **YuNet + SFace embedding gallery pipeline**.

## ✨ Key Features

- 👁️ Face detection using OpenCV YuNet
- 🧠 Deep face embedding extraction using OpenCV SFace
- 👥 Recognition of multiple known identities
- ❓ Unknown-person rejection using a similarity threshold
- 🖼️ Multiple-face detection in a single image
- 🔍 SFace embedding-based identity matching
- 📊 Similarity score display
- 🌐 Flask-based web application
- 📤 Image upload interface
- 🗂️ Stored face-embedding gallery for known identities

## 🛠️ Technology Stack

- **Python**
- **OpenCV**
- **YuNet**
- **SFace**
- **NumPy**
- **Flask**
- **Joblib**
- **HTML / CSS**

> **Note:** `scikit-learn` is used by the separate `train_model.py` SVM training script. The Flask recognition application does not use the SVM model for its recognition pipeline.

## 🧠 Recognition Pipeline

The main web application uses the following recognition approach:

```text
Input Image
     │
     ▼
YuNet Face Detection
     │
     ▼
Face Cropping & Preprocessing
     │
     ▼
SFace Feature Extraction
     │
     ▼
Embedding Normalization
     │
     ▼
Cosine Similarity
     │
     ▼
Compare Against Face Gallery
     │
     ├───────────────┐
     ▼               ▼
Known Person     Unknown Person
```

A similarity threshold of **0.45** is used for unknown-person rejection. Matches below this threshold are classified as **Unknown Person**.

The similarity value displayed by the application is derived from the cosine similarity score. It should be interpreted as a **similarity score**, not as a calibrated probability or confidence percentage.

## 🔄 How It Works

### 1. Face Detection

The uploaded image is processed using **OpenCV YuNet** to detect one or more faces.

### 2. Face Cropping & Preprocessing

For each detected face, the application crops the detected face region and resizes it to the input dimensions required by the SFace model.

### 3. Feature Extraction

**OpenCV SFace** generates a deep face embedding representing the detected face.

### 4. Embedding Comparison

The generated embedding is normalized and compared against the stored face-embedding gallery using **cosine similarity**.

### 5. Identity Matching

The highest similarity result is used for identity matching.

If the similarity score is below **0.45**, the face is rejected as an unknown person.

### 6. Result Display

The Flask web interface displays the recognized identity, similarity score, and results for each detected face.

## 🗂️ Face Gallery

The repository includes scripts for creating the stored face-embedding gallery.

The gallery-building process extracts SFace embeddings from images of known individuals, normalizes the embeddings, and stores the resulting identity representations using **Joblib**.

The recognition application loads this stored gallery and compares embeddings from newly uploaded images against the known identities.

## 🔀 Multiple-Face Recognition

The application can process multiple detected faces within a single uploaded image.

```text
Uploaded Image
      │
      ▼
Face Detection
      │
 ┌────┼────┐
 ▼    ▼    ▼
Face  Face  Face
 1     2     3
 │     │     │
 ▼     ▼     ▼
SFace SFace SFace
 │     │     │
 ▼     ▼     ▼
Match Match Match
```

Each detected face is evaluated independently and can produce its own identity and similarity score.

## 🧪 Separate SVM Training

The repository also contains a separate `train_model.py` script that uses **scikit-learn** to train an SVM classifier on grayscale face images.

This is separate from the main Flask recognition pipeline.

```text
SVM Training Script
        │
        ▼
Grayscale Face Images
        │
        ▼
Image Features
        │
        ▼
SVM Classifier
        │
        ▼
Training / Evaluation
```

The deployed Flask application does **not** use this SVM model for identity recognition. Its active recognition pipeline is based on **YuNet face detection + SFace embeddings + cosine similarity**.

## 📁 Project Structure

```text
Face-Recognition-System/
│
├── app.py                 # Flask web application and recognition pipeline
├── face_utils.py          # Face detection and recognition utilities
├── build_dataset.py       # Face dataset / embedding preparation
├── build_gallery.py       # Face-embedding gallery generation
├── train_model.py         # Separate SVM training and evaluation script
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html         # Flask web interface
├── models/
│   ├── face_detection_yunet.onnx
│   ├── face_recognition_sface.onnx
│   └── face_gallery.pkl
└── README.md
```

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/SurajGoyal13/Face-Recognition-System.git
cd Face-Recognition-System
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Flask Application

```bash
python app.py
```

Open the local Flask address shown in the terminal in your web browser.

## ▶️ Usage

1. Start the Flask application.
2. Open the web interface in your browser.
3. Upload an image containing one or more faces.
4. The application detects the faces using YuNet.
5. Each detected face is converted into an SFace embedding.
6. The embedding is compared against the stored gallery.
7. The application displays the matched identity and similarity score.
8. Faces below the similarity threshold are reported as **Unknown Person**.

## 🎯 Project Objective

The goal of this project is to demonstrate a practical **deep face-embedding-based recognition pipeline** using OpenCV's YuNet and SFace models.

The project combines:

- Face detection
- Face preprocessing
- Deep face embeddings
- Embedding normalization
- Cosine similarity
- Identity matching
- Unknown-person rejection
- Multiple-face processing
- Flask web application development

## ⚠️ Limitations

- Recognition performance depends on the quality and diversity of the images in the stored gallery.
- Similarity scores are not calibrated confidence probabilities.
- The recognition threshold is a predefined value of **0.45**.
- The system may be affected by changes in pose, lighting, image quality, occlusion, and other visual conditions.
- The face gallery must contain suitable reference images for the identities that should be recognized.
- The SVM training script is separate from the active Flask recognition pipeline.

The system should therefore be considered a **computer-vision demonstration and prototype rather than a production-grade biometric identification system**.

## 🔮 Future Improvements

- 📸 Improve robustness across different lighting and poses
- 🗂️ Add easier gallery management
- 👥 Support larger identity galleries
- 🎯 Tune the recognition threshold using validation data
- 🔐 Add authentication and access control
- 📊 Add recognition analytics and evaluation metrics
- 🧪 Add automated tests
- 🌐 Improve the web interface
- ⚡ Optimize recognition performance
- 🛡️ Add stronger privacy and security controls

## 📌 Technical Notes

- **YuNet** is used for face detection.
- **SFace** is used for deep face feature extraction.
- Face embeddings are normalized before similarity comparison.
- Identity matching uses cosine similarity against the stored face gallery.
- A similarity threshold of **0.45** is used for unknown-person rejection.
- **Joblib** is used to store/load the face gallery.
- The Flask application uses the SFace gallery-based recognition pipeline.
- `train_model.py` contains a separate scikit-learn SVM training approach and is not used by the Flask recognition pipeline.

## 👨‍💻 Author

**Suraj Goyal**

Computer Science Student · Python · AI/ML · Computer Vision · Web Development · DSA

---

⭐ **If you find this project useful, consider starring the repository.**
