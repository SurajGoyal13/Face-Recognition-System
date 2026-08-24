from pathlib import Path

import joblib
import numpy as np
from skimage.io import imread

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


DATASET_DIR = Path("dataset")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

X = []
y = []

for person_dir in sorted(DATASET_DIR.iterdir()):
    if not person_dir.is_dir():
        continue

    person_name = person_dir.name
    print(f"Processing {person_name}...")

    for image_path in person_dir.glob("*.jpg"):
        image = imread(image_path)

        if image.ndim != 2:
            continue

        image = image.astype(np.float32) / 255.0

        if image.shape != (62, 47):
            continue

        X.append(image.flatten())
        y.append(person_name)

X = np.array(X)
y = np.array(y)

print(f"\nTotal usable samples: {len(X)}")
print(f"Total classes: {len(np.unique(y))}")

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale",
    probability=True
)

print("\nTraining SVM...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"\nAccuracy: {accuracy:.2%}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

joblib.dump(model, MODEL_DIR / "face_model.pkl")
joblib.dump(label_encoder, MODEL_DIR / "label_encoder.pkl")

print("\nModel saved successfully.")