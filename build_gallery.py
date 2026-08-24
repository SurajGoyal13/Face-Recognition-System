from pathlib import Path

import cv2
import joblib
import numpy as np


DATASET_DIR = Path("dataset")
MODEL_PATH = "models/face_recognition_sface_2021dec.onnx"
OUTPUT_PATH = "models/face_gallery.pkl"

recognizer = cv2.FaceRecognizerSF.create(
    MODEL_PATH,
    ""
)

gallery = {}

for person_dir in sorted(DATASET_DIR.iterdir()):

    if not person_dir.is_dir():
        continue

    person = person_dir.name
    embeddings = []

    print(f"Processing {person}...")

    for image_path in person_dir.glob("*.jpg"):

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        # LFW images are already tightly cropped face images.
        # Use the SAME preprocessing for gallery images
        # that we will use for uploaded images.
        face = cv2.resize(image, (112, 112))

        feature = recognizer.feature(face)

        feature = np.asarray(
            feature,
            dtype=np.float32
        ).flatten()

        norm = np.linalg.norm(feature)

        if norm == 0:
            continue

        feature = feature / norm

        embeddings.append(feature)

    if embeddings:

        # Average embeddings of the same identity
        # to create one identity prototype.
        prototype = np.mean(
            np.vstack(embeddings),
            axis=0
        )

        prototype_norm = np.linalg.norm(prototype)

        if prototype_norm != 0:
            prototype = prototype / prototype_norm

        gallery[person] = prototype

        print(
            f"  {len(embeddings)} embeddings created"
        )


joblib.dump(
    gallery,
    OUTPUT_PATH
)

print("\nGallery rebuilt successfully.")
print(f"Identities: {len(gallery)}")