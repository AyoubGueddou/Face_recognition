"""
@author: AyoubGueddou

Usage:
  train_classifier.py -e <face_encodings> -c <classifier> -t <test_image>
Options:
  -h, --help                          Show this help
  -e, --face_encodings                Path to load pickle face encodings
  -c, --classifier                    Path to save classifier
  -t, --test_image                    Path to test image
"""

import cv2
import dlib
import pickle
import argparse
import numpy as np
import face_recognition
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib import pyplot as plt
np.set_printoptions(precision=3)


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--face_encodings", required=True, type=str,
                help="path to face encodings")
ap.add_argument("-c", "--classifier", required=True, type=str,
                help="path to save classifier")
ap.add_argument("-t", "--test_image", type=str,
                help="path to test_image")
args = vars(ap.parse_args())

#
names = ["Tony", "Steve", "Bruce", "Thor", "Natasha", "Clint", "Hulk", "Wanda", "Vision", "Pietro", "Fury", "Loki"]

face_recognition_model = './models/dlib_face_recognition_resnet_model_v1.dat'
face_encoder = dlib.face_recognition_model_v1(face_recognition_model)

# load the known faces encodings
print("[INFO] loading encodings...")
data = pickle.loads(open(args['face_encodings'], "rb").read())
face_encodings = data['encodings']
ids = data['names']
print("[INFO] Encodings loaded.")

# Train the SVM Classifier
print("[INFO] Training model. It will take a few seconds. Wait ...")
# Train, test split
X_train, X_test, y_train, y_test = train_test_split(face_encodings, ids, test_size=0.2, random_state=0)

# Create and train the baseline SVC classifier
clf = SVC(gamma='scale')
clf.fit(X_train, y_train)
print("[INFO] Trained.")

y_pred = clf.predict(X_test)

print('\nConfusion Matrix\n', confusion_matrix(y_test, y_pred))
print('Accuracy\n', accuracy_score(y_test, y_pred))

# Optimized model
param_grid = [
    {'C': [1, 10, 100, 1000],
     'kernel': ['linear']},
    {'C': [1, 10, 100, 1000],
     'gamma': [0.001, 0.0001],
     'kernel': ['rbf']}
]

grid = GridSearchCV(SVC(probability=True), param_grid, n_jobs=-1, refit=True, cv=5)

# fitting the model for grid search
grid.fit(X_train, y_train)

# print best parameter after tuning
print(grid.best_params_)

# print how our model looks after hyper-parameter tuning
print(grid.best_estimator_)

# Testing
print("[INFO] Testing model. It will take a few seconds. Wait ...")
y_pred = grid.predict(X_test)

print('\nConfusion Matrix\n', confusion_matrix(y_test, y_pred))
print('Accuracy\n', accuracy_score(y_test, y_pred))

# Save model
with open(args['classifier'], 'wb') as file:
    pickle.dump(grid, file)
    print('[INFO] Model Saved')

if args['test_image']:

    # Load the test image with unknown faces into a numpy array
    test_image = cv2.imread(args['test_image'])
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

    # Find all the faces in the test image using the default HOG-based model
    face_locations = face_recognition.face_locations(test_image, 2)
    no = len(face_locations)
    print("Number of faces detected: ", no)

    # Predict all the faces in the test image using the trained classifier
    print("Found:")
    for i in range(no):
        (top, right, bottom, left) = face_locations[i]
        detected_face = test_image[top:bottom, left:right]
        detected_face = cv2.resize(detected_face, (150, 150))
        plt.imshow(detected_face)
        plt.show()
        test_image_enc = np.array(face_encoder.compute_face_descriptor(detected_face))
        # name = clf.predict([test_image_enc])
        proba = grid.predict_proba([test_image_enc])
        if np.max(proba) > 0.35:
            name = names[np.argmax(proba)]
        else:
            name = 'Inconnu'
        print(name)
        print(proba)

        # Draw a box around the face
        cv2.rectangle(test_image, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(test_image, (left, bottom - 10), (right, bottom + 15), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(test_image, name, (left + 2, bottom + 10), font, 0.5, (255, 255, 255), 1)

    plt.figure(figsize=(15, 15))
    plt.imshow(test_image)
    plt.show()
