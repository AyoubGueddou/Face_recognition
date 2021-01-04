"""
@author: AyoubGueddou

Usage:
  encode_faces.py -f <face_folder> -e <face_encodings_path>
Options:
  -h, --help                          Show this help
  -f, --face_folder                   Path to the folder containing face dataset
  -e, --face_encodings                Path to save pickle face encodings
"""

import cv2
import numpy as np
import os
import dlib
import pickle
import argparse


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face_folder", required=True, type=str,
                help="Path to the folder containing face dataset")
ap.add_argument("-e", "--face_encodings", required=True, type=str,
                help="Path to save pickle face encodings")
args = vars(ap.parse_args())

#
names = ["Tony", "Steve", "Bruce", "Thor", "Natasha", "Clint", "Hulk", "Wanda", "Vision", "Pietro", "Fury", "Loki"]

face_recognition_model = './models/dlib_face_recognition_resnet_model_v1.dat'
face_encoder = dlib.face_recognition_model_v1(face_recognition_model)


# function to get the images and label data
def load_faces_id(path_, names_):
    id = 0
    images = [os.path.join(path_, name) for name in names_]
    faces_ = []
    ids = []

    for image in images:
        onlyfiles = [f for f in os.listdir(image) if os.path.isfile(os.path.join(image, f))]
        print(len(onlyfiles), 'face images for ', names[id])
        for file in onlyfiles:
            frame = cv2.imread(os.path.join(image, file))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.resize(frame, (150, 150))
            faces_.append(gray)
            ids.append(id)

        id += 1

    return faces_, ids


print("[INFO] Loading faces. It will take a few seconds. Wait ...")
faces, ids = load_faces_id(args['face_folder'], names)
print("[INFO] Loaded.")

print("[INFO] Encoding faces. It will take a few seconds. Wait ...")
# face_encodings = [np.array(face_encoder.compute_face_descriptor(face)) for face in faces]
face_encodings = np.array(face_encoder.compute_face_descriptor(faces))
print("[INFO] Encoded.")

# dump the facial encodings + names to disk
print("[INFO] serializing encodings...")
data = {"encodings": face_encodings, "names": ids}
f = open(args['face_encodings'], "wb")
f.write(pickle.dumps(data))
f.close()
print("[INFO] Encodings saved")
