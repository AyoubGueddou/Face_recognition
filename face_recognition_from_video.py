"""
@author: AyoubGueddou

Usage:
  face_recognition_from_video.py -c <trained_classifier> -t <test_image> -d <display>
Options:
  -h, --help                          Show this help
  -c, --classifier                    Path to SVM classifier trained on face encodings
  -v, --test_video                    Path to test video to process
  -o, --output_video                  Path to save output video
  -d  --display                       Display processed frames
"""

import cv2
import dlib
import pickle
import argparse
import numpy as np
import face_recognition


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--classifier", required=True, type=str,
                help="path to load classifier")
ap.add_argument("-v", "--test_video", type=str,
                help="path to test_video to process")
ap.add_argument("-o", "--output_video", type=str,
                help="path to save output video")
ap.add_argument("-d", "--display",  type=int, default=1,
                help="display processed frames")
args = vars(ap.parse_args())

names = ["Tony", "Steve", "Bruce", "Thor", "Natasha", "Clint", "Hulk", "Wanda", "Vision", "Pietro", "Fury", "Loki"]

face_recognition_model = './models/dlib_face_recognition_resnet_model_v1.dat'
face_encoder = dlib.face_recognition_model_v1(face_recognition_model)

path_video = args['test_video']
path_output = args['output_video']
nb_frame = 0

# load the known faces encodings
print("[INFO] Loading classifier...")
clf = pickle.loads(open(args['classifier'], "rb").read())
print("[INFO] Classifier loaded.")

# Check if user want to use webcam
if path_video.isnumeric() and int(path_video) == 0:
    path_video = int(path_video)

writer = None
video_capture = cv2.VideoCapture(path_video)
total = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

print("[INFO] Processing video. It will take a few seconds/minutes. Wait ...")
while True:
    nb_frame += 1
    ret, frame = video_capture.read()

    if ret:
        test_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all the faces in the test image using the default HOG-based model
        face_locations = face_recognition.face_locations(test_image, 1, model='hog')
        no = len(face_locations)

        # Predict all the faces in the test image using the trained classifier
        for i in range(no):
            (top, right, bottom, left) = face_locations[i]
            detected_face = test_image[top:bottom, left:right]
            detected_face = cv2.resize(detected_face, (150, 150))
            # plt.imshow(detected_face)
            # plt.show()
            test_image_enc = np.array(face_encoder.compute_face_descriptor(detected_face))
            # id = clf.predict([test_image_enc])
            # name = names[int(id)]
            proba = clf.predict_proba([test_image_enc])
            if np.max(proba) > 0.35:
                name = names[np.argmax(proba)]
            else:
                name = ''
            # print(name)
            # print(proba)

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 5), (right, bottom + 20), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 2, bottom + 10), font, 0.5, (255, 255, 255), 1)

        # If using webcam display frame
        if args['display']:
            # Display the resulting frame
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # if the video writer is None *AND* we are supposed to write
        # the output video to disk initialize the writer
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'MPEG')
            writer = cv2.VideoWriter(path_output, fourcc, 24,
                                     (frame.shape[1], frame.shape[0]), True)
        # if the writer is not None, write the frame with recognized
        # faces to disk
        if writer is not None:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

video_capture.release()
cv2.destroyAllWindows()

# check to see if the video writer point needs to be released
if writer is not None:
    writer.release()
    print("[INFO] Video processed successfully.")
