"""
@author: AyoubGueddou

Usage:
  extract_faces.py -t <input_type> -i <input_path> -d <detection_method>
Options:
  -h, --help                          Show this help
  -t, --input_type                    Input type (either 'video' or 'img_folder')
  -i, --input_path                    Path to input video or image folder
  -d --detection_method               Face detection model to use (either `hog` or 'haar')
"""

import cv2
import os
import dlib
import argparse

# Haar-cascade face detection
face_cascade = cv2.CascadeClassifier("./models/haarcascade_frontalface_default.xml")
# HoG Features face detection
face_detector = dlib.get_frontal_face_detector()


def input_to_name(argument):
    switcher = {
        0: "Unknown",
        1: "Tony",
        2: "Steve",
        3: "Bruce",
        4: "Thor",
        5: "Natasha",
        6: "Clint",
        7: "Fury"
    }
    try:
        tmp = int(argument)
        return switcher.get(tmp, "Invalid name")
    except:
        pass

    return argument


def detect_face(frame_, method='hog'):

    # Convert to grayscale for face detection
    gray_ = cv2.cvtColor(frame_, cv2.COLOR_BGR2GRAY)
    # Resize to half the size for faster processing
    gray_ = cv2.resize(gray_, (0, 0), fx=0.5, fy=0.5)

    # Face detection using chosen method
    if method == 'haar':
        faces = face_cascade.detectMultiScale(gray_, 1.1, 3)
    else:
        faces = face_detector(gray_, 1)

    # Loop through each face and draw a rect around it
    for face in faces:

        # Convert bounding box to original size
        if method == 'haar':
            (top, left, width, height) = face
            top *= 2
            left *= 2
            bottom = top + (height * 2)
            right = left + (width * 2)

            # Face location
            roi_face = frame_[left:right, top:bottom]
        else:
            top = face.top() * 2
            right = face.right() * 2
            bottom = face.bottom() * 2
            left = face.left() * 2

            # Face location
            roi_face = frame_[top:bottom, left:right]

        # Display the detected face
        if roi_face.any():
            cv2.imshow('face', roi_face)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # Choose which person this face belongs to.
            name = input('Press 1: "Tony",  2: "Steve",  3: "Bruce",  4: "Thor",  5: "Natasha",  '
                         '6: "Clint",  7: "Fury" \nOr input the person\'s name (press Q to quit):\n')

            # Press Q to quit
            if name.lower() == "q":
                frame_ = -1
                return frame_

            name = input_to_name(name)
            if name == '':
                name = 'Unknown'
                print('Unknown face')
            else:
                print('It is ' + str(name) + "'s face !")

            # Create folder named after this person if it doesn't exist
            if not os.path.exists(os.path.join(face_folder, name)):
                os.makedirs(os.path.join(face_folder, name))

            # Save cropped face
            number_files = len(os.listdir(os.path.join(face_folder, name)))  # dir is your directory path
            # print(os.path.join(face_folder, name) + '/' + str(number_files + 1) + '.jpg')
            cv2.imwrite(os.path.join(face_folder, name) + '/' + str(number_files + 1) + '.jpg', roi_face)

    return frame_


if __name__ == '__main__':

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--input_type", required=True, type=str, default="img_folder",
                    help="video or folder containing multiple images, either `video` or 'img_folder'")
    ap.add_argument("-i", "--input_path", required=True, type=str,
                    help="path to input video or image")
    ap.add_argument("-d", "--detection_method", type=str, default="hog",
                    help="face detection model to use: either `hog` or 'haar'")
    args = vars(ap.parse_args())

    nb_frame = 0

    # face folder path
    face_folder = './face_folder/'
    if not os.path.exists(face_folder):
        os.makedirs(face_folder)

    if args['input_type'] == 'video':

        # Read video
        video_capture = cv2.VideoCapture(args['input_path'])

        while True:
            # Count frame
            nb_frame += 1
            # if nb_frame < 1000:
            #     continue

            ret, frame = video_capture.read()
            if ret:
                # Read every 15 frames to avoid detecting the same face too much time
                if nb_frame % 15 == 0:
                    # Detect and store faces using chosen method
                    canvas = detect_face(frame, method=args['detection_method'])

                    # Monitor if user wants to quit the program
                    if type(canvas) == int and canvas == -1:
                        print("Program exited")
                        break
            else:
                break

        video_capture.release()
        cv2.destroyAllWindows()
    else:

        onlyfiles = [f for f in os.listdir(args['input_path']) if os.path.isfile(os.path.join(args['input_path'], f))]
        print(len(onlyfiles), 'images to process. \n')

        for file in onlyfiles:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                frame = cv2.imread(os.path.join(args['input_path'], file))  # reads image 'opencv-logo.png' as grayscale
                canvas = detect_face(frame, method=args['detection_method'])

                # Monitor if user wants to quit the program
                if type(canvas) == int and canvas == -1:
                    print("Program exited")
                    break

                # cv2.imshow('processed image', canvas)
                # if cv2.waitKey(0) & 0xFF == ord('q'):
                #     break
                # cv2.destroyAllWindows()
