import os
import cv2 as cv
import numpy as np


people = ['Ben Afflek', 'Elton John','Girish Naik', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
#for i in os.listdir(r"C:\Users\Girish Naik\Desktop\OpenCv\opencv-course-master\Resources\Faces\train"):
#   people.append(i)

DIR = r"C:\Users\Girish Naik\Desktop\OpenCv\opencv-course-master\Resources\Faces\train"
haar_cascade = cv.CascadeClassifier(r"C:\Users\Girish Naik\Desktop\OpenCv\haar_cascade.xml")

features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            face_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            for (x,y,w,h) in face_rect:
                face_roi = gray[y:y+h, x:x+w]
                features.append(face_roi)
                labels.append(label)

create_train()

face_recognizer = cv.face.LBPHFaceRecognizer_create()
features = np.array(features, dtype=object)
labels = np.array(labels)

face_recognizer.train(features, labels)

face_recognizer.save(r"C:\Users\Girish Naik\Desktop\OpenCv\Face_recogizer.yml")
np.save(r"C:\Users\Girish Naik\Desktop\OpenCv\features.npy", features)
np.save(r"C:\Users\Girish Naik\Desktop\OpenCv\labels.npy", labels)

print("Training Done-------")