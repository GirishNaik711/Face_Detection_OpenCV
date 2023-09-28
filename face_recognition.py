import cv2 as cv
import numpy as np



people =  ['Ben Afflek', 'Elton John','Girish Naik', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
features = np.load("features.npy", allow_pickle=True)
labels = np.load("labels.npy")

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read(r"C:\Users\Girish Naik\Desktop\OpenCv\Face_recogizer.yml")

img = cv.imread(r"C:\Users\Girish Naik\Pictures\Camera Roll\WIN_20230328_14_10_18_Pro.jpg")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

haar_cascade = cv.CascadeClassifier(r"C:\Users\Girish Naik\Desktop\OpenCv\haar_cascade.xml")
face_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4)

for (x,y,w,h) in face_rect:
    face_roi = gray[y:y+h,x:x+w]
    label, confidence = face_recognizer.predict(face_roi)
    cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_SIMPLEX , 1.0, (250,250,250), 2)
    cv.rectangle(img, (x,y),(x+w,y+w), (200,200,200), 2)

print(f"The person is: {people[label]}, and Confidence: {confidence}")

cv.imshow("Person", img)

cv.waitKey(0)