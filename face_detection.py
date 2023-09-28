import cv2 as cv

img = cv.imread(r"C:\Users\Girish Naik\Desktop\OpenCv\opencv-course-master\Resources\Photos\lady.jpg")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

haar_cascade = cv.CascadeClassifier(r"C:\Users\Girish Naik\Desktop\OpenCv\haar_cascade.xml")

face_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
print(face_rect)
print(len(face_rect)) 

for (x,y,w,h) in face_rect:
    cv.rectangle(img,(x,y), (x+w, y+h), (0,255,0), 2)

cv.imshow("rectangle drawn", img)


cv.waitKey(0) 