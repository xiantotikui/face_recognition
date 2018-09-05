import os
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

INPUT_DIR = './in_img'
OUTPUT_DIR = './out_img'
IMG_SIZE = (96, 96)

for filename0 in os.listdir(INPUT_DIR):
    size = len(os.listdir(os.path.join(INPUT_DIR, filename0)))
    if size == 0 or size == 1:
        continue
    os.makedirs(os.path.join(OUTPUT_DIR, filename0.lower()))
    for filename1 in os.listdir(os.path.join(INPUT_DIR, filename0)):
        print(os.path.join(INPUT_DIR, filename0, filename1))
        img = cv2.imread(os.path.join(INPUT_DIR, filename0, filename1))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            roi_color = img[y:y+h, x:x+w]

        img = cv2.resize(roi_color, IMG_SIZE)

        cv2.imwrite(os.path.join(OUTPUT_DIR, filename0.lower(), filename1.lower()), img)
