import cv2
import sys
from collections import deque
from data.model import create_model
from img_dataloader import TripletPrediction
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dropout, Dense
from keras.layers import Subtract, Concatenate
import numpy as np
import os

WEIGHTS_NN4SMALL2_PATH = './weights/nn4.small2.final.hdf5'
WEIGHTS_TRANSFER_PATH = './weights/transfer.final.hdf5'

model = create_model()

in_a = Input(shape=(96, 96, 3))
in_p = Input(shape=(96, 96, 3))
in_n = Input(shape=(96, 96, 3))

emb_a = model(in_a)
emb_p = model(in_p)
emb_n = model(in_n)

first_model = Model([in_a, in_p, in_n], [emb_a, emb_p, emb_n])
first_model.load_weights(WEIGHTS_NN4SMALL2_PATH)

for layer in first_model.layers:
    layer.trainable = False

tmp0 = os.listdir('./transfer_img')

x0, x1, x2 = first_model.output
x = Subtract()([x1, x2])
x = Concatenate()([x0, x])
x = Dense(128, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(128, activation="relu")(x)
predictions = Dense(len(tmp0), activation="softmax")(x)

model_final = Model(inputs=first_model.input, outputs=predictions)

model_final.load_weights(WEIGHTS_TRANSFER_PATH)
model_final.compile(loss="categorical_crossentropy", optimizer='adam')

IMG_SIZE = (96, 96)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)

frames = deque(maxlen=2)



while True:
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        frames.append(cv2.resize(frame[y:y+h, x:x+w], IMG_SIZE))

    if len(frames) >= 2:
        triplet = TripletPrediction()
        generator = triplet.triplet_webcam('./transfer_img', frames)
        predictions = model_final.predict_generator(generator=generator, verbose=1, steps=1)
        predictions = np.argmax(predictions, axis=1)
        print(predictions[0])
        cv2.imshow("Video", frames[0])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
