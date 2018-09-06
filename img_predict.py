from data.model import create_model
from img_dataloader import triplet_prediction
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

tmp0 = (os.listdir('./transfer_img'))

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

generator = triplet_prediction('./transfer_img')
prediction = model_final.predict_generator(generator=generator, verbose=1, steps=1)

tmp1 = np.argmax(prediction, axis=1)
print([tmp0[item] for item in tmp1])
