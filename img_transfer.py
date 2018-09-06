from data.model import create_model
from img_dataloader import triplet_generator
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Layer
from keras.layers.core import Dropout, Dense
from keras.optimizers import SGD
from keras.layers import Subtract, Concatenate
from keras.callbacks import ModelCheckpoint
import os

WEIGHTS_PATH = './weights/transfer.final.hdf5'
WEIGHTS_CALLBACK = os.path.join('./weights', 'transfer.{epoch:02d}.hdf5')

model = create_model()

in_a = Input(shape=(96, 96, 3))
in_p = Input(shape=(96, 96, 3))
in_n = Input(shape=(96, 96, 3))

emb_a = model(in_a)
emb_p = model(in_p)
emb_n = model(in_n)

first_model = Model([in_a, in_p, in_n], [emb_a, emb_p, emb_n])
first_model.load_weights('./weights/nn4.small2.final.hdf5')

for layer in first_model.layers:
    layer.trainable = False

x0, x1, x2 = first_model.output
x = Subtract()([x1, x2])
x = Concatenate()([x0, x])
x = Dense(128, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(128, activation="relu")(x)
predictions = Dense(25, activation="softmax")(x)

model_final = Model(inputs=first_model.input, outputs=predictions)

model_final.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])

generator = triplet_generator('./transfer_img', False)

save_weights = ModelCheckpoint(WEIGHTS_CALLBACK, verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=10)

model_final.fit_generator(generator, steps_per_epoch=100, epochs=100, callbacks=[save_weights])

model_final.save_weights(WEIGHTS_PATH)
