from face_recognition.model import create_model
from img_dataloader import triplet_generator
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Layer

WEIGHTS_PATH = './weights/nn4.small2.v1.h5'

model = create_model()

in_a = Input(shape=(96, 96, 3))
in_p = Input(shape=(96, 96, 3))
in_n = Input(shape=(96, 96, 3))

emb_a = model(in_a)
emb_p = model(in_p)
emb_n = model(in_n)

class TripletLossLayer(Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(TripletLossLayer, self).__init__(**kwargs)
    
# -----------------------------------------------------------------------------------------
# 04.09.2018 Code taken from https://github.com/keras-team/keras/issues/9498
# -----------------------------------------------------------------------------------------
    
    def triplet_loss(self, inputs, dist='sqeuclidean', margin='maxplus'):
        anchor, positive, negative = inputs
        positive_distance = K.square(anchor - positive)
        negative_distance = K.square(anchor - negative)
        if dist == 'euclidean':
            positive_distance = K.sqrt(K.sum(positive_distance, axis=-1, keepdims=True))
            negative_distance = K.sqrt(K.sum(negative_distance, axis=-1, keepdims=True))
        elif dist == 'sqeuclidean':
            positive_distance = K.sum(positive_distance, axis=-1, keepdims=True)
            negative_distance = K.sum(negative_distance, axis=-1, keepdims=True)
        loss = positive_distance - negative_distance
        if margin == 'maxplus':
            loss = K.maximum(0.0, 1 + loss)
        elif margin == 'softplus':
            loss = K.log(1 + K.exp(loss))
        return K.mean(loss)
    
    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss
        
triplet_loss_layer = TripletLossLayer(alpha=0.2, name='triplet_loss_layer')([emb_a, emb_p, emb_n])

train = Model([in_a, in_p, in_n], triplet_loss_layer)

generator = triplet_generator()

train.compile(loss=None, optimizer='adam')
train.fit_generator(generator, epochs=10, steps_per_epoch=100) 

train.save_weights(WEIGHTS_PATH)
