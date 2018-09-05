import os
import random
import cv2
import numpy as np
from keras.utils import to_categorical

def triplet_generator(input, logic):
    while True:
        y_size = len(os.listdir(input))
        a_array = []
        p_array = []
        n_array = []
        y_array = []
        for i in range(8):
            a = random.choice(os.listdir(input))
            n = random.choice(os.listdir(input))
            while a == n:
                n = random.choice(os.listdir(input))

            a_path = os.path.join(input, a)
            n_path = os.path.join(input, n)

            a_file = random.choice(os.listdir(a_path))
            p_file = random.choice(os.listdir(a_path))
            while a_file == p_file:
                p_file = random.choice(os.listdir(a_path))

            n_file = random.choice(os.listdir(n_path))

            a_img = cv2.imread(os.path.join(a_path, a_file))
            p_img = cv2.imread(os.path.join(a_path, p_file))
            n_img = cv2.imread(os.path.join(n_path, n_file))

            a_img = np.flip(a_img, 1) / 255
            p_img = np.flip(p_img, 1) / 255
            n_img = np.flip(n_img, 1) / 255

            a_array.append(a_img)
            p_array.append(p_img)
            n_array.append(n_img)

            tmp = to_categorical(np.where(np.asarray(os.listdir(input)) == a), y_size)
            y_array.append(np.reshape(tmp, (y_size)))

        a = np.asarray(a_array)
        p = np.asarray(p_array)
        n = np.asarray(n_array)
        y = np.asarray(y_array)
        if logic:
            yield([a, p, n], None)
        else:
            yield([a, p, n], y)
