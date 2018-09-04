import os
import random
import cv2
import numpy as np

def triplet_generator():
    while True:
        a_array = []
        p_array = []
        n_array = []
        for i in range(4):
            a = random.choice(os.listdir('./out_img'))
            n = random.choice(os.listdir('./out_img'))
            while a == n:
                n = random.choice(os.listdir('./out_img'))
                
            a_path = os.path.join('./out_img', a)
            n_path = os.path.join('./out_img', n)
            
            a_file = random.choice(os.listdir(a_path))
            p_file = random.choice(os.listdir(a_path))
            while a_file == p_file:
                p_file = random.choice(os.listdir(a_path))
                
            n_file = random.choice(os.listdir(n_path))
            
            a_img = cv2.imread(os.path.join(a_path, a_file))
            p_img = cv2.imread(os.path.join(a_path, p_file))
            n_img = cv2.imread(os.path.join(n_path, n_file))
            
            a_img = np.flip(a_img, 1)
            p_img = np.flip(p_img, 1)
            n_img = np.flip(n_img, 1)

            a_array.append(a_img)
            p_array.append(p_img)
            n_array.append(n_img)
                
        a = np.asarray(a_array)
        p = np.asarray(p_array)
        n = np.asarray(n_array)
        
        yield([a, p, n], None)

