# ImageClassifier


import numpy as np
import matplotlib.pyplot as plt 
import os
import cv2

datadir = "New folder"
Categories  = ["Acne", "Psoriasis"]

for cat in Categories:
    path = os.path.join(datadir, cat)
    for img in os.listdir(path):
        img_arr=cv2.imread(os.path.join(path,img))
        plt.imshow(img_arr)
        plt.show()
        break
    break
    
    
IMG_SIZE = 255
new_array = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array)
plt.show()


training_data=[]

def create_training_data():
    for cat in Categories:
        path = os.path.join(datadir, cat)
        class_num = Categories.index(cat)
        for img in os.listdir(path):
            try:
                img_arr=cv2.imread(os.path.join(path,img))
                new_array = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                print(e)

create_training_data()



import random

random.shuffle(training_data)



for sample in training_data[:20]:
    print(sample[1])
    
X=[]  #feature set
y=[]  #lables


for features, lables in training_data:
    X.append(features)
    y.append(lables)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)


import pickle

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()



import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle

X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

X = X/255.0

model = Sequential()
model.add(Conv2D(64,  (3,3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size =(2,2)))


model.add(Flatten())
model.add(Dense(64))

model.add(Dense(64))
model.add(Activation("sigmoid"))

model.compile(loss ="binary_crossentropy",
                optimizer="adam",
                metrics=["accuracy"])

model.fit(X, y, batch_size = 32, epochs = 10, validation_split = 0.1)
