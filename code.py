import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
# %matplotlib inline
import tensorflow
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.preprocessing.image import ImageDataGenerator
import os
os.listdir("../input/")
a=list(os.listdir("../input/train/"))
a.sort()
a
def get_images(directory):
    Images = []
    Labels = []
    for dir_name in os.listdir(directory): 
        for image_file in os.listdir(directory+dir_name):
            image = cv2.imread(directory+dir_name+r'/'+image_file)
            if image is not None:
                image = cv2.resize(image,(300,300),)
                Images.append(image)
                Labels.append(dir_name)
    return Images, Labels
Images, Labels = get_images('../input/train/')
labels = []
mapping = { 'Sugar beet': 0, 'Fat Hen': 1, 'Scentless Mayweed' : 2, 'Charlock' : 3,
           'Small-flowered Cranesbill': 4, 'Maize': 5, 'Shepherds Purse' :6, 'Common wheat': 7,
           'Common Chickweed': 8, 'Cleavers': 9, 'Loose Silky-bent' : 10, 'Black-grass': 11 }
for label in Labels:
    labels.append(mapping[label])
del Labels
Images[0].shape
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Dropout
train_datagen=ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_directory("../input/train/",batch_size=20,target_size=(300,300))
model=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(4,(5,5),activation='relu',input_shape=(300,300,3)),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(8,(5,5),activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(16,(5,5), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(12,activation='softmax')
])

model.summary()
from tensorflow.keras.optimizers import RMSprop

model.compile(loss='categorical_crossentropy',optimizer=RMSprop(lr=0.001),metrics=['accuracy'])
history=model.fit_generator(train_generator,steps_per_epoch=100,epochs=25)

model.save('model.h5')
