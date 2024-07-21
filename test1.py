import os
import sys
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from keras.models import load_model
import tensorflow as tf
graph = tf.get_default_graph()
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform


def loading_model():
   # model=load_model('model1.h5')
    with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        model = load_model('model1.h5')
    print('model loaded')
    return model

def predcit(url,model):
    response=requests.get(url)
    img=Image.open(BytesIO(response.content))
    img=np.array(img.resize((300,300)))
    a=img.reshape(1,300,300,3)
    print(a)
    with graph.as_default():
        b=model.predict(a)
    label=np.argmax(b)
    label_name = {  0:'Black-grass', 1:'Charlock',2: 'Cleavers' ,3: 'Common Chickweed',
           4:'Common Wheat', 5:'Fat Hen', 6:'Loose Silky-bent' , 7:'Maize',
           8:'Scentless Mayweed',9: 'Shepherds Purse',10: 'Small-flowered Cranesbill',11: 'Sugar Beet' }
    return label_name[label]
        

