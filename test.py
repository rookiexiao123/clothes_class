import numpy as np
import imutils
import os
import pickle
import cv2
from keras.preprocessing.image import img_to_array
from keras.models import load_model

path = '1.jpg'
image = cv2.imread(path)
image = cv2.resize(image, (96, 96))

image = image.astype('float')/255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

model = load_model('model_fashion')
proba = model.predict(image)[0]
idxs = np.argsort(proba)[::-1][:2]

print(idxs)

mlb = pickle.loads(open("labelbin","rb").read())
for (i, j) in enumerate(idxs):
    label = '{}: {:.2f}%'.format(mlb.classes_[j], proba[j]*100)
    print(label)