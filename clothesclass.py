'''
图像预处理
文件改名，图像旋转和修改大小，图像增强
网络：FashionNet
'''

import cv2
from imutils import paths
import random
from keras.preprocessing.image import img_to_array
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer   #多分类需要
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import pickle
from fashionNet import FashionNet
from keras.callbacks import TensorBoard

EPOCHS = 15
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (96, 96, 3)

imagePaths = sorted(list(paths.list_images('dataset')))
random.seed(42)
random.shuffle(imagePaths)

data = []
labels = []

for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    image = img_to_array(image)
    data.append(image)

    label = imagePath.split(os.path.sep)[-2].split("_")
    labels.append(label)

data = np.array(data, dtype='float')/255.0
labels = np.array(labels)

print(len(data))

mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)
for (i, label) in enumerate(mlb.classes_):
    print('{}. {}'.format(i+1, label))

(trainX,testX,trainY,testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

model = FashionNet.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0], depth=IMAGE_DIMS[2], classes=len(mlb.classes_), finaAct="sigmoid")
opt = Adam(lr=INIT_LR, decay=INIT_LR/EPOCHS)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

model.fit_generator(aug.flow(trainX, trainY, batch_size=BS), validation_data=(testX, testY), steps_per_epoch=len(trainX)//BS, epochs=EPOCHS, verbose=1, callbacks=[TensorBoard(log_dir='mytensorboard')])
model.save('model_fashion')

f = open('labelbin', 'wb')
f.write(pickle.dumps(mlb))
f.close()