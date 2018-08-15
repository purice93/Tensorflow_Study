""" 
@author: zoutai
@file: test.py 
@time: 2018/08/14 
@description: 简单神经网络-图像分类
"""

from tensorflow import keras
import matplotlib.pyplot as plt
import os
import learnAndUseML.BasicClassification as BC
import numpy as np


(train_images, train_labels), (test_images, test_labels) = BC.load_data()

# test 1 -BasicClassification
model = keras.models.load_model("BC_model.h5")

predictions = model.predict(test_images)
print(predictions[0])
print(np.argmax(predictions[0]))
print(test_labels[0])

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 画出预测图
plt.figure(figsize=(10,10))
for i in range(0,25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.xticks([])
    plt.grid('off')
    plt.imshow(test_images[i],cmap=plt.cm.binary)
    prediction_label = np.argmax(predictions[i])
    if prediction_label == test_labels[i]:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{} ({})".format(class_names[prediction_label],class_names[test_labels[i]]),color=color)

# 使用训练模型预测一个样本
img = test_images[0]
img = (np.expand_dims(img,0)) # 将样本转化为三维输入
predictions = model.predict(img)
print(np.argmax(predictions[0]))
plt.show()