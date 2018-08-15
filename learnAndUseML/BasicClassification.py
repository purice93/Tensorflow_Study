""" 
@author: zoutai
@file: BasicClassification.py 
@time: 2018/08/13 
@description: 
"""

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import gzip
import os
import matplotlib.pyplot as plt

def load_data():
    dirname = os.path.join('E:\Python_Workspace\Tensorflow_Study\datasets', 'fashion-mnist\\')
    base = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/'
    files = [
        'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
    ]

    paths = []
    for fname in files:
        paths.append(dirname+fname)

    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

    return (x_train, y_train), (x_test, y_test)

print(tf.__version__)

(train_images, train_labels), (test_images, test_labels) = load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(train_images.shape)

# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# Turn the axes grids on or off.
# plt.gca().grid(False) # 去掉栅格

train_images = train_images / 255.0
test_images = test_images / 255.0

# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid('off')
#     plt.imshow(train_images[i], cmap=plt.cm.binary) # 默认使用二值图像
#     plt.xlabel(class_names[train_labels[i]])

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)), # 二维转换为一维，即压平
    # 接下来为两个全连接层，即稠密层，分别有128、10个节点
    keras.layers.Dense(128,activation=tf.nn.relu), # max（0,x）
    keras.layers.Dense(10, activation=tf.nn.softmax) # 10个输出按比例分配
])

model.compile(optimizer=tf.train.AdamOptimizer(),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(train_images,train_labels,epochs=5) # 训练5次

test_loss ,test_metrics = model.evaluate(test_images,test_labels)


print(test_loss,test_metrics)

# 保存模型
# 注：模型会保存当前运行的所有代码，包括其中的画图，所以我将画图部分注释掉了
model.save("BC_model.h5")





