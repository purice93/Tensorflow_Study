""" 
@author: zoutai
@file: Regression.py 
@time: 2018/08/14 
@description: 
"""
import numpy as np
from tensorflow import keras


def load_data(path='boston_housing.npz', test_split=0.2, seed=113):
    """Loads the Boston Housing dataset.

    Arguments:
        path: path where to cache the dataset locally
            (relative to ~/.keras/datasets).
        test_split: fraction of the data to reserve as test set.
        seed: Random seed for shuffling the data
            before computing the test split.

    Returns:
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    assert 0 <= test_split < 1
    path = 'E:\Python_Workspace\Tensorflow_Study\datasets\\tf-keras-datasets\\boston_housing.npz'
    with np.load(path) as f:
        x = f['x']
        y = f['y']

    np.random.seed(seed)
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]

    x_train = np.array(x[:int(len(x) * (1 - test_split))])
    y_train = np.array(y[:int(len(x) * (1 - test_split))])
    x_test = np.array(x[int(len(x) * (1 - test_split)):])
    y_test = np.array(y[int(len(x) * (1 - test_split)):])
    return (x_train, y_train), (x_test, y_test)


boston_housing = keras.datasets.boston_housing

(train_data, train_labels), (test_data, test_labels) = load_data()

# Shuffle the training set 随机生成训练集
order = np.argsort(np.random.random(len(train_data)))
train_data = train_data[order]
train_labels = train_labels[order]

# (404, 13)404个样本，13个特征
print(train_data.shape)

# 表格处理，数据统计
import pandas as pd

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                'TAX', 'PTRATIO', 'B', 'LSTAT']
df = pd.DataFrame(train_data, columns=column_names)
print(df.head())

# Normalize features 标准化，即表示远离数据中心x个标准差
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

import tensorflow as tf
model = keras.Sequential()
model.add(keras.layers.Dense(64,activation=tf.nn.relu,input_shape=(train_data.shape[1],)))
model.add(keras.layers.Dense(64,activation=tf.nn.relu))
model.add(keras.layers.Dense(1))

model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='mse',
              metrics=['mae']) # 因为可以多选，所以用列表

print(model.summary())

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs):
        if epoch % 100 == 0:
            print('')
        print('.',end='')
EPOCHS = 500
history = model.fit(train_data,
                    train_labels,
                    epochs=EPOCHS,
                    validation_split=0.2,
                    verbose=0,
                    callbacks=[PrintDot()])

# Visualize the model's training progress
import matplotlib.pyplot as plt

def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('error')
    plt.plot(history.epoch,np.array(history.history['mean_absolute_error']),label='Train Loss')
    plt.plot(history.epoch,np.array(history.history['val_mean_absolute_error']),label='Val Loss')
    plt.legend()

plot_history(history)


# 只关注停止之前的20个（为了查看细节）
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
history = model.fit(train_data,
                    train_labels,
                    epochs=EPOCHS,
                    validation_split=0.2,
                    verbose=0,
                    callbacks=[early_stop,PrintDot()])
plot_history(history)

plt.show()

[loss, mae] = model.evaluate(test_data, test_labels, verbose=0)

print("Testing set Mean Abs Error: ${:7.2f}".format(mae * 1000))



# Predict

test_predictions = model.predict(test_data).flatten()

print(test_predictions)