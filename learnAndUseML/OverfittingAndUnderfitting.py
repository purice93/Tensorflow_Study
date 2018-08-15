""" 
@author: zoutai
@file: OverfittingAndUnderfitting.py 
@time: 2018/08/15 
@description: Explore overfitting and underfitting 过拟合和欠拟合
"""

import json

import numpy as np
from tensorflow import keras
from tensorflow.python.keras.preprocessing.sequence import _remove_long_seq


def load_data(path='imdb.npz',
              num_words=None,
              skip_top=0,
              maxlen=None,
              seed=113,
              start_char=1,
              oov_char=2,
              index_from=3,
              **kwargs):
    f = np.load("E:\Python_Workspace\Tensorflow_Study\datasets\\tf-keras-datasets\imdb.npz")
    x_train, labels_train = f['x_train'], f['y_train']
    x_test, labels_test = f['x_test'], f['y_test']
    np.random.seed(seed)
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    x_train = x_train[indices]
    labels_train = labels_train[indices]

    indices = np.arange(len(x_test))
    np.random.shuffle(indices)
    x_test = x_test[indices]
    labels_test = labels_test[indices]

    xs = np.concatenate([x_train, x_test])
    labels = np.concatenate([labels_train, labels_test])

    if start_char is not None:
        xs = [[start_char] + [w + index_from for w in x] for x in xs]
    elif index_from:
        xs = [[w + index_from for w in x] for x in xs]

    if maxlen:
        xs, labels = _remove_long_seq(maxlen, xs, labels)
        if not xs:
            raise ValueError('After filtering for sequences shorter than maxlen=' +
                             str(maxlen) + ', no sequence was kept. '
                                           'Increase maxlen.')
    if not num_words:
        num_words = max([max(x) for x in xs])

    # by convention, use 2 as OOV word
    # reserve 'index_from' (=3 by default) characters:
    # 0 (padding), 1 (start), 2 (OOV)
    if oov_char is not None:
        xs = [
            [w if (skip_top <= w < num_words) else oov_char for w in x] for x in xs
        ]
    else:
        xs = [[w for w in x if skip_top <= w < num_words] for x in xs]

    idx = len(x_train)
    x_train, y_train = np.array(xs[:idx]), np.array(labels[:idx])
    x_test, y_test = np.array(xs[idx:]), np.array(labels[idx:])

    return (x_train, y_train), (x_test, y_test)


(train_data, train_labels), (test_data, test_labels) = load_data(num_words=1000)


# Multi-hot-encoding 将评论句子转化为1000维的向量
def multi_hot_sequences(sequences, dimension):
    result = np.zeros((len(sequences), dimension))
    for i,word_index in enumerate(sequences):
        result[i,word_index] = 1.0
    return result

NUM_WORDS=1000
train_data = multi_hot_sequences(train_data,NUM_WORDS)
test_data = multi_hot_sequences(test_data,NUM_WORDS)
print("train length is {},test length is {}".format(len(train_data),len(test_data)))


import matplotlib.pyplot as plt
plt.plot(train_data[0])
plt.show()

# Demonstrate overfitting

# 1.Create a baseline model
import tensorflow as tf
baseline_model = keras.Sequential([
    keras.layers.Dense(16,input_shape=(NUM_WORDS,)),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

baseline_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy','binary_crossentropy']
)
print(baseline_model.summary())

baseline_history = baseline_model.fit(train_data,
                                      train_labels,
                                      batch_size=500, # 每次送入500个样本，进行一次梯度更新
                                      epochs=20, # 所有的样本，重复送5次样本
                                      validation_data=(test_data,test_labels)
                                      )

# 2.Create a smaller model
smaller_model = keras.Sequential([
    keras.layers.Dense(4,input_shape=(NUM_WORDS,)),
    keras.layers.Dense(4, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

smaller_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy','binary_crossentropy']
)
print(smaller_model.summary())

smaller_history = smaller_model.fit(train_data,
                                      train_labels,
                                      batch_size=500,
                                      epochs=20,
                                      validation_data=(test_data,test_labels)
                                      )

# 3.Create a bigger model
bigger_model = keras.Sequential([
    keras.layers.Dense(256,input_shape=(NUM_WORDS,)),
    keras.layers.Dense(256, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

bigger_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy','binary_crossentropy']
)
print(bigger_model.summary())

bigger_history = bigger_model.fit(train_data,
                                      train_labels,
                                      batch_size=500,
                                      epochs=20,
                                      validation_data=(test_data,test_labels)
                                      )
# Plot the training and validation loss
def plot_histories(histories, key='binary_crossentropy'):
    plt.figure(figsize=(10,18))
    for name, history in histories:
        val = plt.plot(history.epoch,history.history['val_'+key],'--',label=name.title()+' Val')
        plt.plot(history.epoch,history.history[key],color=val[0].get_color(),label=name.title()+' Train')
    plt.xlabel('Epochs')
    plt.ylabel(key)
    plt.legend()
    plt.xlim([0,max(history.epoch)])

plot_histories([
    ('baseline', baseline_history),
    ('smaller', smaller_history),
    ('bigger', bigger_history)
])


# Strategies:add weight regularization
l2_model = keras.Sequential([
    keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),input_shape=(NUM_WORDS,)),
    keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

l2_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy','binary_crossentropy']
)
print(l2_model.summary())

l2_history = l2_model.fit(train_data,
                          train_labels,
                          batch_size=500,
                          epochs=20,
                          validation_data=(test_data,test_labels)
                          )
plot_histories([('baseline',baseline_history),
                ('l2',l2_history)])


# Add dropout
dpt_model = keras.Sequential([
    keras.layers.Dense(16, input_shape=(NUM_WORDS,)),
    keras.layers.Dropout(0.01),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dropout(0.01),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

dpt_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy','binary_crossentropy']
)
print(dpt_model.summary())

dpt_history = dpt_model.fit(train_data,
                          train_labels,
                          batch_size=500,
                          epochs=20,
                          validation_data=(test_data,test_labels)
                          )

plot_histories([('baseline', baseline_history),
              ('dropout', dpt_history)])

plt.show()