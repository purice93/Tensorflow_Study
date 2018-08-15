""" 
@author: zoutai
@file: TextClassification.py 
@time: 2018/08/14 
@description: Text classification with movie reviews 对电影评论进行文本分类
    神经网络
"""
import json

import numpy as np
import tensorflow as tf
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
    """Loads the IMDB dataset.

    Arguments:
        path: where to cache the data (relative to `~/.keras/dataset`).
        num_words: max number of words to include. Words are ranked
            by how often they occur (in the training set) and only
            the most frequent words are kept
        skip_top: skip the top N most frequently occurring words
            (which may not be informative).
        maxlen: sequences longer than this will be filtered out.
        seed: random seed for sample shuffling.
        start_char: The start of a sequence will be marked with this character.
            Set to 1 because 0 is usually the padding character.
        oov_char: words that were cut out because of the `num_words`
            or `skip_top` limit will be replaced with this character.
        index_from: index actual words with this index and higher.
        **kwargs: Used for backwards compatibility.

    Returns:
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

    Raises:
        ValueError: in case `maxlen` is so low
            that no input sequence could be kept.

    Note that the 'out of vocabulary' character is only used for
    words that were present in the training set but are not included
    because they're not making the `num_words` cut here.
    Words that were not seen in the training set but are in the test set
    have simply been skipped.
    """
    # Legacy support
    # if 'nb_words' in kwargs:
    #   logging.warning('The `nb_words` argument in `load_data` '
    #                   'has been renamed `num_words`.')
    #   num_words = kwargs.pop('nb_words')
    # if kwargs:
    #   raise TypeError('Unrecognized keyword arguments: ' + str(kwargs))
    #
    # origin_folder = 'E:\Python_Workspace\Tensorflow_Study\datasets\\tf-keras-datasets\\'
    # path = origin_folder+'imdb.npz'
    # with np.load(path) as f:
    #   x_train, labels_train = f['x_train'], f['y_train']
    #   x_test, labels_test = f['x_test'], f['y_test']

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


# imdb = keras.datasets.imdb
# imdb.load_data(num_words=10000)
(train_data, train_labels), (test_data, test_labels) = load_data(num_words=10000)
# f = np.load("E:\Python_Workspace\Tensorflow_Study\datasets\\tf-keras-datasets\imdb.npz")
# train_data, train_labels, test_data, test_labels = f['x_train'], f['y_train'],f['x_test'], f['y_test']
print(len(train_data))
print(train_data[0])

# A dictionary mapping words to an integer index
path = "E:\Python_Workspace\Tensorflow_Study\datasets\\tf-keras-datasets\imdb_word_index.json"
f = open(path)
word_index = json.load(f)
# word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


print(decode_review(train_data[0]))

# 设置最大长度为256，多的截断，少的填补；
# value-填补数值
# padding，pre or post 填到之前还是之后
train_data = keras.preprocessing.sequence.pad_sequences(
    train_data,
    value=word_index["<PAD>"],
    padding='post',
    maxlen=256
)

test_data = keras.preprocessing.sequence.pad_sequences(
    test_data,
    value=word_index["<PAD>"],
    padding='post',
    maxlen=256
)
print(train_data[0])

# build the model
vocab_size = 10000
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
# Global average pooling operation for temporal data.
# 即把二维变成一维
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
# 因为只有一个输出，所以不用softmax
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
# 输出网络概要
print(model.summary())

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    batch_size=512,
                    epochs=40,
                    verbose=1,
                    validation_data=(x_val, y_val))

result = model.evaluate(test_data, test_labels)
print(result)

# Create a graph of accuracy and loss over time
history_dict = history.history
dict_keys = history_dict.keys()
print("the result dict is: ", dict_keys)

import matplotlib.pyplot as plt

acc=history_dict['acc']
val_acc=history_dict['val_acc']
loss=history_dict['loss']
val_loss=history_dict['val_loss']

epochs = range(1,len(acc)+1)

plt.plot(epochs,loss,'bo',label='training loss')
plt.plot(epochs,val_loss,'b',label='validation loss')
plt.xlabel('epochs')
plt.ylabel('train loss')
plt.title('training and validation loss')
plt.legend()
plt.show()

plt.clf()

plt.plot(epochs,acc,'bo',label='training accuracy')
plt.plot(epochs,val_acc,'b',label='validation accuracy')
plt.xlabel('epochs')
plt.ylabel('train accuracy')
plt.title('training and validation accuracy')
plt.legend()
plt.show()

