""" 
@author: zoutai
@file: SaveAndRestore.py 
@time: 2018/08/15 
@description: Save and restore models
"""

import os

import tensorflow as tf
from tensorflow import keras

import numpy as np
def load_data(path='mnist.npz'):
  origin_folder = 'E:\Python_Workspace\Tensorflow_Study\datasets\\tf-keras-datasets\mnist.npz'
  with np.load(origin_folder) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

    return (x_train, y_train), (x_test, y_test)


(train_data, train_label), (test_data, test_label) = load_data()

print(train_data.shape)

# 改为一维输入
train_data = train_data[:1000].reshape(-1,28*28) / 255.0
test_data = test_data[:1000].reshape(-1,28*28) / 255.0

train_label = train_label[:1000]
test_label = test_label[:1000]

def create_model():
    model = keras.models.Sequential([
        keras.layers.Dense(512,activation=tf.nn.relu,input_shape=(784,)),
        keras.layers.Dropout(0.01),
        keras.layers.Dense(10,tf.nn.softmax)
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.sparse_categorical_crossentropy,
        # loss=keras.losses.categorical_crossentropy,
        # Error when checking target: expected dense_1 to have shape (10,) but got array with shape (1,)
        metrics=['accuracy']
    )
    return model

model = create_model()
print(model.summary())

# Save checkpoints during training
checkpoint_path = "E:\Python_Workspace\Tensorflow_Study\\training_1\cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    verbose=1,
    save_weights_only=True,
)
model.fit(train_data,
          train_label,
          validation_data=(test_data,test_label),
          epochs=10,
          verbose=1,
          callbacks=[cp_callback],
)

model = create_model()
loss , metrics = model.evaluate(test_data,test_label)
print("the before training model loss is {},metrics is {:0.2f}%".format(loss,metrics*100))

model.load_weights(checkpoint_path)
loss , metrics = model.evaluate(test_data,test_label)
print("the after training model loss is {},metrics is {:0.2f}%".format(loss,metrics*100))


# Checkpoint callback options
checkpoint_path = "E:\Python_Workspace\Tensorflow_Study\\training_1\cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    verbose=1,
    save_weights_only=True,
    # Save weights, every 5-epochs.
    period=5
)
model.fit(train_data,
          train_label,
          validation_data=(test_data,test_label),
          epochs=50,
          verbose=1,
          callbacks=[cp_callback],
)
loss , metrics = model.evaluate(test_data,test_label)
print("the Restored model loss is {},metrics is {:0.2f}%".format(loss,metrics*100))


# 未看
import pathlib

# Sort the checkpoints by modification time.
checkpoints = pathlib.Path(checkpoint_dir).glob("*.index")
checkpoints = sorted(checkpoints, key=lambda cp:cp.stat().st_mtime)
checkpoints = [cp.with_suffix('') for cp in checkpoints]
latest = str(checkpoints[-1])
# the default tensorflow format only saves the 5 most recent checkpoints.
print(checkpoints) #

model = create_model()
model.load_weights(latest)


# Manually save weights
# Save the weights
model.save_weights('./checkpoints/my_checkpoint')

# Restore the weights
model = create_model()
model.load_weights('./checkpoints/my_checkpoint')

loss,acc = model.evaluate(test_data, test_label)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))


# Save the entire model
# The entire model can be saved to a file that contains the weight values, the model's configuration, and even the optimizer's configuration.
# This allows you to checkpoint a model and resume training later—from the exact same state—without access to the original code.
# Saving a fully-functional model in Keras is very useful—you can load them in TensorFlow.js and then train and run them in web browsers.

model = create_model()
model.fit(train_data, train_label, epochs=5)
model.save('my_model.h5')

new_model = keras.models.load_model('my_model.h5')
print(new_model.summary())
loss, acc = new_model.evaluate(test_data, test_label)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))