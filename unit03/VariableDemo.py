""" 
@author: zoutai
@file: VariableDemo.py 
@time: 2018/08/20 
@description: 
"""

import tensorflow as tf

# Create two variables.
weights = tf.Variable(tf.random_normal([784,200],stddev=0.35),name="weights")
biases = tf.Variable(tf.zeros([200]),name='biases')

# Add an op to initialize the variables.
init_op = tf.initialize_all_variables()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# 由另一个变量初始化
w2 = tf.Variable(weights,name='w2')
w3 = tf.Variable(w2*0.2,name='w3')

# Later, when launching the model
with tf.Session() as sess:

    # Run the init operation.
    sess.run(init_op)

    # Save the variables to disk.
    # ckpt - Checkpoint
    save_path = saver.save(sess,'/model.ckpt')

    print("Model saved in file: ", save_path)

    # Restore variables from disk.
    saver.restore(sess,'/model.ckpt')
    print("Model restored.")

    # Use the model

    # 存储部分变量
    saver.save({'my_w2':w2})



