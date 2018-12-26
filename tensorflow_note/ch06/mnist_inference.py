import tensorflow as tf
import sys, os
sys.path.append(os.pardir)
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.python.platform import gfile
from tensorflow.examples.tutorials.mnist import input_data
from dataset import mnist

input_data_num = 784
output_data_num = 10
image_size = 28
num_channel = 1
num_labels = 10

conv1_deep = 32
conv1_size = 5
conv2_deep = 64
conv2_size = 5
fc_size = 512

# def get_weight_varialbe(shape,regularizer):
#     weights = tf.get_variable('weights',shape,initializer=tf.truncated_normal_initializer(stddev=0.1))
#     #print(weights)
#     if regularizer != None:
#         #print(regularizer(weights))
#         #https://stackoverflow.com/questions/37107223/how-to-add-regularizations-in-tensorflow
#         #换了一下储存的集合，原来的集合是自定义的"losses"更换为tf.GraphKeys.REGULARIZATION_LOSSES
#         tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,regularizer(weights)) #给定正则化函数时，将正则化损失加入losses集合--自定义的集合
#     return weights

def inference(input_tensor,train,regularizer):
    with tf.variable_scope('layer1-conv'):
        conv1_weights = tf.get_variable\
            ('weights',[conv1_size,conv1_size,num_channel,conv1_deep],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable('biases',[conv1_deep],initializer=tf.truncated_normal_initializer(stddev=0.0))
        conv1 = tf.nn.conv2d(input_tensor,conv1_weights,strides=[1,1,1,1],padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))

    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    with tf.variable_scope('layer3-conv2'):
        conv2_weights = tf.get_variable('weights',[conv2_size,conv2_size,
                                                   conv1_deep,conv2_deep],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable('biases',[conv2_deep],initializer=tf.truncated_normal_initializer(stddev=0.0))
        conv2 = tf.nn.conv2d(pool1,conv2_weights,strides=[1,1,1,1],padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))

    with tf.name_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,1,1,1],padding='SAME')

    pool_shape = pool2.get_shape().as_list()  #pool_shape[0]为一个batch的数量
    nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
    reshaped = tf.reshape(pool2,[pool_shape[0],nodes])

    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable('weights',[nodes,fc_size],initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc1_biases = tf.get_variable('biases',[fc_size],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc1_weights))
        fc1 = tf.nn.relu(tf.matmul(reshaped,fc1_weights)+fc1_biases)
        if train: tf.nn.dropout(fc1,0.5)

    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable('weights',[fc_size,output_data_num],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer !=None:
            tf.add_to_collection('losses',regularizer(fc2_weights))
        fc2_biases = tf.get_variable('biases',[output_data_num],initializer=tf.truncated_normal_initializer(stddev=0.1))
        logits = tf.matmul(fc1,fc2_weights)+fc2_biases

    return logits











