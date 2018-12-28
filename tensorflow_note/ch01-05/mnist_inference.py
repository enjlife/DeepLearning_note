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
layer1_node = 500

def get_weight_varialbe(shape,regularizer):
    weights = tf.get_variable('weights',shape,initializer=tf.truncated_normal_initializer(stddev=0.1))
    #print(weights)
    if regularizer != None:
        #print(regularizer(weights))
        #https://stackoverflow.com/questions/37107223/how-to-add-regularizations-in-tensorflow
        #换了一下储存的集合，原来的集合是自定义的"losses"更换为tf.GraphKeys.REGULARIZATION_LOSSES
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,regularizer(weights)) #给定正则化函数时，将正则化损失加入losses集合--自定义的集合
    return weights

def inference(input_tensor,regularizer):
    with tf.variable_scope('layer1'):
        weights = get_weight_varialbe([input_data_num,layer1_node],regularizer)
        biases = tf.get_variable('biases',[layer1_node],initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights)+biases)

    with tf.variable_scope('layer2'):
        weights = get_weight_varialbe([layer1_node, output_data_num], regularizer)
        biases = tf.get_variable('biases', [output_data_num], initializer=tf.constant_initializer(0.0))
        layer2 = tf.nn.relu(tf.matmul(layer1, weights) + biases)
    return layer2


# regularization_rate = 0.0001
# regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)
#
# #y = inference([1],regularizer)
# weights = tf.get_variable('weights',[2,2],initializer=tf.truncated_normal_initializer(stddev=0.1))
# # weights = tf.Variable(tf.truncated_normal([2,2],stddev=1))
# with tf.Session() as sess:
#     # tf.global_variables_initializer.run()
#     tf.global_variables_initializer().run()
#     print(sess.run(weights))
#     print(sess.run(regularizer(weights)))
#     print(regularizer(weights))

#样例！！！
#2.auto added and read,but using get_variable
# with tf.variable_scope('x',
#         regularizer=tf.contrib.layers.l2_regularizer(0.1)):
#     var1 = tf.get_variable(name='v1',shape=[1],dtype=tf.float32)
#     var2 = tf.get_variable(name='v2',shape=[1],dtype=tf.float32)
# reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
#here reg_losses is a list,should be summed
