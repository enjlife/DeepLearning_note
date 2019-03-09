import tensorflow as tf
import sys, os
sys.path.append(os.pardir)
import numpy as np
import codecs
import collections
from operator import itemgetter
from ch05 import mnist_inference  # 发现文件夹名字带 - 的无法引用
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
dir_name = os.path.dirname(__file__)

# input1 = tf.constant([1.0,2.0,3.0],name='input1')
# input2 = tf.Variable(tf.random_uniform([3]),name='input2')
# output = tf.add_n([input1,input2],name='add')
#
# writer = tf.summary.FileWriter(dir_name,tf.get_default_graph())
# writer.close()

# with tf.variable_scope('foo'):
#     a = tf.get_variable('bar',[1])
#     print(a.name)
#
# with tf.variable_scope('bar'):
#     b = tf.get_variable('bar',[1])
#     print(b.name)
# with tf.name_scope('a'):
#     a = tf.Variable([1])
#     print(a.name)
#     a = tf.get_variable('b',[1])  # get_variable不受name_scope的影响，所以该变量不在这个空间
#     print(a.name)
#
# with tf.name_scope('b'):
#     tf.get_variable('b',[1])  # 此时会报错，因为名称为b的变量已经定义过

#改进代码，使可视化效果提高
# with tf.name_scope('input1'):
#     input1 = tf.constant([1.0,2.0,3.0],name='input1')
#
# with tf.name_scope('input2'):
#     input2 = tf.Variable(tf.random_uniform([3]),name='input2')
#
# output = tf.add_n([input1,input2],name='add')
# writer = tf.summary.FileWriter(dir_name,tf.get_default_graph())
# writer.close()

# 改造后的mnist_train
from tensorflow.examples.tutorials.mnist import input_data
batch_size = 50
input_node = 784
output_node = 10
layer1_node = 500
regularization_rate = 0.0001
moving_average_decay = 0.99
learning_rate_base = 0.95
learning_rate_decay = 0.96
train_step_num = 10000

def train(mnist):
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32,[None,mnist_inference.input_data_num],name='x-input')
        y_ = tf.placeholder(tf.float32,[None,mnist_inference.output_data_num],name='y-output')

    regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)
    y = mnist_inference.inference(x,regularizer)
    global_step = tf.Variable(0,trainable=False)

    with tf.name_scope('mocing_average'):
        variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay,global_step)
        variable_averages = variable_averages.apply(tf.trainable_variables())

    with tf.name_scope('loss_function'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_,1),logits=y)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        loss = cross_entropy_mean + tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    with tf.name_scope('train_step'):
        learning_rate = tf.train.exponential_decay(learning_rate_base,global_step,mnist.train.num_examples/batch_size,
                                                   learning_rate_decay,staircase=True)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

        with tf.control_dependencies([train_step,variable_averages]):
            train_op = tf.no_op(name='train')

    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(dir_name, tf.get_default_graph())

    with tf.Session() as sess:
        tf.global_variables_initializer().run()  # initializer()需要加括号
        for i in range(train_step_num):
            xs, ys = mnist.train.next_batch(batch_size)
            if i % 1000 == 0:
                # 配置运行时需要记录的信息
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                # 运行时记录运行信息的proto
                run_metadata = tf.RunMetadata()

                _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys},
                                               options=run_options,run_metadata=run_metadata)
                writer.add_run_metadata(run_metadata=run_metadata,tag=('tag%g'%i),global_step=i)
                print('%d iter：%g' % (step, loss_value))

                # saver.save(sess,os.path.join(model_path,model_name),global_step=global_step)
            else:
                _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})



    writer.close()

def main(argv=None):
    mnist = input_data.read_data_sets('/Users/enjlife/learning-deep-learning-from-scratch/MNIST_data',one_hot=True)
    train(mnist)

if __name__=='__main__':
    tf.app.run()




