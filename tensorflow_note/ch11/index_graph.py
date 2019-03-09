import tensorflow as tf
import sys, os
sys.path.append(os.pardir)
import numpy as np
import codecs
import collections
from operator import itemgetter
from ch05 import mnist_inference  # 发现文件夹名字带 - 的无法引用
from tensorflow.examples.tutorials.mnist import input_data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
dir_name = os.path.dirname(__file__)
summary_dir = dir_name
batch_size = 100
train_steps = 3000

def variable_summaries(var,name):
    with tf.name_scope('summaries'):
        tf.summary.histogram(name,var)  # 给出的图表名称(一个空间一个图)和张量

        mean = tf.reduce_mean(var)  #计算变量的平均值
        tf.summary.scalar('mean/' + name,mean)  # 定义生成平均值信息日志，mean/name
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('seddev/' + name,stddev)

def nn_layer(input_tensor,input_dim,output_dim,layer_name,act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights1'):
            weights = tf.Variable(tf.truncated_normal([input_dim,output_dim],stddev=0.1))
            variable_summaries(weights,layer_name + '/weights1')

        with tf.name_scope('biases'):
            biases = tf.Variable(tf.constant(0.0,shape=[output_dim]))
            variable_summaries(biases,layer_name + '/biases')
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor,weights) + biases
            # variable_summaries(layer_name + '/pre_activations',preactivate) 此处应该使用tf.summary.histogram
            tf.summary.histogram(layer_name + 'Wx_plus_b',preactivate)

        activations = act(preactivate,name='activation')
        tf.summary.histogram(layer_name + '/activation',activations)
        return activations


def main(argv=None):
    mnist = input_data.read_data_sets('/Users/enjlife/learning-deep-learning-from-scratch/MNIST_data',one_hot=True)
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32,[None,784],name='x-input')
        y_ = tf.placeholder(tf.float32,[None,10],name='y-input')

    with tf.name_scope('input_reshape'):
        input_reshape = tf.reshape(x,[-1,28,28,1])
        tf.summary.image('input',input_reshape,10)

    hidden1 = nn_layer(x,784,500,layer_name='layer1')
    y = nn_layer(hidden1,500,10,layer_name='layer2',act=tf.identity)

    with tf.name_scope('loss'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))
        tf.summary.scalar('loss',cross_entropy)

    with tf.name_scope('train'):  # train_step 没有需要监测的变量
        train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))

        with tf.name_scope('acc'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

        tf.summary.scalar('accuracy/acc',accuracy)

    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(summary_dir,sess.graph)
        tf.global_variables_initializer().run()

        for i in range(train_steps):
            xs,ys = mnist.train.next_batch(batch_size)
            summary,_ = sess.run([merged,train_step],feed_dict={x:xs,y_:ys})
            summary_writer.add_summary(summary,i)

    summary_writer.close()

if __name__=='__main__':
    main()







