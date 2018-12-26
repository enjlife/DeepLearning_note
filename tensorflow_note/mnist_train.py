import tensorflow as tf
import sys, os
sys.path.append(os.pardir)
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.python.platform import gfile
from tensorflow.examples.tutorials.mnist import input_data

from tensorflow_note import mnist_inference #引用需要先编译通过

model_path = os.path.dirname(__file__)
data_set = model_path+'/MNIST_data'
save_file = model_path + '/model/'

batch_size = 100
learning_rate_base = 0.9
learning_rate_decay = 0.96
regularization_rate = 0.0001
train_step_num = 20000
moving_average_decay = 0.99
model_save_path = save_file
model_name = 'model1.ckpt'

def train(mnist):
    x = tf.placeholder(dtype=tf.float32,shape=[None,mnist_inference.input_data_num],name='x-input')
    regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)
    y = mnist_inference.inference(x,regularizer=regularizer)

    global_step = tf.Variable(0, trainable=False)
    variable_average = tf.train.ExponentialMovingAverage(moving_average_decay,global_step)
    variable_average_op = variable_average.apply(tf.trainable_variables())  #TypeError: 'function' object is not iterable

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))  #y为什么没有使用argmax取index？？
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    #print(tf.get_collection('losses'))
    loss = cross_entropy_mean + tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    #loss = cross_entropy_mean

    learning_rate = tf.train.exponential_decay(learning_rate_base,global_step,mnist.train.num_examples/batch_size,learning_rate_decay)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    with tf.control_dependencies([train_step,variable_average_op]):
        train_op = tf.no_op('train')

    saver = tf.train.Saver()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()  #initializer()需要加括号
        for i in range(train_step_num):
            xs,ys = mnist.train.next_batch(batch_size)
            _,loss_value,step = sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys})

            if i % 1000 == 0:
                print('%d iter：%g' %(step,loss_value))
                #saver.save(sess,os.path.join(model_path,model_name),global_step=global_step)

def main(argv=None):
    mnist = input_data.read_data_sets(data_set,one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()



