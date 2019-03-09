import tensorflow as tf
import sys, os
sys.path.append(os.pardir)
import numpy as np
import codecs
import collections
from operator import itemgetter
from ch05 import mnist_inference  # 发现文件夹名字带 - 的无法引用
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

batch_size = 50
learning_rate_base = 0.95
learning_rate_decay = 0.96
regularization_rate = 0.0001
train_step_num = 10000
moving_average_decay = 0.99

log_dir = os.path.dirname(__file__)
sprite_file = 'mnist_sprite.jpg'
meta_file = 'mnist_meta.tsv'
tensor_name = 'final_logits'

def train(mnist):
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32,shape=[None,mnist_inference.input_data_num],name='x-input')
        y_ = tf.placeholder(tf.float32,shape=[None,mnist_inference.output_data_num],name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)
    y = mnist_inference.inference(x,regularizer)
    global_step = tf.Variable(0,trainable=False)
    with tf.name_scope('moving_avergae'):
        variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay,global_step)
        variable_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_,1),logits=y)
        loss = tf.reduce_mean(cross_entropy) + tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    with tf.name_scope('train_step'):
        learning_rate = tf.train.exponential_decay(learning_rate_base,global_step,mnist.train.num_examples/batch_size,learning_rate_decay)
        train_steps = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step)

        with tf.control_dependencies([train_steps,variable_averages_op]):
            train_op = tf.no_op('train')


    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(train_step_num):
            xs,ys = mnist.train.next_batch(batch_size)

            # TypeError: Fetch argument 2.5755167 has invalid type <class 'numpy.float32'>,
            # must be a string or Tensor. (Can not convert a float32 into a Tensor or Operation.)
            _,loss_value,step = sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys})

            if i % 1000 == 0:
                print('after %d step(s),loss:%g' %(i,loss_value))

        final_result = sess.run(y,feed_dict={x:mnist.test.images})
    return final_result

# def visualisation(final_result):
#     # embedding通过tf中的变量完成的，所以projector可视化的都是tf中的变量，这里新定义一个变量保存输出层向量的取值
#     y = tf.Variable(final_result,name=tensor_name)
#     summary_writer = tf.summary.FileWriter(log_dir)
#
#     config = projector.ProjectorConfig()  # 帮助生成日志文件
#     embedding = config.embeddings.add()  # 增加需要可视化的embedding结果
#     embedding.tensor_name = y.name  # 指定embedding结果对应的tf变量名
#     embedding.metadata_path = meta_file  # 指定向量的标签--原始数据信息
#     embedding.sprite.image_path = sprite_file  # 指定sprite图像
#     embedding.sprite.single_image_dim.extend([28,28])  # 指定单张图片的大小
#     projector.visualize_embeddings(summary_writer,config)  #将projector所需的内容写入日志文件
#
#     sess = tf.InteractiveSession()
#     sess.run(tf.global_variables_initializer())
#     saver = tf.train.Saver()
#     saver.save(sess,os.path.join(log_dir,'model'),train_step_num)
#
#     summary_writer.close()


def visualisation(final_result):
    y = tf.Variable(final_result, name=tensor_name)
    summary_writer = tf.summary.FileWriter(log_dir)

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = y.name

    # Specify where you find the metadata
    embedding.metadata_path = meta_file

    # Specify where you find the sprite (we will create this later)
    embedding.sprite.image_path = sprite_file
    embedding.sprite.single_image_dim.extend([28, 28])

    # Say that you want to visualise the embeddings
    projector.visualize_embeddings(summary_writer, config)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(log_dir, "model"), train_step_num)

    summary_writer.close()

def main(argv=None):
    mnist = input_data.read_data_sets('/Users/enjlife/learning-deep-learning-from-scratch/MNIST_data', one_hot=True)
    final_result = train(mnist)
    visualisation(final_result)

if __name__=='__main__':
    tf.app.run()