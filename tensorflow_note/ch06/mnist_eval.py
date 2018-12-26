import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os,sys
from tensorflow_note.ch06 import mnist_inference,mnist_train


model_path_checkpoint = os.path.dirname(__file__)
eval_internal_secs = 10  #每十秒加载一次新的模型
def evalute(mnist):
    with tf.Graph().as_default() as g:

        x = tf.placeholder(tf.float32,[None,mnist_inference.image_size,
                                       mnist_inference.image_size,mnist_inference.num_channel],name='x-input')
        y_ = tf.placeholder(tf.float32,[None,mnist_inference.output_data_num],name='y-input')
        x_reshape = mnist.validation.images.reshape(mnist.validation.num_examples,mnist_inference.image_size,
                                       mnist_inference.image_size,mnist_inference.num_channel)
        validate_feed = {x:x_reshape,y_:mnist.validation.labels}
        y = mnist_inference.inference(x,None)
        correct_pre = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_pre,tf.float32))

        #变量重命名的方式加载模型
        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.moving_average_decay)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        while True:
            with tf.Session() as sess:
                #get_checkpoint_state通过check文件自动找到最新模型的文件名
                ckpt = tf.train.get_checkpoint_state(model_path_checkpoint)
                print(ckpt)
                print(ckpt.model_checkpoint_path)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]  #split拆分为不同的数组并取值
                    accuracy_score = sess.run(accuracy,feed_dict=validate_feed)
                    print('After %s steps,accuracy = %g' %(global_step,accuracy_score))
                else:
                    print('No checkpoint found')
                    return
            time.sleep(eval_internal_secs)

def main(argv=None):
    mnist = input_data.read_data_sets(mnist_train.data_set,one_hot=True)
    evalute(mnist)

if __name__ =='__main__':
    tf.app.run()

