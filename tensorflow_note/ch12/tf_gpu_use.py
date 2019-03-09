import tensorflow as tf
import os
from datetime import datetime
import time
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from ch05 import mnist_inference

# with tf.device('/cpu:0'):
#     a = tf.constant([1.0,2.0,3.0],shape=[3],name='a')
#     b = tf.constant([1.0,2.0,3.0],shape=[3],name='b')
# with tf.device('/cpu:0'):
#     c = a + b
# # log_device_placement 打印运行每一个运算的设备
# with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
#     print(sess.run(c))
#
# # 有的操作指定在GPU会报错
# a_cpu= tf.Variable(0,name='a_cpu')
#
# with tf.device('/gpu:0'):
#     a_gpu = tf.Variable(0,name='a_gpu')
#
# with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
#     sess.run(tf.global_variables_initializer())

# 通过指定allow_soft_placement参数
# a_cpu = tf.Variable(0,name='a_cpu')
# with tf.device('/gpu:0'):
#     a_gpu = tf.Variable(0,name='a_gpu')
#     sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True))
#     sess.run(tf.global_variables_initializer())
#     sess.close()

# tensorflow会默认占用设备所有GPU和GPU的显存，如果使用部分GPU，可以设置CUDA_VISIBLE_DEVICES
# # 设置以下环境变量
# CUDA_VISIBLE_DEVICE=1 python demo.py  # 只使用第二块
# CUDA_VISIBLE_DEVICES=0,1 python demo_code.py  # 只使用前两块
#
# # 只使用第三块gpu
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
#
# # tf也支持动态分配显存
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
#
# # 或者按固定比例分配
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
# session = tf.Session(config=config,)

#深度学习训练并行模式--异步和同步

# 多GPU并行，在一台机器的多个gpu,因为性能相似，所以采用同步模式训练模型

# batch_size = 50
# learning_rate_base = 0.95
# learning_rate_decay = 0.96
# regularization_rate = 0.0001
# train_step_num = 10000
# moving_average_decay = 0.99
# n_gpu = 2
#
# model_save_path = 'log_and_models'
# model_name = 'model.ckpt'
# data_path = 'output.tfrecord'
# def get_input():
#     dataset = tf.contrib.data.TFRecordDataset([data_path])
#
#     # 定义数据格式
#     def parse(record):
#         features = tf.parse_single_example(record,fetures={'image_raw':tf.FixedLenFeature([],tf.string),
#                                                        'pixels':tf.FixedLenFeature([],tf.int64),
#                                                        'label':tf.FixedLenFeature([],tf.int64)})
#         decoded_image = tf.decode_raw(features['image_raw'],tf.uint8)
#         reshaped_image = tf.reshape(decoded_image,[784])
#         retyped_image = tf.cast(reshaped_image,tf.float32)
#         label = tf.cast(features['label'],tf.int32)
#
#         return retyped_image,label
# # 定义输入队列
#     dataset = dataset.map(parse)
#     dataset = dataset.shuffle(buffer_size=10000)
#     dataset = dataset.repeat(10)
#     dataset = dataset.batch(batch_size)
#     iterator = dataset.make_one_shot_iterator()
#
#     features,labels = iterator.get_next()
#     return features,labels
#
# def get_loss(x,y_,regularizer,scope,reuse_variables=None):
#     with tf.variable_scope(tf.get_variable_scope(),reuse=reuse_variables):
#         y = mnist_inference.inference(x,regularizer)
#     cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_,logits=y))
#     regularizer_loss = tf.add_n(tf.get_collection('losses',scope))
#     loss = cross_entropy + regularizer_loss
#     return loss
#
# def average_gradients(tower_grads):
#     average_grads = []
#
#     for grad_and_vars in zip(*tower_grads):
#         grads = []
#         for g,_ in grad_and_vars:
#             expanded_g = tf.expand_dims(g,0)
#             grads.append(expanded_g)
#         grad = tf.concat(grads,0)
#         grad = tf.reduce_mean(grad,0)
#         # 如果有两个gpu 则是一个tuple包含两个列表，每一个列表包含梯度和变量
#         v = grad_and_vars[0][1]
#         grad_and_var = (grad,v)
#         average_grads.append(grad_and_var)
#     return average_grads
#
# def main(argv=None):
#     with tf.Graph.as_default(),tf.device('/cpu:0'):
#         x,y_ = get_input()
#         regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)
#
#         global_step = tf.get_variable('global_step',[],initializer=tf.constant_initializer(0),trainable=False)
#         learning_rate = tf.train.exponential_decay(learning_rate_base,global_step,60000/batch_size,learning_rate_decay)
#         opt = tf.train.GradientDescentOptimizer(learning_rate)
#
#         tower_grads = []
#         reuse_variables = False
#
#         for i in range(n_gpu):
#             with tf.device('/gpi:%d' %i):
#                 with tf.name_scope('gpu_%d'%i) as scope:
#                     cur_loss = get_loss(x,y_,regularizer,scope,reuse_variables)
#                     reuse_variables = True
#                     grads = opt.compute_gradients(cur_loss)
#                     tower_grads.append(grads)
#         grads = average_gradients(tower_grads)
#         for grad,var in grads:
#             if grad is not None:
#                 tf.summary.histogram('gradients_on_average/%s' % var.op.name,grad)
#
#         apply_gradient_op = opt.apply_gradients(grads,global_step=global_step)
#         for var in tf.trainable_variables():
#             tf.summary.histogram(var.op.name,var)
#
#         variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay,global_step)
#         variables_to_average = (tf.trainable_variables() + tf.moving_average_variables())
#         variables_to_average_op = variable_averages.apply(variables_to_average)
#
#         train_op = tf.group(apply_gradient_op,variables_to_average_op)
#
#         saver = tf.train.Saver()
#         summary_op = tf.summary.merge_all()
#         init = tf.global_variables_initializer()
#         with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)) as sess:
#             init.run()
#             summary_writer = tf.summary.FileWriter(model_save_path,sess.graph)
#
#             for step in range(train_step_num):
#                 start_time = time.time()
#                 _,loss_value = sess.run([train_op,cur_loss])
#                 duration = time.time() - start_time
#
#                 if step !=0 and step % 10 ==0:
#                     num_example_per_step = batch_size*n_gpu
#                     example_per_sec = num_example_per_step/duration
#                     sec_per_batch = duration/n_gpu
#
#                     format_str = ('%s: step %d,loss = %.2f ($.1 example/sec; %.3f sec/batch)')
#                     print(format_str % (datetime.now(),step,loss_value,example_per_sec,sec_per_batch))
#
#                     summary = sess.run(summary_op)
#                     summary_writer.add_summary(summary,step)
#
#                 if step % 1000 ==0 or (step+1) ==train_step_num:
#                     checkpoint_path = os.path.join(model_save_path,model_name)
#                     saver.save(sess,checkpoint_path,global_step=step)
# if __name__=='__main__':
#     tf.app.run()
# GRADED FUNCTION: iou

















