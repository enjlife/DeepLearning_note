import tensorflow as tf
import sys, os
import pandas as pd
sys.path.append(os.pardir)
import numpy as np
import codecs
import collections
from operator import itemgetter
from tensorflow.examples.tutorials.mnist import input_data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""
tensorflow官方的高级封装



"""


file_dataset = '/Users/enjlife/learning-deep-learning-from-scratch/tensorflow_note/ch06/MNIST_data'
tf.logging.set_verbosity(tf.logging.INFO)

# mnist = input_data.read_data_sets(file_dataset,one_hot=True)
# # 指定输入层，所有的输入都会拼接成一个输入
# feature_columns = [tf.feature_column.numeric_column('image',shape=[784])]
#
# estimator = tf.estimator.DNNClassifier(feature_columns=feature_columns,
#                                        hidden_units=[500],
#                                        n_classes=10,
#                                        optimizer=tf.train.AdamOptimizer(),
#                                        model_dir='')
# train_input_fn = tf.estimator.inputs.numpy_input_fn(x={'image':mnist.train.images},
#                                                     y=mnist.train.labels.astype(np.int32),
#                                                     num_epochs=None,
#                                                     batch_size=128,
#                                                     shuffle=True
#                                                     )
# estimator.train(input_fn=train_input_fn,steps=10000)
#
# test_input_fn = tf.estimator.inputs.numpy_input_fn(x={'images':mnist.test.images},
#                                                    y=mnist.test.labels.astype(np.int32),
#                                                    num_epochs=1,
#                                                    batch_size=128,
#                                                    shuffle=False)
# accuracy_score = estimator.evaluate(input_fn=test_input_fn)['accuracy']
# print('\nTest accuracy:%g %%' %(accuracy_score*100))

# estimator自定义模型 自定义模型，套用estimator的训练

#定义网络结构
# def lenet(x,is_training):
#     x = tf.reshape(x,shape=[-1,28,28,1])
#
#     net = tf.layers.conv2d(x,32,5,activation=tf.nn.relu)
#     net = tf.layers.max_pooling2d(net,2,2)
#     net = tf.layers.Conv2D(net,64,3,activation=tf.nn.relu)
#     net = tf.layers.max_pooling2d(net,2,2)
#     net = tf.contrib.layers.flatten(net)
#     net = tf.layers.dense(net,1024)
#     net = tf.layers.dropout(net,rate=0.4,training=is_training) #训练为True，执行dropout
#     return tf.layers.dense(net,10)
# # 定义模型的输出
# # mode 有三种可能 train evalute predict
# def model_fn(features,labels,mode,params):  # model_params可以添加所需的超参数
#     predict = lenet(features['image'],mode==tf.estimator.ModeKeys.TRAIN)
#
#     if mode == tf.estimator.ModeKeys.PREDICT:
#         # 如果是预测模式，直接返回结果
#         return tf.estimator.EstimatorSpec(mode=mode,predictions={'result':tf.argmax(predict,1)})
#
#     loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predict,labels=labels))
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])
#
#     train_op = optimizer.minimize(loss,global_step=tf.train.get_global_step())
#     eval_metric_ops = {'my_metric':tf.metrics.accuracy(tf.argmax(predict,1),labels)}
#
#     return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op,eval_metric_ops=eval_metric_ops)
#
# mnist = input_data.read_data_sets(file_dataset,one_hot=False)
# model_params = {'learning_rate':0.01}
# # estimator.Estimator类 定义整个模型和变量参数
# estimator = tf.estimator.Estimator(model_fn=model_fn,params=model_params)
#
# train_input_fn = tf.estimator.inputs.numpy_input_fn(x={'image':mnist.train.images},
#                                                     y=mnist.train.labels.astype(np.int32),
#                                                     num_epochs=None,
#                                                     batch_size=128,
#                                                     shuffle=True)
# # train会传递 mode值==tf.estimator.ModeKeys.TRAIN 为True
# estimator.train(input_fn=train_input_fn,steps=30000)
#
# test_input_fn = tf.estimator.inputs.numpy_input_fn(x={'image':mnist.test.images},
#                                                    y=mnist.test.labels.astype(np.int32),
#                                                    num_epochs=1,
#                                                    batch_size=128,
#                                                    shuffle=False)
# test_results = estimator.evaluate(input_fn=test_input_fn)
#
# accuracy_score = test_results['my_metric']
# print('\ntest accuracy: %g %%' %(accuracy_score))
#
# predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={'images':mnist.test.images[:10]},
#                                                       num_epochs=1,
#                                                       shuffle=False)
# predictions = estimator.predict(input_fn=predict_input_fn)
# for i,p in enumerate(predictions):
#     print('prediction %s:%s' %(i+1,p['result']))


# 使用数据集作为estimator输出
def my_input_fn(file_path,perform_shuffle=False,repeat_count=1):
    def decode_csv(line):
        passed_line = tf.decode_csv(line,[[0.0],[0.0],[0.0],[0.0],[0]])
        return {'x':passed_line[:-1]},passed_line[-1:]
    dataset = (tf.contrib.data.TextLineDataset(file_path).skip(1).map(decode_csv))  # skip读取时跳过前n行
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.repeat(repeat_count)
    dataset = dataset.bathch(32)
    iterator = dataset.make_one_shot_iterator()

    batch_features,batch_labels = iterator.get_next()
    return batch_features,batch_labels
# 类似placeholder 占位
feature_columns = [tf.feature_column.numeric_column('x',shape=[4])]
# 构建两层全连接层
classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,hidden_units=[10,10],n_classes=3)

classifier.train(input_fn=lambda :my_input_fn('',True,10))

# Estimator要求自定义函数不能有参数，通过lambda表达式可以将函数转化为不带参数的函数
test_results = classifier.evaluate(input_fn=lambda :my_input_fn('',False,1))
print('\ntest accuracy: %g %%' %(test_results['accuracy']*100))





