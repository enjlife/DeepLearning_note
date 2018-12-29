import tensorflow as tf
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #指定系统（服务器或电脑等）中哪些是对 TensorFlow 可见的
from tensorflow.python.platform import gfile
from tensorflow.examples.tutorials.mnist import input_data

# tf框架学习第五章
#思路整理：1.变量初始化   2.定义辅助函数，支持计算参数平均值  3.定义训练函数 添加L2范数的损失函数
#4.y值计算loss，而滑动平均值会维护一份影子变量，最后通过影子变量预测，利用滑动平均模型average_y来做正确率检测



dataset_dir = os.path.dirname(__file__)
save_file = dataset_dir + "/MNIST_data"
# mnist = input_data.read_data_sets(save_file,one_hot=True) #从1.0之后分拆了models
#
# batch_size = 100
# xs,ys = mnist.train.next_batch(batch_size)
# print(ys[0])
# mnist_node = 784
# output_node = 10
# layer1_node = 500
# batch_size = 100
# learning_rate_base = 0.8
# learning_rate_decay = 0.99
# regularization_rate = 0.0001
# train_iter = 30000
# moving_ave_decay = 0.99

# def inference(input_tensor,avg_class,weights1,biases1,weights2,biases2):
#     if avg_class ==None: #由ExponentialMovingAverage生成
#         layer1 = tf.nn.relu(tf.matmul(input_tensor,weights1)+biases1)
#         return tf.matmul(layer1,weights2)+biases2
#     else:
#         layer1 = tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weights1))+avg_class.average(biases1))
#         return tf.matmul(layer1,avg_class.average(weights2))+avg_class.average(biases2)
#
# def train(mnist):
#     x = tf.placeholder(dtype=tf.float32,shape=[None,mnist_node],name='input-x')
#     y_ = tf.placeholder(dtype=tf.float32,shape=[None,output_node],name='output-y')
#
#     #在tf.truncated_normal中如果x的取值在区间（μ-2σ，μ+2σ）之外则重新进行选择
#     weights1 = tf.Variable(tf.truncated_normal([mnist_node,layer1_node],stddev=1))
#     weights2 = tf.Variable(tf.truncated_normal([layer1_node,output_node],stddev=1))
#     biases1 = tf.Variable(tf.constant(0.1,shape=[layer1_node]))
#     biases2 = tf.Variable(tf.constant(0.1,shape=[output_node]))
#     y = inference(x,None,weights1,biases1,weights2,biases2)  #计算loss
#
#     global_step = tf.Variable(0,trainable=False)
#     variable_average = tf.train.ExponentialMovingAverage(moving_ave_decay,global_step)
#     variable_average_op = variable_average.apply(tf.trainable_variables())  #计算所有需要训练的滑动平均值
#     average_y = inference(x,variable_average,weights1,biases1,weights2,biases2)  #检验正确率
#
#     #代价函数，先softmax，然后计算交叉熵
#     cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
#     cross_entropy_mean = tf.reduce_mean(cross_entropy)
#     regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)
#     print(regularizer(weights1))
#     regularization = regularizer(weights1)+regularizer(weights2)
#     loss = cross_entropy_mean+regularization
#
#     learning_rate = tf.train.exponential_decay(learning_rate_base,
#                                                global_step,mnist.train.num_examples/batch_size,learning_rate_decay)
#     train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
#
#     #每过一个batch，反向传播更新参数和参数的滑动平均值，所以将两者一次完成
#     with tf.control_dependencies([train_step,variable_average_op]):
#         train_op = tf.no_op(name='train')
#
#     correct_prediction = tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))  #cast将布尔值转化为float
#
#     with tf.Session() as sess:
#         tf.global_variables_initializer().run()
#         validate_feed = {x:mnist.validation.images,y_:mnist.validation.labels}
#         test_feed = {x:mnist.test.images,y:mnist.test.labels}
#         for i in range(train_iter):
#             if i %1000 ==0:
#                 validate_acc = sess.run(accuracy,feed_dict=validate_feed)
#                 print('After %d, Accuracy:%g' %(i,validate_acc))
#             xs,ys = mnist.train.next_batch(batch_size)
#             sess.run(train_op,feed_dict={x:xs,y_:ys})
#         test_acc = sess.run(accuracy,feed_dict=test_feed)
#         print('test_acc:%g'%(test_acc))
#主程序入口
# def main(argv=None):
#     mnist = input_data.read_data_sets('/tmp/data',one_hot=True)
#     # mnist = {}
#     # (mnist['x_train'],mnist['t_train']),(mnist['x_test'],mnist['t_test']) = mnist.load_mnist(latten=False)
#     train(mnist)

# if __name__ == '__main__':
#     tf.app.run()  #调用上面定义的main（）函数

#运行到29000个迭代出现以下错误
#InvalidArgumentError (see above for traceback): You must feed a value for placeholder tensor
# 'output-y' with dtype float and shape [?,10]
#[[node output-y (defined at /Users/enjlife/learning-deep-learning-from-scratch/tensorflow_note/tensorflow5.py:37)
#  = Placeholder[dtype=DT_FLOAT, shape=[?,10], _device="/job:localhost/replica:0/task:0/device:CPU:0"]()]]

#变量管理
# tf.Variable与tf.get_variable
# v1 = tf.get_variable('v',shape=[1],initializer=tf.constant_initializer(1.0))
# v2 = tf.Variable(tf.constant(1.0,shape=[1],name='v'))
#区别--get_variable的变量名称是一个必填的参数
#如果需要获取创建的变量，需要通过tf.variable_scope来生成一个上下文管理器
# with tf.variable_scope('foo'):
#     v = tf.get_variable('v',initializer=tf.constant_initializer(1.0))
#
# with tf.variable_scope('foo',reuse=True):  #此处不加reuse会报错,添加reuse后variable_scope将只能获取创建过的变量
#     v1 = tf.get_variable('v',[1])
#     print(v == v1)

#当variable_get嵌套时，reuse值随时更新
# with tf.variable_scope('root'):
#     print(tf.get_variable_scope().reuse) #FALSE
#     with tf.variable_scope('foo',reuse=True):
#         print(tf.get_variable_scope().reuse)  #TRUE
#         with tf.variable_scope('bar'):
#             print(tf.get_variable_scope().reuse) #TRUE 还处于上个上下文管理器中
#     print(tf.get_variable_scope().reuse)  #FALSE 退出上面的上下文管理器

#variable_scope管理变量命名空间,获取的变量名会加上空间名--foo/bar/v:0
# with tf.variable_scope('foo'):
#     v = tf.get_variable('v',[1])  #[1]--shape
#
# with tf.variable_scope('foo',reuse=True):
#     v2 = tf.get_variable('v',[1])
#     print(v2.name)
#
# with tf.variable_scope('foo',reuse=True):
#     with tf.variable_scope('bar',reuse=tf.AUTO_REUSE):  #bar空间之前没有变量，需要加tf.AUTO_REUSE
#         v3 = tf.get_variable('v',[1])
#         print(v3.name)
#
#     v4 = tf.get_variable('v',[1])
#     print(v4.name)
#
# with tf.variable_scope('',reuse=tf.AUTO_REUSE):
#     v5 = tf.get_variable('foo/bar/v',[1])  #直接引用带空间名的变量名
#     print(v5==v3)
#     v6 = tf.get_variable('foo/v',[1])
#     print(v6==v4)

#利用vatiable_scope和get_variable对前向传播算法改进
# def inference(input_tensor,reuse=False):
#     with tf.variable_scope('layer1',reuse=reuse):
#         weights =tf.get_variable('weights',[input_node,layer_node],initializer=tf.truncated_normal_initializer(stddev=1))
#         biases = tf.get_variable('biases',[layer_node])
#         #r如果要使用滑动平均模型，每一个with加一条判断avg_class是否为None
#         layer1 = tf.nn.relu(tf.matmul(input_tensor,weights)+biases)
#     with tf.variable_scope('layer2',reuse=reuse):
#         weights =tf.get_variable('weights',[layer1_node,output_node],initializer=tf.truncated_normal_initializer(stddev=1))
#         biases = tf.get_variable('biases',[output_node])
#         layer2 = tf.nn.relu(tf.matmul(layer1,weights)+biases)
#     return layer2
# x = tf.placeholder(dtype=tf.float32,shape=[None,784],name='input-y')
# y = inference(x)


# dataset_dir = os.path.dirname(__file__)
# save_file = dataset_dir + "/model.ckpt"
# save_file1 = dataset_dir+'/model/combined_model.pb'
# save_file_json = dataset_dir+'/model.ckpt.meda.json'


#模型持久化--保存模型和加载模型
v1 = tf.Variable(tf.constant(1.0,shape=[1],name='v1'))
v2 = tf.Variable(tf.constant(2.0,shape=[1],name='v2'))
result = v1+v2
init_op = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init_op)
    saver.save(sess,save_file)

#没有运行变量的初始化过程，而是将变量的值通过已经保存到的模型加载进来
# with tf.Session() as sess:
#     saver.restore(sess,save_file)
#     print(sess.run(result))

#直接加载持久化的图
# saver = tf.train.import_meta_graph(save_file+'.meta')  #注意此处加的点而不是/
# with tf.Session() as sess:
#     saver.restore(sess,save_file)
#     #通过张量的名字获取张量
#     print(sess.run(tf.get_default_graph().get_tensor_by_name('add:0')))

#加载的变量是可以选择和重命名
#saver = tf.train.Saver([v1])  #加载时只加载变量v1，其他变量会变得未初始化
#下列加载会出现模型变量找不到的错误，可以使用一个字典重命名
# v1 = tf.Variable(tf.constant(1.0,shape=[1],name='other-v1'))
# v2 = tf.Variable(tf.constant(2.0,shape=[1],name='other-v2'))
# saver = tf.train.Saver({'v1:v1','v2':v2})  #原来名称为v1的变量加载到变量v1中

#直接保存滑动平均模型的样例，不再需要调用函数获取变量的滑动平均值
# v = tf.Variable(0,dtype=tf.float32,name='v')
# for variables in tf.global_variables():
#     print(variables.name)
# ema = tf.train.ExponentialMovingAverage(0.99)
# maintain_averages_op = ema.apply(tf.global_variables())  #生成一个影子变量 'v:0'和'v/ExponentialMovingAverage:0'
# for variables in tf.global_variables():
#     print(variables.name)
# saver = tf.train.Saver()
# with tf.Session() as sess:
#     init_op = tf.global_variables_initializer()
#     sess.run(init_op)
#
#     sess.run(tf.assign(v,10))
#     sess.run(maintain_averages_op) #滑动平均值
#     saver.save(sess,save_file)  #sess我的理解是保存这个会话，会话有计算的滑动平均值
#     print(sess.run([v,ema.average(v)]))

#通过变量重命名读取变量的滑动平均值---字典
# v= tf.Variable(0,dtype=tf.float32,name='v')
# saver = tf.train.Saver({'v/ExponentialMovingAverage':v})
# with tf.Session() as sess:
#     saver.restore(sess,save_file)
#     print(sess.run(v))  #输出原来变量v的滑动平均值0.099999905

#自动生成字典
# v = tf.Variable(0,dtype=tf.float32,name='v')
# ema = tf.train.ExponentialMovingAverage(0.99)
# print(ema.variables_to_restore())
# saver = tf.train.Saver(ema.variables_to_restore()) #{'v/ExponentialMovingAverage': <tf.Variable 'v:0' shape=() dtype=float32_ref>}
# with tf.Session() as sess:
#     saver.restore(sess,save_file)
#     print(sess.run(v))

#将计算图的变量及取值通过常量的方式保存
# v1 = tf.Variable(tf.constant(1.0,shape=[1],name='v1'))
# v2 = tf.Variable(tf.constant(2.0,shape=[1],name='v2'))
# result = v1+v2
# init_op = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init_op)
#     graph_def = tf.get_default_graph().as_graph_def()  #导出当前计算图的GrapgDef，无关节点并不保存
#     output_graph_def = tf.graph_util.convert_variables_to_constants(sess,graph_def,['add']) #要保存的节点名称
#     with tf.gfile.GFile(save_file1,'wb') as f:
#         f.write(output_graph_def.SerializeToString())

#只需要得到计算图某个节点的取值
# with tf.Session() as sess:
#     model_filename = save_file1
#     with gfile.FastGFile(model_filename,'rb') as f:  #读取保存的文件，将文件解析成对应的Graph Protocol Buffer
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read())
#     result = tf.import_graph_def(graph_def,return_elements=['add:0'])  #将graph_def保存的图加载到当前的图中 return_elements给出返回的张量的名称
#     print(sess.run(result))

#持久化原理及数据格式
#MetaGraph--元图 由MetaGraphDef Protocol Buffer定义
# message Meta{
#     MetaInfoDef meta_info_def = 1
#     MetaInfoDef graph_def = 2;
#     SaverDef saver_def = 3;
#     map<string,CollectionDef> collection_def = 4;
#     map<string,SignatureDef> signature_def = 5
#     repeated AssetFileDef asset_file_def = 6;
# }

#以json格式导出MetaGraphDefProtocol Buffer
# v1 = tf.Variable(tf.constant(1.0,shape=[1],name='v1'))
# v2 = tf.Variable(tf.constant(2.0,shape=[1],name='v2'))
# result = v1+v2
# saver = tf.train.Saver()
# saver.export_meta_graph(save_file_json,as_text=True)  #export_meta_graph

#元数据存储的信息--持久化计算图
#1.meta_info_def--计算图的版本号、用户指定的标签 stripped_op_list--计算图所使用到的运算方法
# stripped_op_list的类型是Oplist，Oplist是一个OpDef类型的列表，定义了运算的名称、
# input_arg和output_arg列表 attr给出了其他的运算参数信息
#tensorflow_version和tenflow_git_version 记录了生成当前计算图的tensorflow版本
#2.graph_def记录tensorflow的节点信息，主要信息存储在node中，记录了tensorflow上的所有节点信息，
# name--节点唯一标识，op--该节点使用的tensorflow运算方法的名称，input--运算的输入--node：src_output,表示指定节点的第几个输出
#当src_output为0可以省略 node:0可以被计为node。 device属性指定了处理这个运算的设备。 attr指定了和当前运算相关的配置信息
#3.saver_def--记录了持久化模型需要用到的一些参数，保存到文件的文件名、保存和加载操作的名称以及保存频率、清理历史记录
#4.collection_def--集合名称到集合内容的映射--4类集合Nodelist维护计算图上节点的集合 BytesList维护字符串或者系列化之后Procotol
#Buffer的集合 Int64List维护整数集合  FloatList维护实数集合

#持久化变量的取值
#model.ckpt.data和model.ckpt.data-****-of-**** 文件保存了所有变量的取值

#查看保存的变量信息
# reader = tf.train.NewCheckpointReader(save_file)  #ckpt
# global_variables = reader.get_variable_to_shape_map()  #获取所有变量列表
# for variable_name in global_variables:
#     print(variable_name,global_variables[variable_name])  #变量名称 and  变量维度
# print('v1 value:',reader.get_tensor('v1'))  #名称为v1的变量的取值
#最后一个文件名是固定的--checkpoint--维护了tensorflow模型文件的文件名，模型删除后，模型对应的文件名也会从checkpoint删除


#模块化--训练和测试--见mnist





