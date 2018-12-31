import numpy as np
import os
import tensorflow as tf
import matplotlib as mpt
# mpt.use('Agg') # 没有GUI时画图，而且必须添加在import matplotlib.pyplot之前，否则无效
from matplotlib import pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


"""
rnn循环神经网络--序列需要规定一个最长长度，否则会出现梯度消散和梯度爆炸的问题
LSTM--长短时记忆网络
双层循环神经网络和深层循环神经网络

假如我们输入有100个句子，每个句子都由5个单词组成，而每个单词用64维的词向量表示。那么samples=100，timesteps=5，input_dim=64
imesteps就是输入序列的长度input_length(视情而定)，因而此节每一层有timestamps个cell，而输入的序列就是一维的向量，隐藏节点的权重为（1，hidden——size）


"""
# x = [1,2]
# state = [0.0,0.0]
# w_cell_state = np.asarray([[0.1,0.2],[0.3,0.4]])
# w_cell_input = np.asarray([0.5,0.6])
# b_cell = np.asarray([0.1,-0.1])
# w_output = np.asarray([[1.0],[2.0]])
# b_output = 0.1
#
# for i in range(len(x)):
#     before_activation = np.dot(state,w_cell_state)+x[i]*w_cell_input+b_cell
#     state = np.tanh(before_activation)
#     filnal_output = np.dot(state,w_output)+b_output
#     print('before activation:',before_activation)
#     print('state:',state)
#     print('output:',filnal_output)

# LSTM
# lstm_hidden_size = 5
# batch_size = 100
# num_steps = 100
# lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_hidden_size)
# state = lstm.zero_state(batch_size,tf.float32)  # 全0初始化状态，state是一个包含两个张量的LSTMStateTuple--state.c和state.h
# loss = 0.0
#
# for i in range(num_steps):  # num_steps 训练数据的序列长度,将循环网络展开成前馈神经网络  dynamic_rnn动态处理变长序列
#     # http://wiki.jikexueyuan.com/project/tensorflow-zh/how_tos/variable_scope.html
#     if i>0: tf.get_variable_scope().reuse_variables()  # 第一时刻声明LSTM结构使用的变量，之后复用之前定义好的变量
#     lstm_output,state = lstm(current_input,state)
#     final_output = fully_connected(lstm_output)
#     loss +=calc_loss(final_output,expected_output)


# 深层循环神经网络
# lstm_cell = tf.nn.rnn_cell.BasicLSTMCell
# # 在BasicLSTMCell基础再封装一层MultiRNNCell`
# stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(lstm_size) for _ in range(number_of_layers)])
# state = stacked_lstm.zero_state(batch_size,tf.float32)
# for i in range(len(num_steps)):
#     if i >0:tf.get_variable_scope().reuse_variables()
#     stacked_lstm_output,state = stacked_lstm(current_input,state)
#     final_output = fully_connected(stacked_lstm_output)
#     loss +=calc_loss(final_output,expected_output)

#循环神经网络的dropout
# lstm_cell = tf.nn.rnn_cell.BasicLSTMCell
# # Dropout类有两个参数input_keep_prob控制输入Dropout  output_keep_prob控制输出Dropout
# stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.DropoutWrapper(lstm_cell(lstm_cell))for _ in range(number_of_layers)])

# 循环神经网络样例应用
hidden_size = 30  # 隐藏节点
num_layers = 2
timesteps = 10  # 序列长度
training_steps = 10000
batch_size = 32

training_examples = 10000
testing_examples = 1000
sample_gap = 0.01  # 采样间隔

# 将序列生成x和y相对应
def generate_data(seq):
    x = []
    y = []
    # 第i项到i+timesteps-1项为输入，第i+timesteps项为输出
    for i in range(len(seq)-timesteps):
        x.append([seq[i:i+timesteps]])
        y.append([seq[i+timesteps]])

    return np.array(x,dtype=np.float32),np.array(y,dtype=np.float32)

def lstm_model(x,y,is_training):
    cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(hidden_size) for _ in range(num_layers)])
    #将多层的lstm结构连接成rnn网络并计算前向传播结果
    outputs,_ =tf.nn.dynamic_rnn(cell,x,dtype=tf.float32)
    # 顶层输出结果，[batch_size,time,hidden_size] hidden指的是每一个cell内的全连接层的隐藏节点数
    output = outputs[:,-1,:]
    predictions = tf.contrib.layers.fully_connected(output,1,activation_fn=None)  # fully_connected(输入，输出一个，激活函数无)
    #print(predictions.get_shape().as_list())
    if not is_training:
        return predictions,None,None  # train和eval函数都提取三个值
    loss = tf.losses.mean_squared_error(labels=y,predictions=predictions)
    # (loss,global_step,learning_rate,optimizer)
    train_op = tf.contrib.layers.optimize_loss(loss,tf.train.get_global_step(),optimizer='Adagrad',learning_rate=0.1)
    return predictions,loss,train_op

def train(sess,train_x,train_y):
    ds = tf.data.Dataset.from_tensor_slices((train_x,train_y))
    ds = ds.repeat().shuffle(1000).batch(batch_size)
    x,y = ds.make_one_shot_iterator().get_next()

    with tf.variable_scope('model'):
        preddictions,loss,train_op = lstm_model(x,y,True)

    sess.run(tf.global_variables_initializer())
    for i in range(training_steps):
        _,l = sess.run([train_op,loss])
        if i % 100==0:
            print('train step:'+str(i)+'loss:'+str(l))

def run_eval(sess,test_x,test_y):
    ds = tf.data.Dataset.from_tensor_slices((test_x,test_y))
    ds = ds.batch(1)
    x,y = ds.make_one_shot_iterator().get_next()
    with tf.variable_scope('model',reuse=True):  # 重用model空间的模型数据
        prediction, _, _ = lstm_model(x,y=[0.0],is_training=False)  # y只在训练时计算loss使用，test不需要y

    predictions = []
    labels = []
    for i in range(testing_examples):
        p,l = sess.run([prediction,y])
        predictions.append(p)
        labels.append(l)

    # squeeze函数的作用是除去维度为1的维度
    # 如[[1],[2],[3]]是一个维度为[3,1]的数组，除去维度为1的维度后
    # 变成一个维度为[3]数组[1,2,3]
    predictions = np.array(predictions).squeeze()
    print(predictions.shape)  # 打印tuple类型
    labels = np.array(labels).squeeze()
    rmse = np.sqrt(((predictions-labels)**2).mean(axis=0))
    print('mean square error is :%f'%rmse)

    plt.figure()
    plt.plot(predictions,label='predictions')
    plt.plot(labels,label='real_sin')
    plt.legend()
    plt.show()

test_start = (training_examples+timesteps)*sample_gap
test_end = test_start+(testing_examples+timesteps)*sample_gap
train_x,train_y = generate_data(np.sin(np.linspace(0,test_start,training_examples+timesteps,dtype=np.float32)))
test_x,test_y = generate_data(np.sin(np.linspace(test_start,test_end,testing_examples+timesteps,dtype=np.float32)))
print(test_x.shape)

with tf.Session() as sess:
    train(sess,train_x,train_y)
    run_eval(sess,test_x,test_y)









