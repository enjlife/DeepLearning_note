import tensorflow as tf
import sys, os
sys.path.append(os.pardir)
import numpy as np
import codecs
import collections
from operator import itemgetter
from tflearn.layers.core import fully_connected
import keras
from keras.preprocessing import sequence
from keras.datasets import mnist
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input,Dense,Flatten,Conv2D,MaxPooling2D,Embedding,LSTM
from keras.datasets import imdb
from keras import backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""
keras底层支持tensorflow、MXNet、CNTK 和Theano

"""
file_dataset = '/Users/enjlife/learning-deep-learning-from-scratch/tensorflow_note/ch06/MNIST_data'

# num_class = 10
# img_rows,img_cols = 28,28
# (trainx,trainy),(testx,testy) = mnist.load_data()  # Downloading data from https://s3.amazonaws.com/img-datasets/mnist.npz
#
# if K.image_data_format() =='channels_first':
#     trainx = trainx.reshape(trainx.shape[0],1,img_rows,img_cols)
#     testx = testx.reshape(testx.shape[0],1,img_rows,img_cols)
#     input_shape = (1,img_rows,img_cols)
# else:
#     trainx = trainx.reshape(trainx.shape[0], img_rows, img_cols, 1)
#     testx = testx.reshape(testx.shape[0], img_rows, img_cols, 1)
#     input_shape = (img_rows, img_cols,1)
#
# trainx = trainx.astype('float32')
# testx = testx.astype('float32')
# trainx /= 255.0
# testx /= 255.0
# # 将标准答案转化为需要的格式(one-hot)
# trainy = keras.utils.to_categorical(trainy,num_class)
# testy = keras.utils.to_categorical(testy,num_class)
#
# # 使用Keras api定义模型
# model = Sequential()
# model.add(Conv2D(32,kernel_size=(5,5),activation='relu',input_shape=input_shape))
# model.add(MaxPool2D(pool_size=(2,2)))
# model.add(Conv2D(64,(5,5),activation='relu'))
# model.add(MaxPool2D(pool_size=(2,2)))
# model.add(Flatten())
# model.add(Dense(500,activation='relu'))
# model.add(Dense(10,activation='softmax'))
# # 定义损失函数、优化函数、评测方法
# model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.SGD(),metrics=['accuracy'])
# model.fit(trainx,trainy,batch_size=128,epochs=20,validation_data=(testx,testy))
#
# score = model.evaluate(testx,testy)
#
# print('test loss:%g' % score[0])
# print(('test accuracy: %g' % score[1]))

# 循环神经网络
# max_features = 20000
# maxlen = 80
# batch_size = 32
# (trainx,trainy),(testx,testy) = imdb.load_data(num_words=max_features)
# print(len(trainx),'train sequence')
# print(len(testx),'test sequence')
# trainx = sequence.pad_sequences(trainx,maxlen)  # 长度不够的用0填充，超过长度的则忽略掉超过的部分
# testx = sequence.pad_sequences(testx,maxlen)
# print('train_shape:',trainx.shape)
# print('test_shape:',testx.shape)
#
# model = Sequential()
# model.add(Embedding(max_features,128))  # 128为词向量的维度
# model.add(LSTM(128,dropout=0.2,recurrent_dropout=0.2))  #只会得到最后一个输出
# model.add(Dense(1,activation='sigmoid'))
#
# model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
# model.fit(trainx,trainy,batch_size=batch_size,epochs=15,validation_data=(testx,testy))
#
# score = model.evaluate(testx,testy,batch_size=batch_size)
# print('test loss:',score[0])
# print(('test accuracy:',score[1]))

# Keras高级用法
# Keras支持以返回值的形式定义网络层结构

num_class = 10
(trainx,trainy),(testx,testy) = mnist.load_data()

trainx = trainx.astype('float32')
testx = testx.astype('float32')
trainx /= 255.0
testx /= 255.0
inputs = Input(shape=(784,)) # 输入层
x = Dense(500,activation='relu')(inputs)
predictions = Dense(10,activation='softmax')(x)  # keras封装的categorical_crossentropy 类别交叉熵
model = Model(inputs=inputs,outputs=predictions)
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.SGD(),metrics=['accuracy'])
model.fit(trainx,trainy,batch_size=128,epochs=20,validation_data=(testx,testy))

#实现类似inception的模型--一对多 多对一
# input_img = Input(shape=(256,256,3))
#
# tower_1 = Conv2D(64,(1,1),padding='same',activation='relu')(input_img)
# tower_1 = Conv2D(64,(3,3),padding='same',activation='relu')(tower_1)
#
# tower_2 = Conv2D(64,(1,1),padding='same',activation='relu')(input_img)
# tower_2 - Conv2D(64,(5,5),padding='same',activation='relu')(tower_2)
# tower_3 = MaxPooling2D((3,3),strides=(1,1),padding='same')(input_img)
# tower_3 = Conv2D(64,(1,1),padding='same',activation='relu')(tower_3)
#
# output = keras.layers.concatenate([tower_1,tower_2,tower_3],axis=1)  #axis连接的应该是深度


#多输入输出
num_class = 10
(trainx,trainy),(testx,testy) = mnist.load_data()
input1 = Input(shape=(784,),name='input1')
input2 = Input(shape=(10,0),name='input2')

trainx = trainx.astype('float32')
testx = testx.astype('float32')
trainx /= 255.0
testx /= 255.0
# 将标准答案转化为需要的格式(one-hot)
trainy = keras.utils.to_categorical(trainy,num_class)
testy = keras.utils.to_categorical(testy,num_class)

x = Dense(1,activation='relu')(input1)
output1 = Dense(10,activation='softmax',name='output1')(x)

y = keras.layers.concatenate([x,input2])
output2 = Dense(10,activation='softmax',name='output2')(y)
model = Model(inputs=[input1,input2],outputs=[output1,output2])

# loss_weight 为不同的输出产生的损失指定权重
# sigmoid和softmax是神经网络输出层使用的激活函数，分别用于两类判别和多类判别。
# binary cross-entropy和categorical cross-entropy是相对应的损失函数。
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.SGD(),loss_weights=[1.0,0.1]
              ,metrics=['accuracy'])

model.fit([trainx,trainy],[trainy,trainy],batch_size=128,epochs=20,validation_data=([testx,testy],[testy,testy]))

# keras还存在两大问题：需要将数据一次加载到内存；原生态keras无法支持分布式训练
# 可以结合keras和原生态tensorflow，会损失一定的易用性，但建模的灵活性提高

from tensorflow.examples.tutorials.mnist import input_data
mnist_data = input_data.read_data_sets(file_dataset,one_hot=True)
x = tf.placeholder(tf.float32,shape=[None,784])
y_ = tf.placeholder(tf.float32,shape=[None,10])

net = tf.keras.layers.Dense(500,activation='relu')(x)
y = tf.keras.layers.Dense(10,activation='softmax')(net)
loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_,y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
acc_value = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(y_,y))  # 这里平均的应该是测试集的多个batch的准确率

with tf.Session() as sess:
    tf.global_variables_initializer.run()
    for i in range(10000):
        xs,ys = mnist_data.train.next_batch(100)
        _,loss = sess.run([train_step,loss],feed_dict={x:xs,y_:ys})
        if i % 1000 == 0:
            print('after %d train step(s) loss : %g' %(i,loss))

    acc = acc_value.eval(feed_dict={x:mnist_data.test.images,y:mnist_data.test.labels})














