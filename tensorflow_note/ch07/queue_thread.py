import tensorflow as tf
import os
import threading
import numpy as np
import time
from ch07 import tfrecord
from ch06 import mnist_inference
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""
队列 多线程
dataset--数据集
引用失败--需要引用的文件通过编译



"""

dir_name = os.path.dirname(__file__)
#操作队列
# q = tf.FIFOQueue(2,'int32')
# init = q.enqueue_many(([0,10],))
# x = q.dequeue()
# y = x+1
# q_inc = q.enqueue([y])
# with tf.Session() as sess:
#     init.run()
#     for _ in range(5):
#         v, _ = sess.run([x,q_inc])
#         print(v)

#利用Coordinator协助线程 shoule_stop/request_stop/join 三个函数
# def MyLoop(coord,work_id):
#     while not coord.should_stop():
#         if np.random.rand()<0.1:
#             print('stop from id:%d\n' %(work_id))
#             coord.request_stop()
#
#         else:
#             print('working on id:%d\n'%(work_id))
#
#         time.sleep(1)
#
# coord = tf.train.Coordinator()
#
# threads = [threading.Thread(target=MyLoop,args=(coord,i)) for i in range(5)]
# for i in threads:
#     i.start()
#
# coord.join(threads)

# 启动多个线程操作队列

# queue = tf.FIFOQueue(100,'float')  # 建立队列
# enqueue_op = queue.enqueue(tf.random_normal([1]))  # 入列
# qr = tf.train.QueueRunner(queue,[enqueue_op]*5)  # 出列
# tf.train.add_queue_runner(qr)  # 将定义过的runner加入指定的集合
# out_tensor = queue.dequeue()
#
# with tf.Session() as sess:
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess=sess,coord=coord)  #启动所有线程
#     for _ in range(3):
#         print(sess.run(out_tensor)[0])
#
#     coord.request_stop()
#     coord.join(threads)

#创建数据样例
# def _int64_feature(value):
#     return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
#
# num_shards = 2
# instances_per_shard = 2
# for i in range(num_shards):
#    filename = (dir_name+'/data.tfrecords-%.5d-of-%.5d'%(i,num_shards))
#
#    writer = tf.python_io.TFRecordWriter(filename)
#
#    for j in range(instances_per_shard):
#        example = tf.train.Example(features=tf.train.Features(feature={'i':_int64_feature(i),
#                                                                       'j':_int64_feature(j)}))
#        writer.write(example.SerializeToString())
#
#    writer.close()

# #利用match_filenames_once和string_input_producer 读取、解析
# files = tf.train.match_filenames_once(dir_name+'/data.tfrecords-*')
# filname_queue = tf.train.string_input_producer(files,shuffle=False)
# reader = tf.TFRecordReader()
# _,serialized_example = reader.read(filname_queue)
# features = tf.parse_single_example(serialized_example,features={'i':tf.FixedLenFeature([],tf.int64),
#                                                                 'j':tf.FixedLenFeature([],tf.int64)})
# with tf.Session() as sess:
#     tf.local_variables_initializer().run()
#     print(sess.run(files))
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess=sess,coord=coord)
#
#     for i in range(4):
#         print(sess.run([features['i'],features['j']]))
#
#     coord.request_stop()
#     coord.join(threads)

#组合训练数据
# files = tf.train.match_filenames_once(dir_name+'/data.tfrecords-*')
# filname_queue = tf.train.string_input_producer(files,shuffle=False)
# reader = tf.TFRecordReader()
# _,serialized_example = reader.read(filname_queue)
# features = tf.parse_single_example(serialized_example,features={'i':tf.FixedLenFeature([],tf.int64),
#                                                                 'j':tf.FixedLenFeature([],tf.int64)})
#
# batch_size = 3
# example,label = features['i'],features['j']
#
# capacity = 1000 + 3*batch_size  # 设置队列大小
# #组合样例
# example_batch,label_batch = tf.train.batch([example,label],batch_size=batch_size,capacity=capacity)
#
# with tf.Session() as sess:
#     #tf.initialize_all_variables().run()
#     # global_variable_initializer和local作用不同，此外以下代码输出3个batch，一个batch3个样例
#     tf.global_variables_initializer().run()
#     tf.local_variables_initializer().run()
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess=sess,coord=coord)
#     for i in range(3):
#
#         cur_example_batch,cur_label_batch = sess.run([example_batch,label_batch])
#         print(cur_example_batch,cur_label_batch)
#     coord.request_stop()
#     coord.join(threads)

#tf.train.shuffle_batch
# files = tf.train.match_filenames_once(dir_name+'/data.tfrecords-*')
# filname_queue = tf.train.string_input_producer(files,shuffle=False)
# reader = tf.TFRecordReader()
# _,serialized_example = reader.read(filname_queue)
# features = tf.parse_single_example(serialized_example,features={'i':tf.FixedLenFeature([],tf.int64),
#                                                                 'j':tf.FixedLenFeature([],tf.int64)})
#
# batch_size = 3
# example,label = features['i'],features['j']
#
# capacity = 1000 + 3*batch_size  # 设置队列大小
# example_batch,label_batch = tf.train.shuffle_batch([example,label],batch_size=batch_size,capacity=capacity,min_after_dequeue=30)
# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     tf.local_variables_initializer().run()
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess=sess,coord=coord)
#     for i in range(3):
#         cur_example_batch, cur_label_batch = sess.run([example_batch, label_batch])
#         print(cur_example_batch,cur_label_batch)
#     coord.request_stop()
#     coord.join(threads)


#输入数据处理框架

# files = tf.train.match_filenames_once(dir_name+'/data.tfrecords-*')
# filname_queue = tf.train.string_input_producer(files,shuffle=False)
# reader = tf.TFRecordReader()
# _,serialized_example = reader.read(filname_queue)
# features = tf.parse_single_example(serialized_example,features={'image':tf.FixedLenFeature([],tf.string),
#                                                                 'label':tf.FixedLenFeature([],tf.int64),
#                                                                 'height':tf.FixedLenFeature([],tf.int64),
#                                                                 'width':tf.FixedLenFeature([],tf.int64),
#                                                                 'channesl':tf.FixedLenFeature([],tf.int64)})
# image,label = features['image'],features['label']
# height,width = features['height'],features['width']
# channels = features['channels']
#
# #从原始图像解析出像素矩阵,根据尺寸还原图像
# decoded_image = tf.decode_raw(image,tf.uint8)
# decoded_image.set_shape([height,width,channels])
#
# image_size = 229
# distored_image = tfrecord.preprocess_for_train(decoded_image,image_size,image_size,None)
#
# min_after_dequeue = 10000
# batch_size = 100
# capacity = min_after_dequeue + 3*batch_size
# image_batch,label_batch = tf.train.shuffle_batch([distored_image,label],batch_size=batch_size,
#                                                  capacity=capacity,min_after_dequeue=min_after_dequeue)
#
# learning_rate = 0.01
# logit = inference(image_batch)
# loss = calc_loss(logit,label_batch)
# train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer(),tf.local_variables_initializer())
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess=sess,coord=coord)
#
#     train_rounds = 5000
#     for i in range(train_rounds):
#         sess.run(train_step)
#
#     coord.request_stop()
#     coord.join(threads)


#数据集
# input_data = [1,2,3,5,8]
# dataset = tf.data.Dataset.from_tensor_slices(input_data)  # 数据集从一个张量中构建
#
# #定义一个迭代器遍历数据集，上面定义的数据集没用placeholder 更加灵活的initializable_iterator
# iterator = dataset.make_one_shot_iterator()
# x = iterator.get_next()  #从遍历器中读取张量
# y = x*x
# with tf.Session() as sess:
#     for i in range(len(input_data)):
#         print(sess.run(y))

#真实数据集通常保存在硬盘，像自然语言处理，训练数据以每行一条数据的形式存在文本文件中
# input_files = ['','']  # 提供多个文件
# dataset = tf.data.TextLineDataset(input_files)
# iterator = dataset.make_one_shot_iterator()
# x = iterator.get_next()
# with tf.Session() as sess:
#     for i in range(3):
#         print(sess.run(x))

#在图像相关任务中，可以使用TFRecord存储，需要提供一个parse来解析读取的TFRecord
# def parse(record):
#     features = tf.parse_single_example(record,features={'feat1':tf.FixedLenFeature([],tf.int64),
#                                                         'feat2':tf.FixedLenFeature([],tf.int64)})
#     return features['feat1'],features['feat2']
# input_files = ['','']
# dataset = tf.data.TFRecordDataset(input_files)
# dataset = dataset.map(parse)
# iterator = dataset.make_one_shot_iterator()
# feat1,feat2 = iterator.get_next()
# with tf.Session() as sess:
#     for i in range(10):
#         f1,f2 = sess.run([feat1,feat2])

#当使用placeholder初始化数据集，需要用到initializable_iterator
# def parse(record):
#     features = tf.parse_single_example(record,features={'feat1':tf.FixedLenFeature([],tf.int64),
#                                                         'feat2':tf.FixedLenFeature([],tf.int64)})
#     return features['feat1'],features['feat2']
#
# input_files = tf.placeholder(tf.string)  # 采用占位，不需要将参数写入计算图
#
# dataset = tf.data.TFRecordDataset(input_files)
# dataset = dataset.map(parse)  #对二进制数据解析
#
# iterator = dataset.make_initializable_iterator()  # 定义遍历dataset的initializable_iterator
# feat1,feat2 = iterator.get_next()
#
# with tf.Session() as sess:
#     sess.run(iterator.initializer,feed_dict={input_files:['','']})
#     #遍历一个epoch，遍历结束时，程序抛出OutOfRangeError
#     while True:
#         try:
#             sess.run()
#         except tf.errors.OutOfRangeError:
#             break

#使用map对每一条数据调用preprocess_for_train
# dataset = dataset.map(lambda x:preprocess_for_train(x,image_size,image_size,None))
#
# dataset = dataset.shuffle(buffer_size)  # 缓冲区保存 buffer_size = min_after_dequeue
# dataset = dataset.batch(batch_size)
# dataset = dataset.repeat(N)  # 将数据重复N份
# concatenate()连接两个数据集
# take(N) 读取前N项数据  skip(N)跳过前N项数据
# flap_map() 从多个数据集中轮流读取数据

#数据输入流程
train_files = tf.train.match_filenames_once('')
test_files = tf.train.match_filenames_once('')
def parse(record):
    features = tf.parse_single_example(record,features={'image':tf.FixedLenFeature([],tf.string),
                                                        'label':tf.FixedLenFeature([],tf.int64),
                                                        'height':tf.FixedLenFeature([],tf.int64),
                                                        'width':tf.FixedLenFeature([],tf.int64),
                                                        'channesl':tf.FixedLenFeature([],tf.int64)})
    decoded_image = tf.decode_raw(features['images'],tf.uint8)
    decoded_image.set_shape(features['height'],features['width'],features['channels'])
    label = features['label']
    return decoded_image,label

image_size = 299
batch_size = 100
shuffle_buffer = 10000

dataset = tf.data.TFRecordDataset(train_files)
dataset = dataset.map(parse)
dataset = dataset.map(lambda image,label:(tfrecord.preprocess(image,image_size,image_size,None),label))
dataset = dataset.shuffle(shuffle_buffer).batch(batch_size)

num_epoch = 10
dataset = dataset.repeat(num_epoch)
iterator = dataset.make_initializable_iterator()
image_batch,label_batch = iterator.get_next()

learning_rate = 0.01
logit = mnist_inference.inference(image_batch)
loss = mnist_inference.calc_loss(logit,label_batch)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

test_dataset = tf.data.TFRecordDataset(test_files)
test_dataset = dataset.map(parse).map(lambda image,label:(tf.image.resize_images(image,[image_size,image_size]),label))
test_dataset = test_dataset.batch(batch_size)

test_iterator = test_dataset.make_initializable_iterator()
test_image_batch,test_label_batch = iterator.get_next()

test_logit = mnist_inference.inference(test_image_batch)
predictions = tf.argmax(test_logit,axis=1,output_type=tf.int32)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer(),
             tf.local_variables_initializer())
    sess.run(iterator.initializer)
    while True:
        try:
            sess.run(train_step)
        except tf.errors.OutOfRangeError:
            break
    sess.run(test_iterator.initializer)
    test_results = []
    test_labels = []
    while True:
        try:
            pred,label = sess.run([predictions,test_label_batch])
            test_results.append(pred)
            test_labels.append(test_label_batch)
        except tf.errors.OutOfRangeError:
            break

    correct = [float(y==y_) for y,y_ in zip(test_results,test_labels)]
    accuracy = sum(correct)/ len(correct)
    print('accuracy:',accuracy)






