import tensorflow as tf
import os
import threading
import numpy as np
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

queue = tf.FIFOQueue(100,'float')  # 建立队列
enqueue_op = queue.enqueue(tf.random_normal([1]))  # 入列
qr = tf.train.QueueRunner(queue,[enqueue_op]*5)  # 出列
tf.train.add_queue_runner(qr)  # 将定义过的runner加入指定的集合
out_tensor = queue.dequeue()

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)  #启动所有线程
    for _ in range(3):
        print(sess.run(out_tensor)[0])

    coord.request_stop()
    coord.join(threads)
