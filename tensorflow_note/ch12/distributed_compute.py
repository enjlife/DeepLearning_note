import tensorflow as tf
import os
from datetime import datetime
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from ch05 import mnist_inference

# c = tf.constant('hello,distributed tnesorflow!')
# server = tf.train.Server.create_local_server()
# with tf.Session(server.target) as sess:
#     print(sess.run(c))

# 第一个任务
c = tf.constant('hello from server1!')
cluster = tf.train.ClusterSpec({'local':['localhost:2222','localhost:2223']})
server = tf.train.Server(cluster,job_name='local',task_index=0)
with tf.Session(server.target,config=tf.ConfigProto(log_device_placement=True)) as sess:
    print(sess.run(c))
server.join()

# 第二个任务
c = tf.constant('hello from server2!')
cluster = tf.train.ClusterSpec({'local':['localhost:2222','localhost:2223']})
server = tf.train.Server(cluster,job_name='local',task_index=1)
with tf.Session(server.target,config=tf.ConfigProto(log_device_placement=True)) as sess:
    print(sess.run(c))
server.join()
