import tensorflow as tf
import sys, os
sys.path.append(os.pardir)
import numpy as np
import codecs
import collections
from operator import itemgetter
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
dir_name = os.path.dirname(__file__)

# input1 = tf.constant([1.0,2.0,3.0],name='input1')
# input2 = tf.Variable(tf.random_uniform([3]),name='input2')
# output = tf.add_n([input1,input2],name='add')
#
# writer = tf.summary.FileWriter(dir_name,tf.get_default_graph())
# writer.close()
