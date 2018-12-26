import glob
import os.path
import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3

dir_name = os.path.dirname(__file__)
# input_data = dir_name + '/data.npy'
# processed_data = np.load(input_data)
# training_images = processed_data[0]
# training_labels = processed_data[1]
# validation_images = processed_data[2]
# validation_labels = processed_data[3]
# test_images = processed_data[4]
# test_labels = processed_data[5]
# print("%d train example,%d validation example,%dtest test  "
#           "example" %(len(training_labels),len(validation_labels),len(test_labels)))
print(dir_name)
