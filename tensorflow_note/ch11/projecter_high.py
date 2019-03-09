import tensorflow as tf
import sys, os
sys.path.append(os.pardir)
import numpy as np
import codecs
import collections
from operator import itemgetter
from ch05 import mnist_inference  # 发现文件夹名字带 - 的无法引用
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
dir_name = os.path.dirname(__file__)
summary_dir = dir_name

log_dir = dir_name
sprite_file = 'mnist_sprite.jpg'
meta_file = 'mnist_meta.tsv'

# 组合projector需要的日志名和地址相关参数
def create_sprite_image(images):
    if isinstance(images,list):
        images = np.array(images)

    img_h = images.shape[1]
    img_w = images.shape[2]

    m = int(np.ceil(np.sqrt(images.shape[0])))
    sprite_image = np.ones((img_h*m,img_w*m))

    for i in range(m):
        for j in range(m):
            cur = m*i+j
            if cur<images.shape[0]:
                sprite_image[i*img_h:(i+1)*img_h,j*img_w:(j+1)*img_w] = images[cur]
    return sprite_image

mnist = input_data.read_data_sets('/Users/enjlife/learning-deep-learning-from-scratch/MNIST_data',one_hot=False)
to_visualise = 1-np.reshape(mnist.test.images,(-1,28,28))
sprite_image = create_sprite_image(to_visualise)
path_mnist_sprites = os.path.join(log_dir,sprite_file)
plt.imsave(path_mnist_sprites,sprite_image,cmap='gray')
plt.imshow(sprite_image,cmap='gray')

path_mnist_metadata = os.path.join(log_dir,meta_file)
with open(path_mnist_metadata,'w') as f:
    f.write('index\tlabel\n')
    for index,label in enumerate(mnist.test.labels):
        f.write('%d\t%d\n' %(index,label))  # \t 横向制表符 \v 纵向制表符

