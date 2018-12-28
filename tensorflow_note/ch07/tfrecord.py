import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mnist_file = '/Users/enjlife/learning-deep-learning-from-scratch/tensorflow_note/MNIST_data'
save_file = os.path.dirname(__file__)+'/output.tfrecords'

#将mnist数据转化为tfrecord格式
# def _init64_feature(value):  #生成整数型的属性
#     return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
# def _bytes_feature(value):  #生成字符串的属性
#     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
# mnist = input_data.read_data_sets(mnist_file,dtype=tf.uint8,one_hot=True)
# labels = mnist.train.labels
# images = mnist.train.images
# pixels = images.shape[1]  #训练数据集的图像分辨率
# print(pixels)
# num_example = mnist.train.num_examples
# filename = save_file+'/output.tfrecords'
# writer = tf.python_io.TFRecordWriter(filename)
#
# for index in range(num_example):
#     image_raw = images[index].tostring()  #将图像矩阵转化为字符串
#     #将样例转化为Example Protocol Buffer 将所有信息写入这个数据结构
#     #tf.train.Example(__init__,SerializeToString()) 通常__init__函数传入tf.train.Features
#     example = tf.train.Example(features=tf.train.Features(feature={'pixels':_init64_feature(pixels),
#                                                                    'label':_init64_feature(np.argmax(labels[index])),
#                                                                    'image_raw':_bytes_feature(image_raw)}))
#     writer.write(example.SerializeToString())
#
# writer.close()

#读取tfrecord格式数据
# reader = tf.TFRecordReader()
# filename_queue = tf.train.string_input_producer([save_file])
# _,serialized_example = reader.read(filename_queue)
# features = tf.parse_single_example(serialized_example,features={'image_raw':tf.FixedLenFeature([],tf.string),
#                                                                 'pixels':tf.FixedLenFeature([],tf.int64),
#                                                                 'label':tf.FixedLenFeature([],tf.int64)})
# image = tf.decode_raw(features['image_raw'],tf.uint8)
# label = tf.cast(features['label'],tf.int32)
# pixels = tf.cast(features['pixels'],tf.int32)
#
# #多线程
# sess = tf.Session()
# coord = tf.train.Coordinator()
# threads = tf.train.start_queue_runners(sess=sess,coord=coord)
# for i in range(10):
#     print(sess.run([image,label,pixels]))

#图像数据处理，解码
# image_raw_data = tf.gfile.GFile('/Users/enjlife/WechatIMG93.jpeg','rb').read()
# with tf.Session() as sess:
#     img_data = tf.image.decode_jpeg(image_raw_data)
#     #print(img_data.eval())  #tensor.eval
#     print(sess.run(img_data))
#     plt.imshow(img_data.eval())
#     plt.show()
    # encoded_image = tf.image.encode_jpeg(img_data)
    # with tf.gfile.GFile(os.path.dirname(__file__)+'/output','wb') as f:
    #     f.write(encoded_image.eval())

#图像大小调节
# image_raw_data = tf.gfile.GFile('/Users/enjlife/WechatIMG93.jpeg','rb').read()
# with tf.Session() as sess:
#     img_data = tf.image.decode_jpeg(image_raw_data)
#     img_data = tf.image.convert_image_dtype(img_data,dtype=tf.float32)
    #img_data.eval()
    #img_data.set_shape([1797, 2673, 3])
    #print(img_data.get_shape())

    # resized = tf.image.resize_images(img_data,[300,300],method=0)
    # #调整图像大小
    # croped = tf.image.resize_image_with_crop_or_pad(resized, 1000, 1000)
    # padded = tf.image.resize_image_with_crop_or_pad(resized, 3000, 3000)
    # #图像翻转
    # flipped1 = tf.image.flip_up_down(img_data)
    # flipped2 = tf.image.random_flip_left_right(img_data)
    # flipped3 = tf.image.transpose_image(img_data)
    # #随机翻转
    # flipped4 = tf.image.random_flip_left_right(img_data)
    # flipped5 = tf.image.random_flip_up_down(img_data)
    # #图像色彩调整
    # adjusted = tf.image.adjust_brightness(img_data,-0.5)  #将图像的亮度调整到-0.5
    # max_delta = 0.5
    # adjusted = tf.image.random_brightness(img_data,max_delta) #在-max_delta到max_delta范围随机调整
    # adjusted = tf.clip_by_value(adjusted,0.0,1.0) #色彩调整可能导致实数值超过0.0-1.0需要在多项处理后截断处理
    # #图像对比度调整
    # adjusted = tf.image.adjust_contrast(img_data,0.5)
    # adjusted = tf.image.adjust_contrast(img_data,0.5)
    # adjusted = tf.image.adjust_contrast(img_data,[lower,upper])
    # #调整图像的色相
    # adjusted = tf.image.adjust_hue(img_data,0.1)
    # adjusted = tf.image.adjust_hue(img_data,0.6)
    # adjusted = tf.image.adjust_hue(img_data,max_delta) #max_delta取值在[0,0.5]
    # #调整图像饱和度
    # adjusted = tf.image.adjust_saturation(img_data,-5)
    # adjusted = tf.image.adjust_saturation(img_data,5)
    # adjusted - tf.image.adjust_saturation(img_data,lower,upper)
    # #图像标准化--均值变为0，方差变为1
    # adjusted = tf.image.per_image_standardization(img_data)

    # plt.imshow(resized.eval())
    # plt.show()

    #处理标注框
    # img_data = tf.image.resize_images(img_data,[180,267],method=1)
    # batched = tf.expand_dims(tf.image.convert_image_dtype(img_data,tf.float32),0) #将解码后的数据添加一维--思维数组
    # boxes = tf.constant([[[0.05,0.05,0.9,0.7],[0.35,0.47,0.5,0.56]]])
    # result = tf.image.draw_bounding_boxes(batched,boxes)
    # print(result.eval())
    # plt.imshow(result[0].eval())
    # plt.show()

    #随机截取图像
    # boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])
    #
    # #随机截取，至少包含boxes的40%的内容，获取三个值，begin，size，bbox  tf.shape()获取[height, width, channels]
    # begin,size,bbox_for_draw = tf.image.sample_distorted_bounding_box(tf.shape(img_data),
    #                                                                   bounding_boxes=boxes,min_object_covered=0.8)
    # batched = tf.expand_dims(tf.image.convert_image_dtype(img_data,tf.float32),0)
    # image_with_box = tf.image.draw_bounding_boxes(batched,bbox_for_draw)
    # distorted_image = tf.slice(img_data,begin,size)
    # plt.imshow(distorted_image.eval())
    # plt.show()


#图像预处理
def distort_color(image,color_ordering=0):   #调整色彩
    if color_ordering == 0 :
        image = tf.image.random_brightness(image,max_delta=32.0/255.0)
        image = tf.image.adjust_saturation(image,lower=0.5,upper=1.5)
        image = tf.image.random_hue(image,max_delta=0.2)
        image = tf.image.random_contrast(image,lower=0.5,upper=1.5)

    elif color_ordering == 1:

        image = tf.image.adjust_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
    return tf.clip_by_value(image,0.0,1.0)

def preprocess_for_train(image,height,width,bbox):  #裁剪并resize
    if bbox is None:
        bbox = tf.constant([0.0,0.0,1.0,1.0],dtype=tf.float32,shape=[1,1,4])

    if image.dtype !=tf.float32:
        image = tf.image.convert_image_dtype(image,dtype=tf.float32)

    bbox_begin,bbox_size,_=tf.image.sample_distorted_bounding_box(tf.shape(image),bounding_boxes=bbox)
    distorted_image = tf.slice(image,bbox_begin,bbox_size)
    distorted_image = tf.image.resize_images(distorted_image,[height,width],method=np.random.randint(4))
    distorted_image = tf.image.random_flip_up_down(distorted_image)
    distorted_image = distort_color(distorted_image,np.random.randint(2))
    return distorted_image

image_raw_data = tf.gfile.GFile('','r').read()

with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    boxes = tf.constant([[[0.05,0.05,0.9,0.7],[0.35,0.47,0.5,0.56]]])
    for i in range(6):
        result = preprocess_for_train(img_data,299,299,boxes)
        plt.imshow(result.eval())
        plt.show()









