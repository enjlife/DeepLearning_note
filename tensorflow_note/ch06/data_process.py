import glob
import os.path
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

dir_name = os.path.dirname(__file__)
input_data = '/Users/enjlife/flower_photos'
output_data = '/Users/enjlife/flower_processed_data.npy'
validation_per = 10
test_per = 10

def create_image_lists(sess,test_per,validation_per):

    sub_dirs=[x[0] for x in os.walk(input_data)]  #返回文件夹路径--父级文件夹、各个子文件夹
    is_root_dir = True
    training_images = []
    training_labels = []
    test_images = []
    test_labels = []
    validation_imges = []
    validation_labels = []
    current_label = 0

    for sub_dir in sub_dirs:
        print(sub_dir)
        if is_root_dir:  #第一个为True，排除掉父级文件夹
            is_root_dir = False
            continue  #跳过本次循环执行下一次循环

        extensions = ['jpg','jpeg','JPG','JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)  #返回path最后的文件名
        for extension in extensions:
            file_glob = os.path.join(input_data,dir_name,'*.' + extension)  #合并目录 os.path.join('/flower_photos',meigui,'*.' jpg)
            file_list.extend(glob.glob(file_glob))  #获得C盘下的所有txt文件--glob.glob(r’c:*.txt’)

        if not file_list: continue  #file_list为空时，跳出本次循环
        print(dir_name)
        i=0
        for file_name in file_list[:200]:
            i += 1
            image_raw_data = gfile.FastGFile(file_name,'rb').read()  #读取并文件
            image = tf.image.decode_jpeg(image_raw_data)  #解析文件
            if image.dtype !=tf.float32:
                image = tf.image.convert_image_dtype(image,dtype=tf.float32)

            image = tf.image.resize_images(image,[299,299])
            image_value = sess.run(image)

            chance = np.random.randint(100)
            if chance<validation_per:
                validation_imges.append(image_value)
                validation_labels.append(current_label)
            elif chance<(test_per+validation_per):
                test_images.append(image_value)
                test_labels.append(current_label)
            else:
                training_images.append(image_value)
                training_labels.append(current_label)
            if i % 200==0:
                print(i,'processed')
        current_label +=1

    state = np.random.get_state()
    np.random.shuffle(training_images)
    np.random.set_state(state)
    np.random.shuffle(training_labels)

    return np.asarray([training_images,training_labels,validation_imges,validation_labels,test_images,test_labels])

def main():
    with tf.Session() as sess:
        processed_data = create_image_lists(sess,test_per,validation_per)
        np.save(output_data,processed_data)

if __name__=='__main__':
    main()

