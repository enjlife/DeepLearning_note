import tensorflow as tf

slim = tf.contrib.slim

#卷积神经网络案例--终于可以体验到tf对卷积网络的友好性了。
# filter_weight = tf.get_variable('weight',[5,5,3,16],initializer=tf.truncated_normal_initializer(stddev=1))  #（过滤器尺寸，深度，当前节点矩阵的深度）
# biases = tf.get_variable('biases',[16],initializer=tf.constant_initializer(0.1))  #input为思维[第几张图片，节点矩阵]
# conv = tf.nn.conv2d(input,filter_weight,strides=[1,1,1,1],padding='SAME') #strides为不同维度上的步长，第一维和最后一维要求是1
# bias = tf.nn.bias_add(conv,biases)
# actived_conv = tf.nn.relu(bias)

#池化层
# pool = tf.nn.max_pool(actived_conv,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')

#LeNet-5 模型

#Inception-v3
net = slim.conv2d = (input,32,[3,3])  #输入节点矩阵，当前过滤器的深度，过滤器的尺寸
with slim.arg_scope([slim.conv2d,slim.maxpool2,slim.avg_pool2d],stride=1,padding='VALID'):
    net = a#上一层的输出节点矩阵
    with tf.variable_scope('Mixed_7c'):
        with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net,320,[1,1],scope='Conv2d_0a_1x1')

        with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net,385,[1,1],scope='Conv2d_0a_1x1')
            branch_1 = tf.concat(3,[slim.conv2d(branch_1,384,[1,3],scope='Conv_0b_1x3'),
                                    slim.conv2d(branch_1,384,[3,1],scope='Conv2d_0c_3x1')])
        with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net,448,[1,1],scope='Con2d_0a_3x3')
            branch_2 = slim.conv2d(branch_2,384,[3,3],scope='Conv_0b_3x3')
            branch_2 = tf.concat(3,[slim.conv2d(branch_2,384,[1,3],scope='Conv2d_0c_1x3'),
                                    slim.conv2d(branch_2,384,[3,1],scope='Conv2d_0d_3x1')])

        with tf.variable_scope('Branch_3'):
            branch_3 = slim.avg_pool2d(net,[3,3],scope='AvgPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3,384,[3,1],scope='Conv2d_0d_3x1')
            net = tf.concat(3,[branch_0,branch_1,branch_2,branch_3])

#卷积网络迁移学习--将一个问题上训练好的模型通过简单的调整使其适用于一个新的问题
#下载数据集
# wget http://download.tensorflow.org/example_images/flower_photos.tgz
# tar xzf flower_photo.tgz
