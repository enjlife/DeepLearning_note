import glob
import os.path
import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

dir_name = os.path.dirname(__file__)
input_data = '/Users/enjlife/flower_processed_data.npy'
train_file = dir_name
ckpt_file = '/Users/enjlife/inception_v3.ckpt'
learning_rate = 0.0001
steps = 300
n_classes = 5

# 不需要从谷歌训练好的模型中加载的参数。
CHECKPOINT_EXCLUDE_SCOPES = 'InceptionV3/Logits,InceptionV3/AuxLogits'
# 需要训练的网络层参数明层，在fine-tuning的过程中就是最后的全联接层。
TRAINABLE_SCOPES='InceptionV3/Logits,InceptionV3/AuxLogit'

def get_tuned_variables():
    exclusions = [scope.strip() for scope in CHECKPOINT_EXCLUDE_SCOPES.split(',')] #strip()默认删除空白符（包括'\n', '\r',  '\t',  ' ')
    variables_to_restore = []
    for var in slim.get_model_variables():  #枚举模型中所有的参数，判断是否需要从加载列表删除
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startwith(exclusion):  #直接打印.op.name就能打印出名称 .get_shape().as_list()就输出shape
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
    return variables_to_restore

def get_trainable_variable():
    scopes = [scope.strip() for scope in TRAINABLE_SCOPES.split(',')]
    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope)
        variables_to_train.extend(variables)
    return variables_to_train

def main():
    processed_data = np.load(input_data)
    training_images = processed_data[0]
    training_labels = processed_data[1]
    validation_images = processed_data[2]
    validation_labels = processed_data[3]
    test_images = processed_data[4]
    n_training_example = len(training_labels)
    test_labels = processed_data[5]
    print("%d train example,%d validation example,%dtest test  "
          "example" %(n_training_example,len(validation_labels),len(test_labels)))
    images = tf.placeholder(tf.float32,[None,299,299,3],name='input_images')
    labels = tf.placeholder(tf.int64,[None],name='labels')

    with slim.arg_scope(inception_v3.inception_v3_arg_scope):
        logits,_ = inception_v3.inception_v3(images,num_classes=n_classes)

    trainable_variables = get_trainable_variable()
    #softmax_cross_entropy(onehot_labels=,logits,weights=1) onehot也是一种概率分布

    #sparse_softmax_cross_entropy()，参数logits的形状 [batch_size, num_classes] 和labels的形状[batch_size]--输入的label为 tf.argmax(Y, axis=1)
    #softmax_cross_entropy_with_logits labels,logits必须有相同的形状 [batch_size, num_classes]概率分布

    tf.losses.softmax_cross_entropy(tf.one_hot(labels,n_classes),logits,weights=1.0)
    train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(tf.losses.get_total_loss())  #get_global_loss--返回其值表示总损失的张量

    with tf.name_scope('evalution'):
        correct_prediction = tf.equal(tf.argmax(logits,1),labels)
        evalution_step = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    load_fn = slim.assign_from_checkpoint_fn(ckpt_file,get_tuned_variables(),ignore_missing_vars=True)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        print('loading tuned variable from %s' %ckpt_file)
        load_fn(sess)

        start = 0
        end = BATCH
        for i in range(steps):
            sess.run(train_step,feed_dict={images:training_images[start:end],labels:training_labels[start:end]})
            if i % 30 ==0 or i+1==steps:
                saver.save(sess,train_file,global_step=i)
                validation_accuracy = sess.run(evalution_step,feed_dict={images:validation_images,labels:validation_labels})
                print('Step %d:Validation accuracy:%.1f%%'%(i,validation_accuracy*100))

            start = end
            if start == n_training_example:
                start = 0
            end = start + BATCH
            if end > n_training_example:
                end = n_training_example

        test_accuracy = sess.run(evalution_step,feed_dict={images:test_images,labels:test_labels})
        print('Final test accuracy = %.1f%%' %(test_accuracy*100))

if __name__=='__main__':
    tf.app.run()


