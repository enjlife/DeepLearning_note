import tensorflow as tf
import sys, os
sys.path.append(os.pardir)
import numpy as np
import codecs
import collections
from operator import itemgetter
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


"""
1.两种方式计算交叉熵
2.文本文件预处理
3.与第八章的循环神经网络不同的是--词向量层和softmax层
4.完整的训练场程序--参考https://blog.csdn.net/mydear_11000/article/details/52414342

"""


raw_data = '/Users/enjlife/simple-examples/data/ptb.train.txt'
dir_name = os.path.dirname(__file__)
#vocab_output = dir_name+'/ptb.vocab'
vocab_output = 'ptb.vocab'
output_data = 'ptb.train'



# word_labels = tf.constant([2,0])
# predict_logits = tf.constant([[2.0,-1.0,3.0],[1.0,0.0,-0.5]])
# loss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=word_labels,logits=predict_logits)
#
# # labels以概率的形式给出
# word_prob_distribution = tf.constant([[0.0,0.0,1.0],[1.0,0.0,0.0]])
# loss2 = tf.nn.softmax_cross_entropy_with_logits(labels=word_prob_distribution,logits=predict_logits)
#
# # 使用label smoothing的技巧，避免模型与数据过拟合
# word_prob_smooth = tf.constant([[0.01,0.01,0.98],[0.98,0.01,0.01]])
# loss3 = tf.nn.softmax_cross_entropy_with_logits(labels=word_prob_smooth,logits=predict_logits)
# with tf.Session() as sess:
#     print(sess.run(loss1))
#     print(sess.run(loss2))
#     print(sess.run(loss3))

# counter = collections.Counter()
# with codecs.open(raw_data,'r','utf-8') as f:
#     for line in f:
#         for word in line.strip().split():
#             counter[word] +=1
#
# sorted_word_to_cnt = sorted(counter.items(),key=itemgetter(1),reverse=True)
# sorted_words = [x[0] for x in sorted_word_to_cnt]
# sorted_words = sorted_words+['<eos>']  # 此处添加会影响列表的顺序
# #sorted_words = sorted_words+['<unk>']+['<sos>']+['<eos>']
# # if len(sorted_words)>10000:
# #     sorted_words = sorted_words[:10000]
# with codecs.open(vocab_output,'w','utf-8') as file_output:
#     for word in sorted_words:
#         file_output.write(word+'\n')

#将词汇文件转化为单词编号

#读取词汇表，并建立词汇到单词的映射
# with codecs.open(vocab_output,'r','utf-8') as f_vocab:
#     vocab = [w.strip() for w in f_vocab.readlines()]  # 一行一个单词
#
# word_to_id = {k:v for (k,v) in zip(vocab,range(len(vocab)))}
#
# def get_id(word):
#     return word_to_id[word] if word in word_to_id else word_to_id['<unk>']
#
# fin = codecs.open(raw_data,'r','utf-8')
# fout = codecs.open(output_data,'w','utf-8')
# for line in fin:
#     words = line.strip().split()+['<eos>']
#     out_line = ' '.join([str(get_id(w)) for w in words]) + '\n'  # list=['1','2','3','4','5'] ' '.join(list)='1 2 3 4 5'
#     fout.write(out_line)
#
# fin.close()
# fout.close()

#ptb数据的batching
# train_data = 'ptb.train'
# train_batch_size = 1
# train_num_step = 1
#
# def read_data(file_path):
#     with open(file_path,'r') as fin:
#         id_string = ''.join([line.strip() for line in fin.readlines()])
#     id_list = [int(w) for w in id_string.split()]
#     return id_list
#
# def make_batch(id_list,batch_size,num_step):
#     num_batches = (len(id_list)-1) // (batch_size*num_step)  # 整数除法
#     data = np.array(id_list[:num_batches*batch_size*num_step])  # 截取整乘后得到的list
#     data = np.reshape(data,[batch_size,num_batches*num_step])
#     data_batches = np.split(data,num_batches,axis=1)  # 切分成num_batches个batch，存入一个数组
#
#     label = np.array(id_list[1:num_batches*batch_size*num_step+1])
#     label = np.reshape(label,[batch_size,num_batches*num_step])
#     label_batches = np.split(label,num_batches,axis=1)
#     return list(zip(data_batches,label_batches))  # 返回一个num_batches的数组，每一项包含一个data矩阵和label矩阵
#
# def main():
#     train_batches = make_batch(read_data(train_data),train_batch_size,train_num_step)
#
# if __name__ == '__main__':
#     main()

# 词向量的维度是emb_size，词汇表的大小是vocab_size，那么可以将所有单词放入一个大小为vocab_size*emb_size的矩阵
# embedding = tf.get_variable('embedding',[vocab_size,emb_size])
# # 使用look_up读取词向量
# input_embedding = tf.nn.embedding_lookup(embedding,input_data)
# # softmax层--将循环神经网络的输出映射为一个维度与词汇表大小相同的向量即logits
# weight = tf.get_variable('weight',[hidden_size,vocab_size])
# bias = tf.get_variable('bias',[vocab_size])
# # 计算线性映射--output输出的维度[batch_size*num_steps,hidden_size]
# logits = tf.nn.bias_add(tf.matmul(output,weights),bias)
# # 转化为加和为1的概率
# probs = tf.nn.softmax(logits)
#
# loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(set.targets,[-1]),logits=logits)

#完整的训练程序
train_data = 'ptb.train'
eval_data = 'ptb.valid'
test_data = 'ptb.test'
hidden_size = 300
num_layers = 2
vocab_size = 10000
train_batch_size = 20
train_num_step = 35
eval_num_batch = 1
eval_num_step = 1
num_epoch = 5 #使用训练数据的轮数
lstm_keep_prob = 0.9  # lstm节点不被dropout的概率
embedding_keep_prob = 0.9  # 词向量不被dropout的概率
max_grad_norm = 5  # 控制梯度膨胀的梯度大小上限
share_emb_and_softmax = True  # 是否共享参数

# 定义一个类来表示模型
class PTBModel(object):
    def __init__(self, is_training, batch_size, num_steps):
        self.batch_size = batch_size
        self.num_steps = num_steps

        self.input_data = tf.placeholder(tf.int32,[batch_size,num_steps])
        self.targets = tf.placeholder(tf.int32,[batch_size,num_steps])  # 预期输出

        dropout_keep_prob = lstm_keep_prob if is_training else 1.0  # lstm节点
        lstm_cells = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(hidden_size),output_keep_prob=dropout_keep_prob) for _ in range(num_layers)]
        cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)

        self.initial_state = cell.zero_state(batch_size,tf.float32)  # 初始化最初的状态，只在每个epoch初始化第一个batch时使用 需要[batch_size，s]来存储整个batch每个seq的状态

        #embedding = tf.get_variable('embedding',[vocab_size,hidden_size])  # 定义单词的词向量矩阵，词汇表的大小*词向量的维度（hidden_size)
        embedding = tf.get_variable("embedding", [vocab_size, hidden_size])
        inputs = tf.nn.embedding_lookup(embedding,self.input_data)  # 将输入的单词转化为词向量矩阵，维度batch_size * num_steps * hidden_size
        if is_training:
            inputs = tf.nn.dropout(inputs,embedding_keep_prob)  # 由于这个仅对tensor进行dropout（而非rnn_cell进行wrap），因此调用的是tf.nn.dropout。

        outputs = []
        state = self.initial_state
        with tf.variable_scope('rnn'):
            for time_step in range(num_steps):
                if time_step>0:
                    tf.get_variable_scope().reuse_variables()
                cell_output,state = cell(inputs[:,time_step,:],state)
                outputs.append(cell_output)
        output = tf.reshape(tf.concat(outputs,1),[-1, hidden_size])  # 在第二个维度做连接，然后reshape

        if share_emb_and_softmax:
            weight = tf.transpose(embedding)
        else:
            weight = tf.get_variable('weight',[hidden_size,vocab_size])
        bias = tf.get_variable('bias',[vocab_size])
        logits = tf.matmul(output,weight) + bias

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(self.targets,[-1]),logits=logits)
        self.cost = tf.reduce_sum(loss)/batch_size
        self.final_state = state

        if not is_training: return   # 如果不在训练，返回

        trainable_variables = tf.trainable_variables()
        grads,_ = tf.clip_by_global_norm(tf.gradients(self.cost,trainable_variables),max_grad_norm)  # 控制梯度大小  clip_gradient=max_grad_norm
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)  # 定义优化方法
        self.train_op = optimizer.apply_gradients(zip(grads,trainable_variables))  # 训练

def run_epoch(session,model,batches,train_op,output_log,step):
    total_costs = 0.0
    iters = 0
    state = session.run(model.initial_state)  # 执行获得全0状态
    for x,y in batches:

        cost,state,_ = session.run([model.cost,model.final_state,train_op],{model.input_data:x,model.targets:y,model.initial_state:state})
        total_costs += cost
        iters += model.num_steps
        if output_log and step % 100==0:  #只在训练输出日志
            print('after %d step,perplexity is %.3f' %(step,np.exp(total_costs/iters)))
        step +=1
    return step,np.exp(total_costs/iters)

# def read_data(file_path):
#     with open(file_path,'r') as fin:
#         id_string = ' '.join([line.strip() for line in fin.readlines()])
#     id_list = [int(w) for w in id_string.split()]
#     return id_list

def read_data(file_path):
    with open(file_path, "r") as fin:
        # 将整个文档读进一个长字符串。
        id_string = ' '.join([line.strip() for line in fin.readlines()])
    id_list = [int(w) for w in id_string.split()]  # 将读取的单词编号转为整数
    return id_list

def make_batches(id_list,batch_size,num_step):
    num_batches = (len(id_list)-1)//(batch_size*num_step)
    data = np.array(id_list[:num_step*batch_size*num_batches])
    data = np.reshape(data,[batch_size,-1])
    data_batches = np.split(data,num_batches,axis=1)

    label = np.array(id_list[1:num_step*num_batches*batch_size+1])
    label = np.reshape(label,[batch_size,-1])
    label_batches = np.split(label,num_batches,axis=1)  # 切成包含num_batches个数组的列表，每个数组的维度是batch_size*num_step
    return list(zip(data_batches,label_batches))

def main():
    initializer = tf.random_uniform_initializer(-0.05,0.05)
    with tf.variable_scope('language_model',reuse=None,initializer=initializer):
        train_model = PTBModel(True,train_batch_size,train_num_step)

    with tf.variable_scope('language_model',reuse=True,initializer=initializer):
        eval_model = PTBModel(False,eval_num_batch,eval_num_step)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        train_batches = make_batches(read_data(train_data),train_batch_size,train_num_step)
        eval_batches = make_batches(read_data(eval_data),eval_num_batch,eval_num_step)
        test_batches = make_batches(read_data(test_data),eval_num_batch,eval_num_step)
        step = 0

        for i in range(num_epoch):
            print('in iteration: %d'%(i+1))
            step,train_pplx = run_epoch(sess,train_model,train_batches,train_model.train_op,True,step)
            print('epoch:%d train perplexity:%0.3f'%(i+1,train_pplx))

            _,eval_pplx = run_epoch(sess,eval_model,eval_batches,tf.no_op(),False,0)
            print('epoch:%d eval perplexity:%0.3f' %(i+1,eval_pplx))

        _,test_pplx = run_epoch(sess,eval_model,test_batches,tf.no_op,False,0)
        print("test perplexity: %.3f" % test_pplx)

if __name__ == '__main__':
    main()































