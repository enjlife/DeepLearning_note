import tensorflow as tf
import sys, os
sys.path.append(os.pardir)
import numpy as np
import codecs
import collections
from operator import itemgetter
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""
1.在seq2seq的基础上修改添加attention算法
2.修改内容为-编码器的神经网络 编码器的构建 解码器的构建



"""

dir_name = os.path.dirname(__file__)
src_data = 'train.en'
trg_data = 'train.zh'
checkpoint_path = dir_name+'/attention_ckpt'

hidden_size = 1024
num_layers = 2
src_vocab_size = 10000
trg_vocab_size = 4000
batch_size = 100
num_epoch = 5
keep_prob = 0.8
max_grad_norm = 5
share_emb_and_softmax = True

max_len = 50
sos_id = 1

# 数据预处理
def MakeDataset(file_path):
    dataset = tf.data.TextLineDataset(file_path)  # 通过dataset读取的数据可以使用map <TextLineDataset shapes: (), types: tf.string>
    #<MapDataset shapes: (?,), types: tf.string>
    dataset = dataset.map(lambda string:tf.string_split([string]).values)  # st.values： ['hello', 'world', 'a', 'b', 'c'] st.indices
    #< MapDatasetshapes: (?,), types: tf.int32 >
    dataset = dataset.map(lambda string:tf.string_to_number(string,tf.int32))
    #<MapDataset shapes: ((?,), ()), types: (tf.int32, tf.int32)>
    dataset = dataset.map(lambda x: (x,tf.size(x)))  #tuple
    return dataset

def MakeSrcTrgDataset(src_path,trg_path,batch_size):
    src_data = MakeDataset(src_path)
    trg_data = MakeDataset(trg_path)
    dataset = tf.data.Dataset.zip((src_data,trg_data))

    def FilterLength(src_tuple,trg_tuple):  # 删除内容为空和长度过长
        ((src_input,src_len),(trg_label,trg_len)) = (src_tuple,trg_tuple)
        src_len_ok = tf.logical_and(tf.greater(src_len,1),tf.less_equal(src_len,max_len))  #比较src>1 比较src<=max_len
        trg_len_ok = tf.logical_and(tf.greater(trg_len,1),tf.less_equal(trg_len,max_len))
        return tf.logical_and(src_len_ok,trg_len_ok)
    dataset = dataset.filter(FilterLength)

    def MakeTrgInput(src_tuple,trg_tuple):
        ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
        trg_input = tf.concat([[sos_id],trg_label[:-1]],axis=0)  #在trg_label最后连接，输入以sos为前提，预测需以eos为结尾
        return (src_input,src_len),(trg_input,trg_label,trg_len)
    dataset = dataset.map(MakeTrgInput)
    dataset = dataset.shuffle(10000)  #buffer size
    # 规定填充后输出的数据维度  None表示长度未知  []表示单个数字
    padded_shapes = ((tf.TensorShape([None]),tf.TensorShape([])),
                     (tf.TensorShape([None]),tf.TensorShape([None]),tf.TensorShape([])))
    batched_dataset = dataset.padded_batch(batch_size,padded_shapes)
    return batched_dataset

class NMTModel(object):
    def __init__(self):
        #self.enc_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(hidden_size) for _ in range(num_layers)])
        self.enc_cell_fw = tf.nn.rnn_cell.LSTMCell(hidden_size)
        self.enc_cell_bw = tf.nn.rnn_cell.LSTMCell(hidden_size)

        self.dec_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(hidden_size) for _ in range(num_layers)])

        self.src_embedding = tf.get_variable('src_emb',[src_vocab_size,hidden_size])  # 定义词向量
        self.trg_embedding = tf.get_variable('trg_emb',[trg_vocab_size,hidden_size])

        if share_emb_and_softmax:
            self.softmax_weight = tf.transpose(self.trg_embedding)

        else:
            self.softmax_weight = tf.get_variable('weight',[hidden_size,trg_vocab_size])
        self.softmax_bias = tf.get_variable('softmax_bias',[trg_vocab_size])

    def forward(self,src_input,src_size,trg_input,trg_label,trg_size):
        batch_size = tf.shape(src_input)[0]  # 为何这里有batch_size，默认已经定义，计算loss的最后一组batch可能会变化,所以重新取值
        src_emb = tf.nn.embedding_lookup(self.src_embedding,src_input)
        trg_emb = tf.nn.embedding_lookup(self.trg_embedding,trg_input)

        src_emb = tf.nn.dropout(src_emb,keep_prob)  # 对输入进行dropout
        trg_emb = tf.nn.dropout(trg_emb,keep_prob)

        with tf.variable_scope('encoder'):  # 构建编码器
            #enc_outputs,enc_state = tf.nn.dynamic_rnn(self.enc_cell,src_emb,src_size,dtype=tf.float32)
            enc_outputs,enc_state = tf.nn.bidirectional_dynamic_rnn(self.enc_cell_fw,self.enc_cell_bw,src_emb,src_size)
            enc_outputs = tf.concat([enc_outputs[0],enc_outputs[1]],-1)  # 拼接

        with tf.variable_scope('decoder'):  # 构建解码器
            #dec_outputs,dec_state = tf.nn.dynamic_rnn(self.dec_cell,trg_emb,trg_size,initial_state=enc_state)
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(hidden_size,enc_outputs,memory_sequence_length=src_size)
            # 将解码器和注意力一起封装成更高层的循环神经网络
            attention_cell = tf.contrib.seq2seq.AttentionWrapper(self.dec_cell,attention_mechanism,attention_layer_size=hidden_size)
            dec_outputs,_ = tf.nn.dynamic_rnn(attention_cell,trg_emb,trg_size,dtype=tf.float32)

        # 计算每一步的log perplity
        output = tf.reshape(dec_outputs,[-1,hidden_size])
        logits = tf.matmul(output,self.softmax_weight) + self.softmax_bias
        #spars输出目标idx
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(trg_label,[-1]),logits=logits)  # 将trg_label reshape为一维数组
        #loss是与trg_label维度相同的一维矩阵

        # 计算平均损失，将填充位置权重设为零
        # trg_size是个数的数组，获取整个dataset的第二个维度，填充1或者0
        label_weights = tf.sequence_mask(trg_size,maxlen=tf.shape(trg_label)[1],dtype=tf.float32)
        label_weights = tf.reshape(label_weights,[-1])  # 变为一维数组
        cost = tf.reduce_sum(loss*label_weights)  # 计算总的loss [1*n]*[n*maxlen]
        cost_per_token = cost / tf.reduce_sum(label_weights)  #计算每一个权重的平均loss

        # 反向传播
        trainble_variables = tf.trainable_variables()
        grads = tf.gradients(cost / tf.to_float(batch_size),trainble_variables)  # 计算的cost为一个batch的损失，除batch_size获取每一步的平均损失
        grads,_ = tf.clip_by_global_norm(grads,max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        train_op = optimizer.apply_gradients(zip(grads,trainble_variables))
        return cost_per_token,train_op

def run_epoch(sess,cost_op,train_op,saver,step):
    while True:
    # 重复训练直到遍历完dataset所有数据
        try:
            cost,_ = sess.run([cost_op,train_op])
            if step % 10 ==0:
                print('after %d steps,per token cost is %.3f' %(step,cost))
            if step % 200 ==0:
                saver.save(sess,checkpoint_path,global_step=step)
            step +=1
        except tf.errors.OutOfRangeError:
            break
    return step

def main():
    initializer = tf.random_uniform_initializer(-0.05,0.05)
    with tf.variable_scope('nmt_model',reuse=None,initializer=initializer):
        train_model = NMTModel()

    data = MakeSrcTrgDataset(src_data,trg_data,batch_size)
    iterator = data.make_initializable_iterator()
    # get_next 获得一个epoch的数据，但是每次sess.run的时候只能读取一个batch的数据，相当于使用placeholder每次喂一个batch的数据。
    (src,src_size),(trg_input,trg_label,trg_size) = iterator.get_next()
    cost_op,train_op = train_model.forward(src,src_size,trg_input,trg_label,trg_size)

    saver = tf.train.Saver()
    step = 0
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(num_epoch):
            print('in iteration:%d' %(i+1))
            sess.run(iterator.initializer)  # 重新获取一个epoch的数据
            step = run_epoch(sess,cost_op,train_op,saver,step)

if __name__=='__main__':
    main()