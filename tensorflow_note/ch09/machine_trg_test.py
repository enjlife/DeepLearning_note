import tensorflow as tf
import sys, os
sys.path.append(os.pardir)
import numpy as np
import codecs
import collections
from operator import itemgetter
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""
1.使用tf.while_loop

"""
dir_name = os.path.dirname(__file__)
checkpoint_path = dir_name+'/seq2seq_ckpt-0'
hidden_size = 1024
num_layers = 2
src_vocab_size = 10000
trg_vocab_size = 4000
share_emb_and_softmax = True

sos_id = 1
eos_id = 2

class NMTModel(object):
    def __init__(self):
        self.enc_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(hidden_size) for _ in range(num_layers)])
        self.dec_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(hidden_size) for _ in range(num_layers)])

        self.src_embedding = tf.get_variable('src_emb',[src_vocab_size,hidden_size])  # 定义词向量
        self.trg_embedding = tf.get_variable('trg_emb',[trg_vocab_size,hidden_size])

        if share_emb_and_softmax:
            self.softmax_weight = tf.transpose(self.trg_embedding)

        else:
            self.softmax_weight = tf.get_variable('weight',[hidden_size,trg_vocab_size])
        self.softmax_bias = tf.get_variable('softmax_bias',[trg_vocab_size])
    def inference(self,src_input):
        src_size = tf.convert_to_tensor([len(src_input)],dtype=tf.int32)
        src_input = tf.convert_to_tensor([src_input],dtype=tf.int32)
        src_emb = tf.nn.embedding_lookup(self.src_embedding,src_input)

        with tf.variable_scope('encoder'):
            enc_outputs,enc_state = tf.nn.dynamic_rnn(self.enc_cell,src_emb,src_size,dtype=tf.float32)

        max_dec_len = 100  # 设置解码的最大步数
        with tf.variable_scope('decoder/rnn/multi_rnn_cell'):
            # 使用一个变长的TensorArray来存储生成的句子
            # 含有init_array.read()  init.write() 传入--init.unstack(array或者tensor)
            # init.write(index, value, name=None) 指定index位置写入Tensor
            # ta.unstack(value, name=None) 可以看做是stack的反操作，输入Tensor，输出一个新的TensorArray对象
            # init.stack(name=None) 将TensorArray中元素叠起来当做一个Tensor输出
            init_array = tf.TensorArray(dtype=tf.int32,size=0,dynamic_size=True,clear_after_read=False)
            init_array = init_array.write(0,sos_id)
            # 构建初始的循环状态--rnn的隐藏状态，保存生成句子的tensorarray，记录解码步数的整数step
            init_loop_var = (enc_state,init_array,0)

            def continue_loop_condition(state,trg_ids,step):
                # 循环直到解码器输出<eox>，或者达到最大步数
                # 计算一个张量在维度上元素的“逻辑和”，如果有axis设定轴，则在设定的维度上计算逻辑和，否则在所有维度上计算逻辑和--对应numpy的all
                # 输出不等于<eos> 和 step < max_dec_len-1 ，有一个为False则退出
                return tf.reduce_all(tf.logical_and(tf.not_equal(trg_ids.read(step),eos_id),tf.less(step,max_dec_len-1)))

            def loop_body(state,trg_ids,step):
                trg_input = [trg_ids.read(step)] # 读取第step个值
                trg_emb = tf.nn.embedding_lookup(self.trg_embedding,trg_input)  # 用词向量方式表示
                # 这里不使用dynamic_rnn，而是直接调用dec_cell向前计算一步，因为测试时每一步的输出需要作为下一步的输入
                dec_outputs,next_state = self.dec_cell.call(state=state,inputs=trg_emb)
                output = tf.reshape(dec_outputs,[-1,hidden_size])  # 这里只有一个batch_size为1 num_step也只有一个

                logits = (tf.matmul(output,self.softmax_weight)+self.softmax_bias)
                # 选取logit最大的值作为输出
                next_id = tf.argmax(logits,axis=1,output_type=tf.int32)
                trg_ids = trg_ids.write(step+1,next_id[0])
                return next_state,trg_ids,step+1

            state,trg_ids,step = tf.while_loop(continue_loop_condition,loop_body,init_loop_var)
            return trg_ids.stack()

def main():
    with tf.variable_scope('nmt_model',reuse=None):
        model = NMTModel()
        test_sentence = [90,13,0,689,4,2]

        output_op = model.inference(test_sentence)
        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess,checkpoint_path)

        output = sess.run(output_op)
        print(output)
        sess.close()

if __name__=='__main__':
    main()
