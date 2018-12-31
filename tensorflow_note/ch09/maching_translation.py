import tensorflow as tf
import sys, os
sys.path.append(os.pardir)
import numpy as np
import codecs
import collections
from operator import itemgetter
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

raw_data_zh = '/Users/enjlife/Downloads/en-zh/train.tags.en-zh.zh'
raw_data_en = '/Users/enjlife/Downloads/en-zh/train.tags.en-zh.en'
vocab_out_zh = 'ted_zh.vocab'
vocab_out_en = 'ted_en.vocab'
output_en = 'ted.src'
output_zh = 'ted.trg'





counter = collections.Counter()
with codecs.open(raw_data_en,'r','utf-8') as f:
    for line in f:
        for word in line.strip().split():
            counter[word] +=1

sorted_word_to_cnt = sorted(counter.items(),key=itemgetter(1),reverse=True)
sorted_words = [x[0] for x in sorted_word_to_cnt]
#sorted_words = sorted_words+['<eos>']  # 此处添加会影响列表的顺序
sorted_words =['<unk>']+['<sos>']+['<eos>']+ sorted_words
if len(sorted_words)>10000:
    sorted_words = sorted_words[:10000]
with codecs.open(vocab_out_en,'w','utf-8') as file_output:
    for word in sorted_words:
        file_output.write(word+'\n')

#将词汇文件转化为单词编号

# #读取词汇表，并建立词汇到单词的映射
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