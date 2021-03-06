#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
"""
from Config import BilstmCrfConfig as config
from collections import Counter
import os
import codecs
import tensorflow as tf
import numpy as np
from utils import *

def build_vocab(train_data,test_data,vocab_path):
    train_wf = open(train_data)
    test_wf = open(test_data)
    vocab_train = [line.split()[0].lower() for line in train_wf if len(line.strip())!=0]
    vocab_test = [line.split()[0].lower() for line in test_wf if len(line.strip())!=0]
    train_wf.close()
    test_wf.close()
    vocab = vocab_train+vocab_test
    word2count = Counter(vocab)
    if not os.path.exists("Vocab"):os.mkdir("Vocab")
    with codecs.open(vocab_path,'w',encoding='utf-8') as wf:
        wf.write("{}\t100000\n{}\t100000\n{}\t100000\n{}\t100000\n".
                 format("<PAD>","<START>","<END>","<UNK>"))
        for vocab,count in word2count.most_common(len(word2count)):
            wf.write("{}\t{}\n".format(vocab,count))

#load label and vocab.
def load_vocab():
    vocab = [line.split('\t')[0] for line in open(config.vocab_path)
             if int(line.strip().split('\t')[1])>=config.min_vocab]
    label = [line.strip().split('\t')[0] for line in open('data/label_id.txt')]
    idx2word = {idx:word for idx,word in enumerate(vocab)}
    word2idx = {word:idx for idx,word in enumerate(vocab)}
    idx2labl = {idx:label for idx,label in enumerate(label)}
    labl2idx = {label:idx for idx,label in enumerate(label)}
    return idx2word,word2idx,idx2labl,labl2idx

def create_data(sentence_words,sentence_tags):
    idx2word, word2idx, idx2labl, labl2idx = load_vocab()
    embedding_trim(idx2word, config.trimed_path)
    widx_list,lidx_list,sentence_word,sentence_tag=[],[],[],[]
    for sw,st in zip(sentence_words,sentence_tags):
        widx = [word2idx.get(word,3) for word in [u"<START>"]+sw+[u"<END>"]]
        lidx = [labl2idx.get(label,3) for label in [u"<START>"]+st+[u"<END>"]]
        assert len(widx)==len(lidx)
        widx_list.append(np.array(widx))
        lidx_list.append(np.array(lidx))
        sentence_word.append(sw)
        sentence_tag.append(st)
    X = np.zeros([len(widx_list),config.maxlen],np.int32)
    Y = np.zeros([len(lidx_list),config.maxlen],np.int32)
    seq_len = []
    for i,(x,y) in enumerate(zip(widx_list,lidx_list)):

        if len(x)<=config.maxlen:
            seq_len.append(len(x))
            X[i] = np.lib.pad(x,[0,config.maxlen-len(x)],'constant',constant_values=(0,0))
            Y[i] = np.lib.pad(y,[0,config.maxlen-len(y)],'constant',constant_values=(0,0))
        else:
            seq_len.append(config.maxlen)
            X[i] = x[:config.maxlen]
            Y[i] = y[:config.maxlen]
    return X,Y,sentence_word,sentence_tag,seq_len

#data: train_data or test_data
def build_sentences(data,usemaxlenght=False):
    sentence_words = []
    sentence_tags = []
    words = []
    tags = []
    rf = open(data)
    lines = rf.readlines()
    for i,line in enumerate(lines):
        contents = line.strip().split()
        if len(line.strip())!=0:
            words.append(contents[0].lower())
            tags.append(contents[-1])
            if contents[0] in [".","!","?"] and len(lines[i+1].strip())==0:
                if usemaxlenght:
                    if len(words)>config.maxlen:
                        config.maxlen=len(words)
                else:
                    sentence_words.append(words)
                    sentence_tags.append(tags)
                    words=[]
                    tags=[]
    rf.close()
    #[allsentence,max_len]
    return sentence_words,sentence_tags

def load_traindata():
    sentence_words, sentence_tags = build_sentences(config.train_data)
    X, Y, _,_,seq_len = create_data(sentence_words, sentence_tags)
    return X,Y,seq_len

def load_testdata():
    sentence_words, sentence_tags = build_sentences(config.test_data)
    X, Y, sentence_word, sentence_tag,seq_len = create_data(sentence_words, sentence_tags)
    return X,Y,sentence_word,sentence_tag, seq_len

def get_batch_data(train=True):
    build_vocab(config.train_data, config.test_data, config.vocab_path)
    if train:
        X, Y, seq_len = load_traindata()
    else:
        X, Y, sentence_word, sentence_tag, seq_len = load_testdata()
    num_batch = len(X)//config.batch_size
    X = tf.convert_to_tensor(X,tf.int32)
    Y = tf.convert_to_tensor(Y,tf.int32)
    data_length = {
        "X":X,
        "Y":Y,
        "seq_len":seq_len
    }

    dataset = tf.data.Dataset.from_tensor_slices(data_length).\
        batch(config.batch_size).shuffle(buffer_size=1000).repeat(config.EPOCH)
    iterator = dataset.make_one_shot_iterator()
    # iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    return next_element,iterator,num_batch

if __name__=="__main__":
    """
    Test:this part was used to test functions in this scripts!
    Please try it and if it dont have problems stop it in time!
    """
    sess = tf.Session()
    next_element ,iterator, num_batch= get_batch_data(train=True)
    # sess.run(iterator.initializer)

    try:
        while True:
            data = sess.run(next_element)
            length = len(data["seq_len"])

            print(len(data["seq_len"]))

    except tf.errors.OutOfRangeError:
        print("Train finished")

