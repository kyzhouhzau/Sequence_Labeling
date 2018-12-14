#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
"""

from Bilstm_crf import *
from utils import eval_result
from data_load import get_batch_data,load_vocab

def train():
    model_graph = Model_Graph(True)
    graph,train_op, losses, accs, predicts, true_labels,global_step = model_graph.Graph()
    sv = tf.train.Supervisor(
        graph=graph,
        logdir=config.logdir,
        save_model_secs=100,
        checkpoint_basename='./model/model.ckpt',
        global_step=global_step,
        summary_writer=tf.summary.FileWriter(r'./logdir/summary/')
    )
    with sv.managed_session() as sess:
        counter=0
        try:
            while True:
                if sv.should_stop(): break
                counter+=1
                sess.run(train_op)
                # sess.run(g.global_step)
                true_label, predict, acc, loss = sess.run([true_labels,predicts,accs,losses])
                if counter%10==0:
                    eval_result(true_label,predict,acc,loss,counter)
        except tf.errors.OutOfRangeError:
            print("Train finished")

if __name__=="__main__":
    train()














