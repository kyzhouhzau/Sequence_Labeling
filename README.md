# Sequence_Labeling
Try to finish, Bilstm+CRF, Bert+CRF, model


First Model： Bilstm CRF
Optional parameter：
+ sinusoid (For position embedding.True means you could use sinusoid to initial position embedding.)
+ use_pretrain (True means use golve or word2vec to initial tokens embedding.)
+ cell ("lstm" or "gru")
+ use_crf (True use crf as decode layer)
+ normalize (Use batch normaliz)
+ others (see Config.py for more information!)

Tricks:
label smoothing


Using:
+ Build dir first (logdir,vocab,Embedding)
+ download glove or word2vec (here i use glove200d)
+ python traing.py
+ python eval.py
