import pandas as pd
import numpy as np
import tensorflow as tf
from random import shuffle
import pickle as cPickle

####################################
# parameters 
####################################
user_num = 4641 
item_num = 2235
emb_dim = 64
set_length = 5 #k_sized length of a set
lr = 1e-4
decay_step = 100
decay = 0.95

sigmoid_lbda = 0.01
epochs = 100
runs = 1
batch_size = 1024
emb_init_mean = 0.
emb_init_std = 0.01
diag_init_mean = 1.
diag_init_std = 0.01
regu_weight = 0.

def get_sets(pos_set_file, neg_set_file):
    
    upos_sets = []
    with open(pos_set_file) as f:
        for l in f.readlines():
            sstr = l.strip().split(';')
            u, sets = int(sstr[0]), sstr[1:]

            for s in sets:
                a_set = []
                s1 = s.split(',')
                for id in s1:
                    a_set.append(int(id))
                if len(a_set) == set_length:
                    upos_sets.append(a_set)

    uneg_sets = []
    with open(neg_set_file) as f:
        for l in f.readlines():
            sstr = l.strip().split(';')
            u, sets = int(sstr[0]), sstr[1:]

            for s in sets:
                a_set = []
                s1 = s.split(',')
                for id in s1:
                    a_set.append(int(id))
                if len(a_set) == set_length:
                    uneg_sets.append(a_set)
    return np.array(upos_sets), np.array(uneg_sets)

def set_det(item_sets):
    subV = tf.gather(weights['V'], item_sets)
    subD = tf.linalg.diag(tf.square(tf.gather(weights['D'], item_sets)))
    K1 = tf.matmul(subV, tf.transpose(subV, perm=[0, 2, 1]))
    K = tf.add(K1, subD)
    eps = tf.eye(tf.shape(K)[1], tf.shape(K)[1], batch_shape=[tf.shape(K)[0]])
    K = tf.add(K, eps)
    res = tf.linalg.det(K)
    return res

def logsigma(itemSet):
    return tf.reduce_mean(tf.math.log(1 - tf.exp(-sigmoid_lbda * set_det(itemSet))))

def regularization(itemSet):
    itemsInBatch, _ = tf.unique(tf.reshape(itemSet, [-1]))
    subV = tf.gather(weights['V'], itemsInBatch)
    subD = tf.gather(weights['D'], itemsInBatch)
    subV_norm = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(subV), axis=1)))
    subD_norm = tf.norm(subD)
    return subV_norm + subD_norm


################################
# tf graph
################################

pset_input = tf.keras.Input(shape=(None,), dtype=tf.int32)   #item sets
nset_input = tf.keras.Input(shape=(None,), dtype=tf.int32)   #item sets

#get processed sets
pos_sets, neg_sets = get_sets('pos_item_sets_3.txt', 'neg_item_sets_3.txt')
train_size = len(pos_sets)

print(pos_sets.shape, neg_sets.shape)
for run in range(runs):
    # Store layers weight & bias
    initializer = tf.keras.initializers.GlorotNormal()
    weights = {
        'V': tf.Variable(initializer([item_num, emb_dim]), name='item_embeddings'),
        'D': tf.Variable(initializer([item_num]), name='item_bias')
    }
    # Construct model
    loss = logsigma(pset_input) + tf.math.log(1 - logsigma(nset_input)) # - regu_weight*regularization(pset_input) + regu_weight*regularization(nset_input)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.01, beta_2=0.01)
    train_op = optimizer.minimize(-loss)

    # Initializing the variables
    init = tf.compat.v1.global_variables_initializer()

    print("start training...")
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        # Training cycle
        for epoch in range(epochs):
            ave_cost = 0.
            nbatch = 0
            while True:
                if nbatch*batch_size <= train_size:
                    pos_batch = pos_sets[nbatch*batch_size: (nbatch+1)*batch_size]
                    neg_batch = neg_sets[nbatch*batch_size: (nbatch+1)*batch_size]
                else:
                    if train_size - (nbatch-1)*batch_size > 0:
                        pos_batch = pos_sets[(nbatch-1)*batch_size: train_size]
                        neg_batch = neg_sets[(nbatch-1)*batch_size: train_size]
                    break
                nbatch += 1

                _, c = sess.run([train_op, loss], feed_dict={pset_input: pos_batch, nset_input: neg_batch})
                ave_cost += c / nbatch

        param = sess.run(weights)
        cPickle.dump(param, open('item_kernel_3.pkl', 'wb')) #T=3
