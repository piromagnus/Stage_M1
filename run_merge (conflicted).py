#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 11:06:53 2021

@author: piromagnus
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import sys
import numpy as np
sys.path.append('./data/')
sys.path.append('./model/')
sys.path.append('./')
from merge_sort_data import merge_data
from model import *
from utils import *


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)



## Dataset Constant
T_SIZE=1000
V_SIZE=100
BATCH_SIZE=32
BINARY_SIZE=8
EPOCHS=300
LENGTH=10

##Model Constants
d_model = 16 # dimension de l'embedding. (without emmbeding the dimension is 16)
num_layers = 6
dff = 128 ## Dim de la couche du feed forward

num_filters = 16
filter_size = 3

target_vocab_size = BINARY_SIZE  ## taille de la sortie
dropout_rate = 0.1
res_ratio = 1.5

end_token = 2 ** BINARY_SIZE
start_token_dec = 0
    
make_sym = True


seq_len_1 = 1
seq_len_2 = 1
seq_len = seq_len_1 + seq_len_2

state_size = seq_len_1

out_num_1 = True
out_pos_1 = False
assert(out_num_1 or out_pos_1)
USE_positioning_1 = True
pos = 23
discount = 0.005

checkpoint_path_data="."
checkpoint_path_msk="."



#Creation Dataset
t_ds,v_ds=merge_data(T_SIZE,V_SIZE,LENGTH,BINARY_SIZE)

t_ds=t_ds.batch(BATCH_SIZE)
v_ds=v_ds.batch(BATCH_SIZE)


#Creating the model


transformer = Transformer(num_layers, d_model, BINARY_SIZE, dff, pos, target_vocab_size, make_sym, USE_positioning_1,
                            out_num_1, out_pos_1, res_ratio, dropout_rate)
msk_transform = mask_transform(num_filters, filter_size, dropout_rate)
    
learning_rate_data = CustomSchedule(d_model)
learning_rate_msk = CustomSchedule(num_filters)

optimizer_data = tf.keras.optimizers.Adam(learning_rate_data, beta_1=0.9, beta_2=0.98, 
                                    epsilon=1e-9)

optimizer_msk = tf.keras.optimizers.Adam(learning_rate_msk, beta_1=0.9, beta_2=0.98, 
                                    epsilon=1e-9)

ckpt_data = tf.train.Checkpoint(transformer=transformer,
                       optimizer_data=optimizer_data)
ckpt_msk = tf.train.Checkpoint(msk_transform=msk_transform,
                       optimizer_msk=optimizer_msk)


ckpt_manager_data = tf.train.CheckpointManager(ckpt_data, checkpoint_path_data, max_to_keep=5)
ckpt_manager_msk = tf.train.CheckpointManager(ckpt_msk, checkpoint_path_msk, max_to_keep=5)


train_data_loss = tf.keras.metrics.Mean(name='train_data_loss')
train_data_content_loss = tf.keras.metrics.Mean(name='train_data_content_loss')
train_data_pos_loss = tf.keras.metrics.Mean(name='train_data_pos_loss')
train_msk_loss = tf.keras.metrics.Mean(name='train_msk_loss')
train_data_accuracy = tf.keras.metrics.Accuracy(name='train_data_accuracy')
train_msk_accuracy = tf.keras.metrics.Accuracy(name='train_msk_accuracy')
d_loss = tf.keras.metrics.Mean(name='d_loss')
d_accuracy = tf.keras.metrics.Accuracy(name='d_accuracy')

def create_masks(mask, length):
    #mask.shape = dataset,2*length+1 (number of loop in the merge), 2
    n=mask.shape[0]
    m=mask.shape[1]
    res=np.zeros((n,m,2*length+2),dtype=np.int64)
    for i in range(n):
        for j in range(m-1):
            if 2+mask[i][j][0]+mask[i][j][1] !=length+2+mask[i][j][1]:
                res[i,j,1+mask[i][j][0]+mask[i][j][1]]=1
            if length+2+mask[i][j][1]<2*length+2:
                res[i,j,length+2+mask[i][j][1]]=1
            res[i,j,j]=1
    combined_mask = None
    enc_padding_mask = 1-res
    enc_padding_mask = enc_padding_mask[:,:,tf.newaxis,:]
    dec_padding_mask = enc_padding_mask
    return enc_padding_mask, combined_mask, dec_padding_mask

    
        
    
def eval_val_1(dataset, seq_1, binary_size, name='Validation'):
    d_loss.reset_states()
    d_accuracy.reset_states()
    state_size = seq_1
    for element in dataset:
        inp, seed, tar = element
        batch_size = seed.shape[0]
        enc_inp = tf.tile(inp[:, tf.newaxis, :],[1, state_size, 1])
        dec_inp = tf.ones((batch_size, state_size, 1), dtype=tf.int64)*start_token_dec
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks_1(seed, seq_1)
        predictions, _ = transformer_1(enc_inp, dec_inp, False, enc_padding_mask, combined_mask, dec_padding_mask)
        pred = tf.squeeze(predictions, -2)
        loss = loss_function(binary_encoding(tar, binary_size), pred)
        d_loss(loss)
        pred_binary = tf.cast(tf.greater(pred, 0), tf.int64)
        d_accuracy(tar, back2int(pred_binary))
    print('{}_Loss {:.4f} {}_Accuracy {:.4f}'.format(name, d_loss.result(), name, d_accuracy.result()))
    return d_accuracy.result()

#@tf.function
def train_data(dataset,length,binary_size):
    for (batch,(inp,mask,tar)) in enumerate(dataset):
    
        batch_size = mask.shape[0]
        n_loop=mask.shape[1]# 2*length +1
        state_size = inp.shape[2] ## 2*length+2
        ##inp.shape = (batch_size,2*length+1,2*length+2)
        ## mask.shape =(batch_size,2*length+1,2)
        ## tar.shape = (batch_size,2*length+2)        
        dec_inp = tf.zeros((batch_size, 1,state_size), dtype=tf.int64)
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(mask,length)
        
        for i in range(n_loop-1):
            pred,pred_pos=train_data_loop(i,inp,enc_padding_mask,combined_mask,dec_padding_mask,dec_inp,length,binary_size)
            pointer=pred_pos[list(pred_pos)[-1]]
            train_mask_loop(i,enc_padding_mask,pointer)
        if batch % 500 == 0:
            print('Epoch {} Batch {}:\nTraining_loss {:.4f} Training_Accuracy {:.4f}'.format(
                epoch + 1, batch, train_data_loss.result(), train_data_accuracy.result()))
            pred_binary = tf.cast(tf.greater(pred, 0), tf.int64)
            pred_binary = back2int(pred_binary)
            print(list(pred_binary)[0])
            print(inp[0,-1,:])
        return pred_binary

def train_data_loop(i,inp,enc_mask,com_mask,dec_mask,dec_inp,length,binary_size):
    enc_inp = tf.tile(inp[:,i, tf.newaxis, :], [1, state_size, 1])
    #enc_tar = tf.tile(inp[:,i+1, tf.newaxis, :], [1, state_size, 1])
    enc_mask= tf.tile(enc_mask[:,i, tf.newaxis,:, :], [1, state_size,1, 1])
    enc_mask=tf.cast(enc_mask,tf.float32)
    dec_mask= tf.tile(dec_mask[:,i, tf.newaxis,:,:], [1, state_size,1, 1])
    dec_mask=tf.cast(dec_mask,tf.float32)
    ##chg_mask = tf.one_hot(pos, seq_1+seq_2+2)
    with tf.GradientTape() as tape:
        predictions,predicted_pos = transformer(enc_inp, dec_inp, True, enc_mask, com_mask, dec_mask)
        predictions = tf.squeeze(predictions)
        # Il y a 22 lignes identiques dans prédictions (enfin normalement), on pourra peut-être choisir toujours la meilleure pour optimiser un peu.
        #weights = tf.concat([tf.ones((tar.shape[0], seq_1+seq_2)), tf.ones((tar.shape[0], 1))*discount],-1)
        loss_content = loss_function(binary_encoding(inp[:,i+1,:], binary_size), predictions)#, weights)
        #loss_position = loss_pos(, predicted_pos)#, weights)
        loss = loss_content #+ loss_position
    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer_data.apply_gradients(zip(gradients, transformer.trainable_variables))
    train_data_loss(loss)
    train_data_content_loss(loss_content)
    tf.summary.scalar("data_loss", train_data_loss.result(), step=optimizer_data.iterations)
    tf.summary.scalar("data_content_loss", train_data_content_loss.result(), step=optimizer_data.iterations)
    pred_binary = tf.cast(tf.greater(predictions, 0), tf.int64)
    pred_binary = back2int(pred_binary)
    train_data_accuracy(inp[:,i+1,:], pred_binary)    
    return predictions,predicted_pos

def train_mask_loop(i,mask,pointer):
    mask=tf.cast(mask,tf.float32)
    #mask=tf.reshape(mask,(mask.shape[0],mask.shape[-1],mask.shape[-2])) # batch_size,state_size,1
    with tf.GradientTape() as tape:
        predict_msk = msk_transform(tf.reshape(mask[:,i,:],(mask.shape[0],mask.shape[-1],1)), True) 
        #print(predict_msk)
        #predict_msk = tf.reshape(predict_msk, (tar_msk.shape[0], tar_msk.shape[-2], tar_msk.shape[-1]))
        loss_msk = loss_function(tf.squeeze(mask[:,i+1,:]), predict_msk)    
    gradients = tape.gradient(loss_msk, msk_transform.trainable_variables)
    optimizer_msk.apply_gradients(zip(gradients, msk_transform.trainable_variables))
    train_msk_loss(loss_msk)
    tf.summary.scalar("msk_loss", train_msk_loss.result(), step=optimizer_msk.iterations)
    predict_msk_binary = tf.cast(tf.greater(predict_msk, 0), tf.float32)
    err = tf.reduce_sum(mask[:,i+1,:]-predict_msk_binary, axis=-1)
    train_msk_accuracy(err, tf.zeros_like(err))
        
        
        
        
        
"""        
for inp,mask,tar in t_ds:
    batch_size = mask.shape[0]
    n_loop=mask.shape[1]# 2*length +1
    state_size = inp.shape[2] ## 2*length+2
    ##inp.shape = (batch_size,2*length+1,2*length+2)
    ## mask.shape =(batch_size,2*length+1,2)
    ## tar.shape = (batch_size,2*length+2)        
    dec_inp = tf.zeros((batch_size, state_size,1), dtype=tf.int64)
    enc_padding_mask, combined_mask, dec_padding_mask = cre@tf.functionate_masks(mask,LENGTH)
    for i in range(mask.shape[1]-1):
        pred,pred_pos=train_data_loop(i,inp,enc_padding_mask,combined_mask,dec_padding_mask,dec_inp,LENGTH,BINARY_SIZE)
        pointer=pred_pos[list(pred_pos)[-1]]
        train_mask_loop(i,enc_padding_mask,pointer)
"""    
        
        
## Pour l'instant on train les boucles les uns après les autres mais avec 22 fois la même liste pour que les dims soit bonnes et
## dans le masque on fait aussi step by step.
## Est-ce qu'on pourrait faire en même temps toute les boucles ? (comme le batch)
## Est-ce qu'on pourrait choisir la meilleur des 22 lignes pour optimiser dans le prédicteur.
## Est-ce que je rajoute l'élément à échanger dans le masque ?
## Est



for epoch in range(EPOCHS):
    start = time.time()
    train_data_loss.reset_states()
    train_data_accuracy.reset_states()
    #with summary_writer_1.as_default()
    prediction=train_data(t_ds,LENGTH,BINARY_SIZE)
    
    """    
        if batch % 500 == 0:
            print('Epoch {} Batch {}:\nTraining_loss {:.4f} Training_Accuracy {:.4f}'.format(
                epoch + 1, batch, train_loss.result(), train_accuracy.result()))
            d_acc = eval_val_1(v_ds, seq_len_1, BINARY_SIZE)
            tf.summary.scalar("val_acc", d_acc, step=optimizer_1.iterations)
    """
    if (epoch+1) % 5 == 0:
        checkpoint_path_data=ckpt_manager_data.save()
        checkpoint_path_msk= ckpt_manager_msk.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch+1, checkpoint_path_data))
    print('Epoch {}:\nTraining_Data_Loss {:.4f} Training_Data_Accuracy {:.4f}\nTraining_Mask_Loss {:.4f} Training_Mask_Accuracy {:.4f}'.format(epoch + 1,
                                                train_data_loss.result(),
                                                train_data_accuracy.result(),
                                                train_msk_loss.result(),
                                                train_msk_accuracy.result()))
    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
    """d_acc = eval_val_1(v_ds, seq_len_1, BINARY_SIZE)
    tf.summary.scalar('val_acc', d_acc, step=optimizer_1.iterations)
    
    """