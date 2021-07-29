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
from merge_sort_data import merge_data,ins_data

from model import *
from utils import *

"""
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
"""


## Dataset Constant
T_SIZE=100
V_SIZE=20
BATCH_SIZE=32
BINARY_SIZE=8
EPOCHS=10000
LENGTH=3

##Model Constants
d_model = 8 # dimension de l'embedding. (without emmbeding the dimension is 16)
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


out_num = True
out_pos = True
assert(out_num or out_pos)
USE_positioning = False
pos = 3
discount = 0.005

checkpoint_path_data="."
checkpoint_path_msk="."



#Creation Dataset
t_ds,v_ds=ins_data(T_SIZE,V_SIZE,LENGTH,BINARY_SIZE)
#t_ds,v_ds=merge_data(T_SIZE,V_SIZE,LENGTH,BINARY_SIZE)

t_ds=t_ds.batch(BATCH_SIZE)
v_ds=v_ds.batch(BATCH_SIZE)


#Creating the model


transformer = Transformer(num_layers, d_model, BINARY_SIZE, dff, pos,
                          target_vocab_size, make_sym, USE_positioning,
                            out_num, out_pos, res_ratio, dropout_rate)
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


if ckpt_manager_data.latest_checkpoint:
    ckpt_data.restore(ckpt_manager_data.latest_checkpoint)
    print ('Model_data checkpoint restored!!')

if ckpt_manager_msk.latest_checkpoint:
    ckpt_msk.restore(ckpt_manager_msk.latest_checkpoint)
    print ('Model_msk checkpoint restored!!')

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

def create_masks_ins(mask,length):
    n=mask.shape[0]
    m=mask.shape[1]
    res=np.zeros((n,m,length),dtype=np.int64)
    for i in range(n):
        for j in range(m):
            res[i,j,mask[i][j][1]]=1
            res[i,j,mask[i][j][0]]=1
    combined_mask = None
    enc_padding_mask = 1-res
    enc_padding_mask = enc_padding_mask[:,:,tf.newaxis,:]
    dec_padding_mask = enc_padding_mask
    return enc_padding_mask, combined_mask, dec_padding_mask


def diag(att_weight):
    batch=att_weight.shape[0]
    n_loop=att_weight.shape[1]
    ls=att_weight.shape[2]
    res=np.zeros((batch,n_loop,ls))
    for i in range(ls):
        res[:,:,i]=att_weight[:,:,i,i]
    return tf.convert_to_tensor(res,dtype=tf.float32)
    
        
    
def eval_data(dataset,length,binary_size):
    for (batch,(inp,mask,tar)) in enumerate(dataset):
        batch_size = mask.shape[0]
        n_loop=mask.shape[1]# 2*length +1
        state_size = inp.shape[2] ## 2*length+2
        ##inp.shape = (batch_size,2*length+1,2*length+2)
        ## mask.shape =(batch_size,2*length+1,2)
        ## tar.shape = (batch_size,2*length+2)        
        dec_inp = tf.zeros((batch_size, n_loop-1,state_size), dtype=tf.int64)
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(mask,length)
        com_mask=combined_mask
        enc_inp=inp[:,0,:]
        enc_mask=enc_padding_mask[:,0,:,:]
        res=[]
        mask_res=[]
        for i in range(n_loop):
            enc_inp=tf.tile(enc_inp[:,tf.newaxis,:],[1,n_loop-1,1])
            
            enc_mask=tf.tile(enc_mask[:,tf.newaxis,:,:],[1,n_loop-1,1,1])
            enc_mask=tf.cast(enc_mask,tf.float32)
            dec_mask=enc_mask
            print(enc_inp.shape,enc_mask.shape)
            predictions,_,predicted_pos = transformer(enc_inp, dec_inp, False, enc_mask, com_mask, dec_mask)
            
            #print(enc_inp.shape,enc_mask.shape,predictions.shape,predicted_pos.shape)
            #print(back2int(tf.cast(tf.greater(predictions[0],0),tf.int64)))
            predictions = tf.squeeze(predictions)
            print(predictions.shape)
            mean=tf.math.reduce_mean(predictions,axis=1) # mean of the same line.
            print(mean.shape)
            enc_mask=tf.squeeze(enc_mask)
            pointer=diag(predicted_pos)
            enc_mask=tf.reshape(enc_mask,(enc_mask.shape[0],enc_mask.shape[1],enc_mask.shape[2],1))
            pointer=tf.reshape(pointer,(pointer.shape[0],pointer.shape[1],pointer.shape[2],1))
            #print(mask,pointer)
            mask_concat=tf.concat([enc_mask,pointer],axis=-1)
            mask_concat = tf.reshape(mask_concat,[-1, mask_concat.shape[-2], mask_concat.shape[-1]])
            mask_concat = 2*mask_concat-1
            #print(mask_concat.shape)
            predict_msk = msk_transform(mask_concat,False)#tf.reshape(mask[:,i,:],(mask.shape[0],mask.shape[-1],1)), True)
            #print(predict_msk.shape)
            predict_msk = tf.reshape(predict_msk, (enc_mask.shape[0], enc_mask.shape[-3], enc_mask.shape[-2]))
            #print(predict_msk.shape)
            
            enc_inp=tf.cast(tf.greater(mean,0),tf.int64)
            enc_inp=back2int(enc_inp)
            print(enc_inp.shape)
            res.append(enc_inp)
            
            enc_mask=tf.reshape(predict_msk,[predict_msk.shape[0],predict_msk.shape[1],1,predict_msk.shape[2]])
            enc_mask=tf.math.reduce_mean(enc_mask,axis=1)
            mask_res.append(tf.squeeze(enc_mask))
            loss=loss_function(binary_encoding(inp[:,i+1,:],binary_size),mean)
            d_loss(loss)
            d_accuracy(inp[:,-1,:],res[-1])
            print("Step {} : Loss content : {}, Accuracy :{}".format(i,d_loss.result(),d_accuracy.result()))
            
            
#@tf.function
def train_data(dataset,length,binary_size):
    for (batch,(inp,mask,tar)) in enumerate(dataset):
    
        batch_size = mask.shape[0]
        n_loop=mask.shape[1]# 2*length +1
        state_size = inp.shape[2] ## 2*length+2
        ##inp.shape = (batch_size,2*length+1,2*length+2)
        ## mask.shape =(batch_size,2*length+1,2)
        ## tar.shape = (batch_size,2*length+2)        
        dec_inp = tf.zeros((batch_size, n_loop,state_size), dtype=tf.int64)
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks_ins(mask,length)
        #enc_padding_mask, combined_mask, dec_padding_mask = create_masks(mask,length)
        
        #for i in range(n_loop-1):
        pred,pred_pos=train_data_loop(0,inp,dec_inp,enc_padding_mask,combined_mask,dec_padding_mask,dec_inp,length,binary_size)
        #pointer=pred_pos[list(pred_pos)[-1]]
        train_mask_loop(0,enc_padding_mask,pred_pos)
        if batch % 500 == 0:
            print('Epoch {} Batch {}:\nTraining_loss {:.4f} Training_Accuracy {:.4f}'.format(
                epoch + 1, batch, train_data_loss.result(), train_data_accuracy.result()))
            pred_binary = tf.cast(tf.greater(pred, 0), tf.int64)
            pred_binary = back2int(pred_binary)
        return pred_binary

def train_data_loop(i,inp,dec,enc_mask,com_mask,dec_mask,dec_inp,length,binary_size):
    #enc_inp = tf.tile(inp[:,i, tf.newaxis, :], [1, state_size, 1])
    enc_inp=inp[:,:-1,:]
    #print(enc_inp[0])
    #dec_inp=inp[:,:-1,:]
    #print(dec_inp[0])
    #dec_inp = tf.tile(inp[:,i+1, tf.newaxis, :], [1, state_size, 1])
    #enc_mask= tf.tile(enc_mask[:,i, tf.newaxis,:, :], [1, state_size,1, 1])
    
    
    mask=enc_mask[:,:,:,:]
    #mask=enc_mask[:,:-1,:,:]
    mask=tf.cast(mask,tf.float32)
    #dec_mask= tf.tile(dec_mask[:,i, tf.newaxis,:,:], [1, state_size,1, 1])
    dec_mask=dec_mask[:,:,:,:]
    #dec_mask=dec_mask[:,:-1,:,:]
    dec_mask=tf.cast(dec_mask,tf.float32)
    ##chg_mask = tf.one_hot(pos, seq_1+seq_2+2)
    with tf.GradientTape() as tape:
        
        predictions,_,predicted_pos = transformer(enc_inp, dec, True, mask, com_mask, dec_mask)
        #print(enc_inp.shape,enc_mask.shape,predictions.shape,predicted_pos.shape)
        #print(back2int(tf.cast(tf.greater(predictions[0],0),tf.int64)))
        predictions = tf.squeeze(predictions)
        # Il y a 22 lignes identiques dans prédictions (enfin normalement), on pourra peut-être choisir toujours la meilleure pour optimiser un peu.
        #weights = tf.concat([tf.ones((tar.shape[0], seq_1+seq_2)), tf.ones((tar.shape[0], 1))*discount],-1)
        loss_content = loss_function(binary_encoding(inp[:,1:,:], binary_size), predictions)#, weights)
        loss_position = loss_pos(diag(predicted_pos[:,:-1,:,:]),tf.squeeze(tf.cast(enc_mask[:,1:,:,:],tf.float32)))#, weights)
        #oss_position = loss_pos(diag(predicted_pos),tf.squeeze(tf.cast(enc_mask[:,1:,:,:],tf.float32)))#, weights)
        loss = loss_content #+ loss_position
    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer_data.apply_gradients(zip(gradients, transformer.trainable_variables))
    train_data_loss(loss)
    train_data_content_loss(loss_content)
    tf.summary.scalar("data_loss", train_data_loss.result(), step=optimizer_data.iterations)
    tf.summary.scalar("data_content_loss", train_data_content_loss.result(), step=optimizer_data.iterations)
    pred_binary = tf.cast(tf.greater(predictions, 0), tf.int64)
    pred_binary = back2int(pred_binary)
    train_data_accuracy(inp[:,1:,:], pred_binary)    
    return predictions,predicted_pos

def train_mask_loop(i,mask,att_weight):
    mask=tf.cast(mask,tf.float32)
    mask=tf.squeeze(mask)
    pointer=diag(att_weight)
    mask=tf.reshape(mask,(mask.shape[0],mask.shape[1],mask.shape[2],1))
    pointer=tf.reshape(pointer,(pointer.shape[0],pointer.shape[1],pointer.shape[2],1))
    #print(mask,pointer)
    mask_concat=tf.concat([mask,pointer],axis=-1)
    #mask_concat=tf.concat([mask[:,:-1,:],pointer],axis=-1)
    mask_concat = tf.reshape(mask_concat,[-1, mask_concat.shape[-2], mask_concat.shape[-1]])
    mask_concat = 2*mask_concat-1
    
    #mask=tf.reshape(mask,(mask.shape[0],mask.shape[-1],mask.shape[-2])) # batch_size,state_size,1
    with tf.GradientTape() as tape:
        predict_msk = msk_transform(mask_concat,True)#tf.reshape(mask[:,i,:],(mask.shape[0],mask.shape[-1],1)), True)
        #print(predict_msk.shape,mask.shape)
        #print(tf.shape(mask[:,1:,:]), tf.shape(predict_msk))
        predict_msk = tf.reshape(predict_msk, (mask.shape[0], mask.shape[-3], mask.shape[-2]))
        #predict_msk = tf.reshape(predict_msk, (mask.shape[0], mask.shape[-3]-1, mask.shape[-2]))
        loss_msk = loss_function(tf.squeeze(mask[:,1:,:]), predict_msk[:,:-1,:])   
    gradients = tape.gradient(loss_msk, msk_transform.trainable_variables)
    optimizer_msk.apply_gradients(zip(gradients, msk_transform.trainable_variables))
    train_msk_loss(loss_msk)
    tf.summary.scalar("msk_loss", train_msk_loss.result(), step=optimizer_msk.iterations)
    predict_msk_binary = tf.cast(tf.greater(predict_msk, 0), tf.float32)
    err = tf.reduce_sum(tf.squeeze(mask[:,1:,:])-predict_msk_binary[:,:-1,:], axis=-1)
    #err = tf.reduce_sum(tf.squeeze(mask[:,1:,:])-predict_msk_binary, axis=-1)
    print(predict_msk[-1],mask[-1])
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
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(mask,LENGTH)
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
## Est-ce que je rajoute l'entrée de chaque boucle en teacher forcing. 



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

    tf.summary.scalar('val_acc', d_acc, step=optimizer_1.iterations)"""