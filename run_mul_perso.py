from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import sys
import numpy as np
sys.path.append('./data/')
sys.path.append('./model/')
sys.path.append('./')
from mul_data_perso import mul_data_p
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
T_SIZE=10000
V_SIZE=1000
BATCH_SIZE=32
BINARY_SIZE=10
EPOCHS=300

##Model Constants
d_model = 16 # dimension de l'embedding. (without emmbeding the dimension is 16)
num_layers = 6
dff = 128 ## Dim de la couche du feed forward

target_vocab_size = 2*BINARY_SIZE  ## taille de la sortie
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
pos = 3






#Creation Dataset
t_ds,v_ds=mul_data_p(T_SIZE,V_SIZE,BINARY_SIZE)

t_ds=t_ds.batch(BATCH_SIZE)
v_ds=v_ds.batch(BATCH_SIZE)


#Creating the model


transformer_1 = Transformer(num_layers, d_model, 2*BINARY_SIZE, dff, pos, target_vocab_size, make_sym, USE_positioning_1,
                            out_num_1, out_pos_1, res_ratio, dropout_rate)
learning_rate = CustomSchedule(d_model)
optimizer_1 = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                epsilon=1e-9)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Accuracy(name='train_accuracy')
d_loss = tf.keras.metrics.Mean(name='d_loss')
d_accuracy = tf.keras.metrics.Accuracy(name='d_accuracy')

def create_masks_1(seed, seq_1):
    state_size = seq_1
    mask_source = tf.one_hot(seed, seq_1*2)[:, tf.newaxis, :]
    change_mask = tf.concat([tf.eye(state_size), tf.zeros((state_size, state_size))], -1)
    enc_padding_mask = tf.maximum(change_mask[tf.newaxis, :, :], mask_source)
    combined_mask = None
    enc_padding_mask = 1-enc_padding_mask
    enc_padding_mask = enc_padding_mask[:,:,tf.newaxis,:]
    dec_padding_mask = enc_padding_mask
    return enc_padding_mask, combined_mask, dec_padding_mask
@tf.function
def train_step_1(inp, seed, tar, seq_1, binary_size):
    batch_size = seed.shape[0]
    state_size = seq_1
    enc_inp = tf.tile(inp[:, tf.newaxis, :],[1, state_size, 1])
    dec_inp = tf.ones((batch_size, state_size, 1), dtype=tf.int64)*start_token_dec
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks_1(seed, seq_1)
    with tf.GradientTape() as tape:
        predictions, _ = transformer_1(enc_inp, dec_inp, True, enc_padding_mask, combined_mask, dec_padding_mask)
        pred = tf.squeeze(predictions, -2)
        loss = loss_function(binary_encoding(tar, binary_size), pred)
    gradients = tape.gradient(loss, transformer_1.trainable_variables)
    optimizer_1.apply_gradients(zip(gradients, transformer_1.trainable_variables))
    train_loss(loss)
    tf.summary.scalar("loss", train_loss.result(), step=optimizer_1.iterations)
    pred_binary = tf.cast(tf.greater(pred, 0), tf.int64)
    train_accuracy(tar, back2int(pred_binary))
    
    
    
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

for epoch in range(EPOCHS):
    start = time.time()
    train_loss.reset_states()
    train_accuracy.reset_states()
    #with summary_writer_1.as_default():
    for (batch, (inp, seed, tar_add)) in enumerate(t_ds):
        train_step_1(inp, seed, tar_add, seq_len_1, 2*BINARY_SIZE)
        
        if batch % 500 == 0:
            print('Epoch {} Batch {}:\nTraining_loss {:.4f} Training_Accuracy {:.4f}'.format(
                epoch + 1, batch, train_loss.result(), train_accuracy.result()))
            d_acc = eval_val_1(v_ds, seq_len_1, 2*BINARY_SIZE)
            tf.summary.scalar("val_acc", d_acc, step=optimizer_1.iterations)
    """ if (epoch+1) % 5 == 0:
        ckpt_save_path = ckpt_manager_1.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))"""
    print('Epoch {}:\nTraining_Loss {:.4f} Training_Accuracy {:.4f}'.format(epoch + 1,
                                                train_loss.result(),
                                                train_accuracy.result()))
    d_acc = eval_val_1(v_ds, seq_len_1, 2*BINARY_SIZE)
    tf.summary.scalar('val_acc', d_acc, step=optimizer_1.iterations)
    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
        