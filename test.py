from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import sys
import numpy as np
sys.path.append('./data/')
sys.path.append('./model/')
sys.path.append('./')
from sel_sort_data import sel_sort_data_gen
from addition_data import add_data
from model import *
from utils import *

reload_from_dir_2=False
binary_size = 8  #### data representation using binary_size bit binary
d_model = 16     #### size of embedding for each bit representation
filter_size = 3  #### filter size of 1D convnet
num_filters = 16  ##### number of filters

Train_SIZE = 20000 #### random sequences
BATCH_SIZE = 64  ###### batch size
Val_size = 2000  ##### number of validation example

num_layers = 6  #### number of layers in encoder/decoder
dff = 128  ##### hidden units in MLP

train_sub_size = 50 #### number of small step sequences for training data per group
val_sub_size = 10  #### number of small steps sequences for validation data per group
rep_num = 20 ###### repeatition number

target_vocab_size = binary_size + 2   #### end_token, inf
dropout_rate = 0.1  #### dropout rate
res_ratio = 1.5  #### coefficient in residual connection

inf = 2 ** binary_size  ##### number representing infinity
end_token = 2 ** (binary_size+1)  ###### number representing end token
start_token_dec = 0 ####### number representing start token

if reload_from_dir_2:
    EPOCHS_2 = 1
else:
    EPOCHS_2 = 100   ###### epochs to train
seq_len_p2 = 8
state_size_2 = seq_len_p2 + 1  ###### sequence length

out_num_2 = True  ###### whether to output number
out_pos_2 = True  ##### whether to output position
assert(out_num_2 or out_pos_2)
USE_positioning_2 = False  ##### whether using positional encoding
pos_2 = seq_len_p2 + 1
num_max_2 = 2 ** binary_size  ##### double check
make_sym = True ##### whether making the attention symmetric

def encode_t2(tr, mask, srt):
    return tr, mask, srt
def tf_encode_t2(tr, mask, srt):
    return tf.py_function(encode_t2, [tr, mask, srt], [tf.int64, tf.float32, tf.int64])

def encode_wo_mask(tr, srt):
    tr = np.hstack((tr, [end_token]))
    srt = np.hstack((srt, [end_token]))
    return tr, srt
def tf_encode_wo_mask(tr, srt):
    return tf.py_function(encode_wo_mask, [tr, srt], [tf.int64, tf.int64])





Train_dataset_2, Val_dataset_2, Texmp_size = sel_sort_data_gen(".", reload_from_dir_2,
                                                               "", num_max_2, Train_SIZE, Val_size, train_sub_size,
                                                              val_sub_size, rep_num, seq_len_p2, state_size_2, end_token, inf)


print(j)
msk_cum = tf.cumsum(j, axis=-2)
print(msk_cum)
init_msk = msk_cum[:,:-1,:,tf.newaxis] ############### init_msk is the mask used in the encoder
print(init_msk)
msk_chg = j[:,1:,:,tf.newaxis]  ######### pointer output from the modified transformer
print(msk_chg)
tar_msk = msk_cum[:,1:,:]
print(tar_msk)
######### concatenate the initial mask with the pointer as the input #####################
x = tf.concat([init_msk, msk_chg], axis=-1) ### batch_size, seq_len, seq_len, 2
print(x)
x = tf.reshape(x, (-1, x.shape[-2], x.shape[-1])) batch size * seq_len, seq_len, 2
print(x)



"""

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
binary_size = 8
d_model = 24

Train_size = 15000
BATCH_SIZE = 64
Val_size = 1500

num_layers = 6
dff = 128

target_vocab_size = binary_size + 1
dropout_rate = 0.1
res_ratio = 1.5

end_token = 2 ** binary_size
start_token_dec = 0

make_sym = True

EPOCHS_1 = 1


seq_len_1 = 1
seq_len_2 = 1
seq_len = seq_len_1 + seq_len_2

state_size = seq_len_1

out_num_1 = True
out_pos_1 = False
assert(out_num_1 or out_pos_1)
USE_positioning_1 = True
pos = 3

num_max_1 = 2 ** binary_size
Train_dataset_1, Val_dataset_1, Texmp_size = add_data(".", False, "", num_max_1, Train_size, Val_size, end_token, binary_size)

num_max_1=100
Train_size=100
Val_size=10
binary_size=8
end_token = 2 ** binary_size
inf_com = [num_max_1+1]
possible_combinations = np.concatenate([np.arange(num_max_1, 0, -2, dtype='int64'), inf_com], -1)  ##### possible_combinations[i] counts number of (i,j), j>=i
end_ind = np.cumsum(possible_combinations) ### starting indices for each group
Train_val_ind = np.random.choice(range(end_ind[-1]),(Train_size + Val_size), replace=False)
Train_ind = Train_val_ind[:Train_size]
Val_ind = Train_val_ind[Train_size:]

def from_ind_to_seq(num):
    ele_1 = np.int64(next(x for x, val in enumerate(end_ind) if val >= num+1))
    ele_2 = num_max_1-ele_1-(end_ind[ele_1]-num)
    if ele_1 == 2**(binary_size-1):
        ele_2 = num_max_1 + 1 - (end_ind[ele_1]-num)
        ele_1 = end_token
    return np.array([ele_1, ele_2])
Train_exmp = np.zeros((Train_size, 2), dtype='int64')
for i, num in enumerate(Train_ind):
    Train_exmp[i,:] = from_ind_to_seq(num)
Train_exmp = np.concatenate([Train_exmp, np.fliplr(Train_exmp)])
Train_add_out = np.sum(Train_exmp, -1)
Train_add_out[Train_add_out>end_token] = end_token
Train_seed = 1*np.ones((Train_exmp.shape[0]), dtype='int64')

"""