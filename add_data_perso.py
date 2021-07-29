from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np



def add_data_p(t_size,v_size,binary_size):
    num_max=2**binary_size
    add_max=2**(binary_size+1)
    op=[]
    seed=[]
    res=[]
    for i in range(num_max):
        for j in range(i+1):
            if i+j<=add_max:
                op.append([i,j])
                seed.append(1)
                res.append([i+j])
    
    indices = np.arange(t_size+v_size)
    np.random.shuffle(indices)
    op = np.array(op)[indices]
    seed = np.array(seed)[indices]
    res=np.array(res)[indices]
    
    op_t=op[:t_size]
    seed_t=seed[:t_size]
    res_t=res[:t_size]
    
    op_v=op[t_size:]
    seed_v=seed[t_size:]
    res_v=res[t_size:]
    
    train_dataset=tf.data.Dataset.from_tensor_slices((op_t,seed_t,res_t))
    val_dataset=tf.data.Dataset.from_tensor_slices((op_v,seed_v,res_v))
    
    return train_dataset,val_dataset
    


