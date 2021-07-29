from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import sys
sys.path.append("../")
from utils import *
                

def merge(l1,l2,res,interm,sort):
    if l1==[]:
        return l2
    if l2==[]:
        return l1
    if l1[0]<=l2[0]:
        res.append([res[-1][0]+1,res[-1][1]])
        interm.append([list(l1[1:]),list(l2)])
        sort.append(list(sort[-1]+[l1[0]]))
        return [l1[0]]+merge(l1[1:],l2,res,interm,sort)
    else:
        res.append([res[-1][0],res[-1][1]+1])
        interm.append([list(l1),list(l2[1:])])
        sort.append(list(sort[-1]+[l2[0]]))
        return [l2[0]]+merge(l1,l2[1:],res,interm,sort)


def correct_inter(mask,interm,sort,length):
    if mask[-1][0]==length:
        init=mask[-1][1]
        nb=length-mask[-1][1]-1
        for i in range(nb):
            mask.append([length,init+i+1])
            tmp=list(sort[-1])
            tmp.append(interm[-1][1][0])
            sort.append(tmp)
            interm.append([[],interm[-1][1][1:]])
    else:
        init=mask[-1][0]
        nb=length-mask[-1][0]-1
        #print(nb)
        for i in range(nb):
            mask.append([init+i+1,length])
            tmp=list(sort[-1])
            tmp.append(interm[-1][0][0])
            sort.append(tmp)
            interm.append([interm[-1][0][1:],[]])
            
def mini(l,i):
    n=len(l)
    val_m=l[i]
    ind_m=i
    for i in range(i,n):
        if l[i]<val_m:
            val_m=l[i]
            ind_m=i
    return ind_m
            
            
            
def ins_sort(l):
    n=len(l)
    res=[l.copy()]
    msk=[]
    for i in range(n):
        ind=mini(l,i)
        tmp=l[i]
        l[i]=l[ind]
        l[ind]=tmp
        msk.append([i,ind])
        res.append(l.copy())
    return res,msk
        

def ins_data(t_size,v_size,length,binary_size):
    token=0
    num_max=2**binary_size-1
    nb=t_size+v_size
    liste=np.random.randint(1,num_max,(nb,length))
    dataset_value=np.zeros((nb,length+1,length),dtype=np.int)
    dataset_mask=np.zeros((nb,length,2),dtype=np.int)
    for i in range(nb):
        res,msk=ins_sort(liste[i])
        dataset_value[i,:,:]=res
        dataset_mask[i,:,:]=msk
    train= tf.data.Dataset.from_tensor_slices((dataset_value[:t_size],dataset_mask[:t_size],dataset_value[:t_size,-1,:]))
    val= tf.data.Dataset.from_tensor_slices((dataset_value[t_size:],dataset_mask[t_size:],dataset_value[t_size:,-1,:]))
    return train,val
        


def merge_data(t_size,v_size,length,binary_size):
    num_max=2**binary_size
    liste=np.random.randint(1,num_max,(t_size+v_size,2,length)) # each liste to be merge
    liste=np.sort(liste,axis=-1) 
    dataset_inter=np.zeros((len(liste),2*length+1,2*length+2),dtype=np.int)
    dataset_mask=np.zeros((len(liste),2*length+1,2),dtype=np.int)     
    dataset_res=np.zeros((len(liste),2*length+2),dtype=np.int)
    for j in range(len(liste)):
        mask=[[0,0]]
        interm=[]
        sort=[[]]
        liste_sort=merge(list(liste[j,0]),list(liste[j,1]),mask,interm,sort)
        #print(sort)
        
        if len(interm)<2*length-1:
            correct_inter(mask,interm,sort,length)
        #print(sort)
            
        res_inter=np.zeros((len(mask),2*length+2),dtype=np.int)
        res_inter[0,0]=0
        res_inter[0,1:length+1]=np.array(liste[j,0])
        res_inter[0,length+1]=0
        res_inter[0,length+2:]=np.array(liste[j,1])
        for i in range(len(mask)-1):
            tmp=sort[i+1].copy()
            #print(tmp)
            res_inter[i+1,:i+1]=np.array(tmp,dtype=np.int)
            #print(res_inter[i+1])
            res_inter[i+1,i+1]=num_max
            res_inter[i+1,i+2:i+2+len(interm[i][0])]=np.array(list(interm[i][0]),dtype=np.int)
            res_inter[i+1,i+1+len(interm[i][0])+1]=num_max
            res_inter[i+1,i+1+len(interm[i][0])+2:]=np.array(list(interm[i][1]),dtype=np.int)
        #print(res_inter.shape)
        liste_sort.append(num_max)
        liste_sort.append(num_max)
        dataset_inter[j,:-1,:]=res_inter
        dataset_inter[j,-1,:]=np.array(liste_sort,dtype=np.int)
        dataset_mask[j,:-1,:]=np.array(mask,dtype=np.int)
        dataset_mask[j,-1,:]=np.array([10,10])
        dataset_res[j,:]=np.array(liste_sort,dtype=np.int)
        
        train=tf.data.Dataset.from_tensor_slices((dataset_inter[:t_size],dataset_mask[:t_size],dataset_res[:t_size]))
        val=tf.data.Dataset.from_tensor_slices((dataset_inter[t_size:],dataset_mask[t_size:],dataset_res[t_size:]))
        
    return train,val
        #res_inter_bin=binary_encoding(res_inter,8)
        #res_bin=binary_encoding(np.array(liste_sort,dtype=np.int64),8)
def create_masks(mask, length):
    n=mask.shape[0]
    m=mask.shape[1]

    res=np.zeros((n,m,2*length+2),dtype=np.int)
    
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


train,val=merge_data(50,10,10,8)

#print(mask,res_inter)
