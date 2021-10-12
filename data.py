import tensorflow as tf
import sparse
import numpy as np

def read(name):
    Gs=sparse.load_npz(path+'.npz')
    chong=np.load(path+'_chong.npy')

