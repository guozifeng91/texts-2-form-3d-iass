__author__    = ['Zifeng Guo' ]

import io
import numpy as np

def to_npy_byte(arr):
    with io.BytesIO() as t:
        np.save(t, arr)
        return t.getvalue()
    
def from_npy_byte(b):
    with io.BytesIO(b) as t:
        return np.load(t)
    
def write_npy_zip(zipfile, name, nd_array):
    zipfile.writestr(name,to_npy_byte(nd_array))
    
def read_npy_zip(zipfile, name):
    with zipfile.open(name) as f:
        d=f.read()
    return from_npy_byte(d)