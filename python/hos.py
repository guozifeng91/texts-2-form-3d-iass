'''
implementation of high-order statistics
'''

import numpy as np

def standardized_moment(x, k):
    '''
    returns the standardized moment of degree k for the input variable x
    https://en.wikipedia.org/wiki/Standardized_moment
    
    x must be a 1-d or 2-d numpy array. for 2-d array, the standardized_moment is calculated for the columns
    '''

    if k==1:
        return np.zeros_like(x[...,0])
    elif k==2:
        return np.zeros_like(x[...,0]) + 1
    else:
        n=x.shape[-1]

        mean = x.mean(axis=-1).astype(np.float64)
        diff = x - mean if len(x.shape)==1 else x - mean[...,None]

        bottom=(diff**2).sum(axis=-1)**(k/2)
        top=(diff**k).sum(axis=-1)
        coeff=n**(k/2-1)
    return coeff*top/bottom
            