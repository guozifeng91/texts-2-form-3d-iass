'''
encoding using bag of words (BoW) method

BoW encoding
-first discretizes an object to vectors of (A x M) dimensions (called low-level encoding), this is typically done by other codes such as a NN-based feature extractor. currently there are no low-level encoding implemented in this file

-then computes the association between each of the low-level encoding and the reference vectors of (B x M) dimensions (typically by cosine distance or Euclidean distance), resulting vectors of (A x B) dimensions (called mid-level encoding)

-then computes the pooling of the mid-level encoding, producing a vector of B dimensions (called high-level encoding)
'''

__author__    = ['Zifeng Guo' ]

import numpy as np

# process the activation of an object (a collection of images) on SOM
def dist_matrix(v1, v2):
    return np.sqrt(np.abs(dist2_matrix(v1,v2)))

def dist2_matrix(v1, v2):
    '''
    scipy's distance_matrix is way too slow
    
    maybe this algorithm has lower precision (4 digits after the decimal), but it is worthy for its speed
    '''
    # the squared euclidean length of each input vector 
    x2=np.einsum('ij,ij->i', v2, v2)
    # the squared euclidean length of each som cell vector 
    y2=np.einsum('ij,ij->i', v1,v1)
    # the dot project of each input vector to each som cell vector
    d=np.dot(v1, v2.T)
    # the squared euclidean distance
    d = x2[None,...] - 2*d + y2[...,None]
    
    return d

# process the activation of an object (a collection of images) on SOM
def dist_matrix_f64(v1, v2):
    return np.sqrt(np.abs(dist2_matrix_f64(v1,v2)))

def dist2_matrix_f64(v1, v2):
    '''
    scipy's distance_matrix is way too slow
    
    maybe this algorithm has lower precision (4 digits after the decimal), but it is worthy for its speed
    '''
    v1=v1.astype(np.float64)
    v2=v2.astype(np.float64)
    
    # the squared euclidean length of each input vector 
    x2=np.einsum('ij,ij->i', v2, v2)
    # the squared euclidean length of each som cell vector 
    y2=np.einsum('ij,ij->i', v1,v1)
    # the dot product of each input vector to each som cell vector
    d=np.dot(v1, v2.T)
    # the squared euclidean distance
    d = x2[None,...] - 2*d + y2[...,None]
    
    return d

# def cosine_matrix(v1,v2):
#     '''
#     cosine=a.b/(|a||b|)
#     '''
#     m1 = np.sqrt(np.einsum('ij,ij->i', v1, v1))
#     m2 = np.sqrt(np.einsum('ij,ij->i', v2, v2))

#     d=np.dot(v1, v2.T)
    
#     print(m1)
#     print(m2)
#     return d / m2[None,...] / m1[...,None]

pooling_mean=lambda x:x.mean(axis=0)
pooling_max=lambda x:x.max(axis=0)

class minmax_normalizer:
    '''
    rescale the input vector so that all its elements are between 0 and 1
    '''
    def __init__(self, p=1):
        '''
        p: a parameter which will be applied to the output vector by y^p
        '''
        self.p=p
        
    def __call__(self, x):
        _min=x.min(axis=1)[...,None]
        _max=x.max(axis=1)[...,None]

        _range=_max-_min
        
        return ((x-_min) / _range)**self.p
    
class prob_normalizer:
    '''
    rescale the input vector so that the sum of all its elements is 1
    
    the computation is taken as (x^p) / sum(i, (x_i)^p)
    '''
    def __init__(self,p=1):
        self.p=p
        
    def __call__(self, x):
        x=x**self.p
        _sum=x.sum(axis=1)[...,None]
        return x/_sum
    
class softmax_normalizer:
    '''
    rescale the input vector so that the sum of all its elements is 1
    
    the computation is taken as exp(x^p) / sum(i, exp((x_i)^p))
    '''
    def __init__(self,p=1):
        self.p=p
        
    def __call__(self, x):
        x=np.exp(x**self.p)
        _sum=x.sum(axis=1)[...,None]
        return x/_sum

class magnitude_normalizer:
    '''
    rescale the input vector so that its magnitude is 1
    '''
    def __init__(self, p=1):
        '''
        p: a parameter which will be applied to the output vector by y^p
        '''
        self.p=p
        
    def __call__(self, x):
        _mag=np.sqrt((x*x).sum(axis=-1))
        _mag=_mag[...,None]

        return (x/_mag)**self.p

class mean_std_normalizer:
    '''
    normalize the vector by dimension-wised mean and std
    '''
    def __init__(self, data=None, m=None, s=None):
        if data is not None:
            self.m=data.mean(axis=0)
            self.s=data.std(axis=0)
        else:
            self.m=m
            self.s=s
        self.s[self.s == 0] = 1  # prevent: when sd = 0, normalized result = NaN
        
    def __call__(self, x):
        return (x-self.m)/self.s
    
from sklearn.metrics.pairwise import cosine_similarity
cosine = lambda x,y:cosine_similarity(x,y)
inverse_exp_cosine=lambda x,y: np.exp(-cosine_similarity(x,y))

euclidean=dist_matrix
inverse_euclidean=lambda x,y: 1.0/dist_matrix(x,y)
inverse_exp_euclidean=lambda x,y: np.exp(-dist_matrix(x,y))
inverse_euclidean_p1=lambda x,y: 1.0/(1+dist_matrix(x,y))

class Inverse_Exp_Euclidean:
    def __init__(self,sigma=1):
        self.sig=sigma
    def __call__(self,x,y):
        return np.exp(-dist_matrix(x,y) / self.sig)

euclidean2=dist2_matrix
inverse_euclidean2=lambda x,y: 1.0/dist2_matrix(x,y)
inverse_exp_euclidean2=lambda x,y: np.exp(-dist2_matrix(x,y))
inverse_euclidean2_p1=lambda x,y: 1.0/(1+dist2_matrix(x,y))

class Inverse_Exp_Euclidean2:
    def __init__(self,sigma=1):
        self.sig=sigma
    def __call__(self,x,y):
        return np.exp(-dist2_matrix(x,y) / self.sig)

    
class mid_level_coding():
    '''
    mid-level-coding
    '''
    def __init__(self, f, normalizer_input=None, normalizer_output=None):
        '''
        f: a function of two arguments that computs the mid-level encoding (assocition between two sets of vectors that is returned as a matrix)
        
        normalizer_input: normalization function to pre-process the input low-level encoding
        normalizer_output: normalization function to post-process the output mid-level encoding
        '''
        
#         self.ref_vec=ref_vec
        self.f=f
        self.n_in=normalizer_input
        self.n_out=normalizer_output
    
    def __call__(self, x, v):
        '''
        process the mid-level encoding
        '''
        if self.n_in is not None:
            x=self.n_in(x)
            
        y=self.f(x, v)
        
        if self.n_out is not None:
            y=self.n_out(y)
        
        return y

def get_default_mid_level_encoding(normalizer_in=None):
    '''
    default mid-level encoding used by my PhD.
    i.e., inverse_euclidean with minmax_normalizer(8)
    
    normalizer_in: the normalizer which pre-processes the input low-level encoding
    '''
    return mid_level_coding(inverse_euclidean, normalizer_in, minmax_normalizer(8))

def high_level_encoding(v, mid_lv, pooling):
    '''
    returns a callable that performs high level encoding (BoW encoding)
    
    v: a list of reference vectors
    mid_lv: a callable of two arguments, typically an instance of mid_level_coding. it can also be a function such as dist_matrix
    pooling: a callable of one arguments, computing the pooling of a list of vectors (i.e., from M x N vectors to N vectors), can be either pooling_mean or pooling_max
    '''
    
    return lambda x:pooling(mid_lv(x,v))

def get_default_high_level_encoding(v, normalizer_in=None):
    '''
    default high-level encoding used by my PhD.
    i.e., get_default_mid_level_encoding with pooling_mean
    
    v: reference vectors
    normalizer_in: the normalizer which pre-processes the input low-level encoding
    '''
    return high_level_encoding(v, get_default_mid_level_encoding(normalizer_in), pooling_mean)