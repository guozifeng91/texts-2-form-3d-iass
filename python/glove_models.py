__author__    = ['Zifeng Guo' ]

import numpy as np
import os

def GloVe(path):
    '''
    load the GloVe word embedding model from a given location where 'tokens.m' and 'weights.npy' exist.
    
    'tokens.m' is a text file that defines the indices of token words.
    
    'weights.npy' is a float32 numpy array of M rows and N columns, where M = num(token words)+1 and N is the dimensionality of the embedding vectors
    
    the function returns a callable object that converts texts to a list of embedding vectors. e.g., GloVe('mypath/glove300')('hello world').shape=(2,300)
    '''
    class model:
        def __init__(self,w,t):
            self.w=w
            self.t=t
            self.k=t.keys()
#             self.size=len(t)
        
        def __call__(self, words):
            k=self.k
            w=self.w
            t=self.t
            words=words.lower().split(' ')
            indices=[t[wd] if wd in k else -1 for wd in words]
            
            return w[indices]
            
    fw=os.path.join(path,'weights.npy')
    ft=os.path.join(path,'tokens.m')
    if os.path.exists(fw) and os.path.exists(ft):
        weights=np.load(fw)
        
        with open(ft) as f:
            tokens=f.read()
        print(len(tokens))
        tokens='['+tokens[tokens.find('{')+1:tokens.rfind('}')]+']'
        tokens=eval(tokens)
        
        tokens={tokens[i]:i for i in range(len(tokens))}
        
        return model(weights, tokens)
        