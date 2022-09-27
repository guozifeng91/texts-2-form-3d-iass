'''
untils for sompy by vahid

'''

__author__    = ['Zifeng Guo' ]

import numpy as np
import sompy

def find_bmus_for_vectors(v1, v2, axis=0, dist=False, k=1):
    '''
    axis:
    0, find the bmu for each element of v2 in v1
    1, find the bmu for each element of v1 in v2, which is equivalant to swapping v1 and v2
    
    dist: return only the indices if False and otherwise both indices and distances 
    
    k: return the nearest k neighbors, note that the first k neighbors are not sorted
    '''
    # the squared euclidean length of each input vector 
    x2=np.einsum('ij,ij->i', v2, v2)
    # the squared euclidean length of each som cell vector 
    y2=np.einsum('ij,ij->i', v1,v1)
    # the dot project of each input vector to each som cell vector
    d=np.dot(v1, v2.T)
    # the squared euclidean distance
    d = x2[None,...] - 2*d + y2[...,None]
    # partition the result, such that the 2nd item (indice=1) is in the correct place of a sorted list
    # and thus the 1st item is the minimum one
    
    # see https://numpy.org/doc/stable/reference/generated/numpy.partition.html#numpy.partition
    # argpartition returns the indice (bmus), while partition returns the partitioned list (dist)
    if axis==0:
        if dist:
            return np.argpartition(d,k-1,axis=0)[:k].T, np.partition(d,k-1,axis=0)[:k].T
        else:
            return np.argpartition(d,k-1,axis=0)[:k].T
    else:
        d=d.T
        if dist:
            return np.argpartition(d,k-1,axis=0)[:k].T, np.partition(d,k-1,axis=0)[:k].T
        else:
            return np.argpartition(d,k-1,axis=0)[:k].T
        
def find_bmus(som, data, axis=0, nth=1):
    '''
    vahid's source code missed the x2 component, therefore the returned distance is not correct (but the bmu is correct)
    
    this function gets the bmus and the distances
    
    axis:
    0, find the bmu (closest cell for each data)
    1, find the best-matching-item (closest data for each cell)
    
    '''
    data=np.asarray(data)
    # expand dimension
    if len(data.shape)==1:
        data=data[None,...]
        
    # normalize data if needed
    if som._normalizer.name != 'None':
        data=som._normalizer.normalize_by(som.data_raw, data)
    
    # the squared euclidean length of each input vector 
    x2=np.einsum('ij,ij->i', data, data)
    # the squared euclidean length of each som cell vector 
    y2 = np.einsum('ij,ij->i', som.codebook.matrix, som.codebook.matrix)
    # the dot project of each input vector to each som cell vector
    d=np.dot(som.codebook.matrix, data.T)
    # the squared euclidean distance
    d = x2[None,...] - 2*d + y2[...,None]
    # partition the result, such that the 2nd item (indice=1) is in the correct place of a sorted list
    # and thus the 1st item is the minimum one
    
    # see https://numpy.org/doc/stable/reference/generated/numpy.partition.html#numpy.partition
    # argpartition returns the indice (bmus), while partition returns the partitioned list (dist)
    if nth==1:
        if axis==0:
            return np.argpartition(d,1,axis=0)[0], np.partition(d,1,axis=0)[0]
        else:
            d=d.T
            return np.argpartition(d,1,axis=0)[0], np.partition(d,1,axis=0)[0]
    else:
        if axis==0:
            return np.argpartition(d,nth-1,axis=0)[:nth], np.partition(d,nth-1,axis=0)[:nth]
        else:
            d=d.T
            return np.argpartition(d,nth-1,axis=0)[:nth], np.partition(d,nth-1,axis=0)[:nth]
import io
import json
import zipfile

import sompy.normalization

class _VarianceNormalizer(sompy.normalization.Normalizer):
    '''
    a wrapper class to save the normalization info without keeping the dataraw
    '''
    name = 'var'
    def __init__(self,me,st):
        self.me=me
        self.st=st
        
    def _mean_and_standard_dev(self, data):
        # reuse the saved information regardless the input
        # data has no use, but is kept in case of errors
        return self.me, self.st

    def normalize(self, data):
        me = self.me
        st = self.st
        st[st == 0] = 1  # prevent: when sd = 0, normalized result = NaN
        return (data-me)/st

    def normalize_by(self, raw_data, data):
        # raw_data has no use
        me = self.me
        st = self.st
        st[st == 0] = 1  # prevent: when sd = 0, normalized result = NaN
        return (data-me)/st

    def denormalize_by(self, data_by, n_vect):
        me = self.me
        st = self.st
        return n_vect * st + me

def _get_som_meta_info(som):
    nname=som._normalizer.name
    me=None
    st=None
    if nname=='var':
        # calculate the mean and std from dataraw if a VarianceNormalizer is used (saving a new SOM)
        # or, reuse the saved mean and std regardless the dataraw if a _VarianceNormalizer is used (when a saved SOM is saved again)
        me,st=som._normalizer._mean_and_standard_dev(som.data_raw)
        # to list and python primitive
        me=[float(v) for v in me]
        st=[float(v) for v in st]
    return {
      'mapsize':som.codebook.mapsize,
      'mask':None,
      'mapshape':som.mapshape,
      'lattice':som.codebook.lattice,
      'normalization':nname,
      'initialization':som.initialization,
      'neighborhood':som.neighborhood.name,
      'training':som.training,
      'name':som.name,
      'component_names':None,
      'me':me,
      'st':st
    }

def save_som_to_file(filename, som, dataraw=False):
    '''
    save a trained som to zipfile
    
    note that dataraw (training data without normalization) if by default truned off
    '''
    with zipfile.ZipFile(filename,'w') as zip_file:
        # write meta information in json
        strio=io.StringIO()
        json.dump(_get_som_meta_info(som),strio)
        zip_file.writestr('meta.json',strio.getvalue())
        strio.close()
        
        # write data and codebook in numpy
        bio=io.BytesIO()
        np.save(bio,som._bmu)
        zip_file.writestr('_bmu.npy',bio.getvalue())
        bio.close()
        
        if dataraw:
            bio=io.BytesIO()
            np.save(bio,som.data_raw)
            zip_file.writestr('_dataraw.npy',bio.getvalue())
            bio.close()
            
#         if _data:
#             bio=io.BytesIO()
#             np.save(bio,som._data)
#             zip_file.writestr('_data.npy',bio.getvalue())
#             bio.close()
        
        bio=io.BytesIO()
        np.save(bio,som.codebook.matrix)
        zip_file.writestr('_codebook.npy',bio.getvalue())
        bio.close()

def load_som_from_file(filename, dataraw=None):
    '''
    load a saved som from file
    
    if dataraw is given, then the dataraw saved in the zipfile (if any) will not be read. By the next time the som is saved again, the new dataraw will replace the old one in the zip
    
    it is the user's responsibility to makesure that the given dataraw has the same distribution of the one used to train the SOM, otherwise the normalization will not work (if 'var' is used), making the SOM useless
    
    if dataraw is saved in the zip, please leave dataraw as None
    
    if dataraw is not saved in the zip AND cannot be recovered elsewhere, please also leave dataraw as None. In this case, a wrapper class will be activated (if normalization == 'var'), making the normalization remains valid.
    
    if normalization == 'None', the dataraw has no affect on the normalization process
    '''
    normalizer=None
    
    with zipfile.ZipFile(filename) as zip_file:
        # read meta
        with zip_file.open('meta.json','r') as ff:
            meta=ff.read()
        meta=json.loads(meta.decode())

        with zip_file.open('_bmu.npy','r') as ff:
            arr=ff.read()
        bio=io.BytesIO(arr)
        bmu=np.load(bio)
        bio.close()
        
        with zip_file.open('_codebook.npy','r') as ff:
            arr=ff.read()
        bio=io.BytesIO(arr)
        codebook=np.load(bio)
        bio.close()
        
        if dataraw is None:
            # read data if no dataraw is given
            if '_dataraw.npy' in zip_file.namelist():
                with zip_file.open('_dataraw.npy','r') as ff:
                    arr=ff.read()
                bio=io.BytesIO(arr)
                data=np.load(bio)
                bio.close()
            elif '_data.npy' in zip_file.namelist() and meta['normalization']=='None':
                print('old version detected')
                with zip_file.open('_data.npy','r') as ff:
                    arr=ff.read()
                bio=io.BytesIO(arr)
                data=np.load(bio)
                bio.close()
            else:
                print('dataraw is not given nor can be read from file, use random data instead (the bmu is not anymore valid)')
                if meta['normalization']=='var':
                    print('wrapper normalizer is used, the normalization info will not changed by dataraw')
                    # build a wrapper normalizer
                    me=np.asarray(meta['me'])
                    st=np.asarray(meta['st'])
                    normalizer=_VarianceNormalizer(me,st)
                data=np.random.random(size=(10,codebook.shape[-1]))
        else:
            data=dataraw

        som = sompy.SOMFactory.build(data,
                                     mapsize=meta['mapsize'],
                                     mask=meta['mask'],
                                     mapshape=meta['mapshape'],
                                     lattice=meta['lattice'],
                                     normalization=meta['normalization'], # use None for simplicity
                                     initialization=meta['initialization'],
                                     neighborhood=meta['neighborhood'],
                                     training=meta['training'],
                                     name=meta['name'])
        # override the normalizer
        if normalizer is not None:
            som._normalizer=normalizer
        
        # initialize the code book
        if som.initialization == 'random':
            som.codebook.random_initialization(som._data)

        elif som.initialization == 'pca':
            som.codebook.pca_linear_initialization(som._data)
        
        # override the training results
        som.codebook.matrix=codebook
        som._bmu=bmu
        
        return som