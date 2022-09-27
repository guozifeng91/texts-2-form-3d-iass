import numpy as np
import pathlib
import os

def _read_topology_csv(csv_file,delimiter=','):
    '''
    import an topology csv.
    
    the csv should have a shape of (M) or (N,2), where M = N * 2
    the csv represents edge-node relationa by i1,j1,i2,j2,..., where i and j are node ids
    '''
    edges=np.genfromtxt(csv_file, delimiter=',')
    if len(edges.shape)==1 and len(edges) % 2==0:
        edges=edges.reshape((len(edges)//2,2))
    if len(edges.shape)!=2:
        raise Exception('csv cannot be interpreted as edge-node relations')
    
    if edges.dtype != np.int32:
        edges=edges.astype(np.int32)
    
    return edges

__path__ = pathlib.Path(__file__).parent.resolve()
__dataT__ = np.genfromtxt(os.path.join(__path__,'data.csv'), delimiter=',').T
__rows__=__dataT__.shape[1]
__topology__ = _read_topology_csv(os.path.join(__path__,'topology.csv'), delimiter=',')

# the constants related with the dataset
floors=10
node_num=176
trail_num=160
dev_num=240

node_x=10+floors+node_num
node_y=node_x+node_num
node_z=node_y+node_num

trail_length=node_z+trail_num
trail_mag=trail_length+trail_num

dev_length=trail_mag+dev_num
dev_mag=dev_length+dev_num

name_dict={'num_floors':[0],
           'num_nodes':[1],
           'num_trail_edges':[2],
           'num_dev_edges':[3],
           'grid_size_x':[4],
           'grid_size_y':[5],
           'load_x':[6],
           'load_y':[7],
           'load_z':[8],
           'total_lp':[9],
           'floor_heights':list(range(10,10+floors)),
           'node_x':list(range(10+floors,node_x)),
           'node_y':list(range(node_x,node_y)),
           'node_z':list(range(node_y,node_z)),
           'trail_length':list(range(node_z,trail_length)),
           'trail_mag':list(range(trail_length,trail_mag)),
           'dev_length':list(range(trail_mag,dev_length)),
           'dev_mag':list(range(dev_length,dev_mag)),
          }

names=list(name_dict.keys())
_get_all_cols=lambda x: ([i for x_ in x for i in name_dict[x_]]) if type(x) is list else name_dict[x]

def get_dataset_name():
    '''
    returns the name of the dataset
    '''
    return 'cem-towers'

def get_num_samples():
    '''
    return the total number of forms of the dataset (i.e., row number)
    '''
    return __rows__

def get_feature_names():
    '''
    returns all feature names of the dataset (i.e., column labels)
    '''
    return names

def get_forms(i):
    '''
    returns a dictionary that represents the i-th form diagram of the datasets
    
    ----- parameters -----
    i: the indice(s) of the form(s)
    
    ----- returns -----
    a dictionary / dictionaries
    '''
    if type(i) is int or type(i) is np.int32 or type(i) is np.int64:
        x=__dataT__[_get_all_cols('node_x')].T[i]
        y=__dataT__[_get_all_cols('node_y')].T[i]
        z=__dataT__[_get_all_cols('node_z')].T[i]

        coords=np.concatenate([x[...,None],y[...,None],z[...,None]],axis=-1)
        edges=__topology__
        forces=__dataT__[_get_all_cols(['trail_mag','dev_mag'])].T[i]

        return {'coords':coords,'edges':edges,'forces':forces}
    elif hasattr(i,'__len__'):
        return [get_forms(i_) for i_ in i]

def get_vector(indices, features):
    '''
    get the vector(s) for machine learning, by specifying indices (i.e., which rows) and features (i.e., which columns)
    
    the function returns an array of shape (m,n) sampled from the dataset based on given indice(s) and feature name(s), where m=len(indices) and n>=1, if multiple feature names are given, then n should be the column-concatenation of the vectors by each of the feature names
    
    for example, if feature name "pos" returns a three-column vector (x,y,z), and "len" returns a two-column vector (l1,l2),
    then get_vector([0,2,3], ['pos', 'len']) should returns a 3 x 5 matrix, in which the columns are (x,y,z,l1,l2)
    
    ----- parameters -----
    indices: integer or list of integers, row slice of the dataset
    features: string or list of strings, column slice of the dataset
    
    ----- returns -----
    a 2d array
    '''
    return __dataT__[_get_all_cols(features)].T[indices]

def get_dataset():
    '''
    deprecated
    
    returns a tuple of three items <D, i, names> for machine learning purpose
    
    where D is a callable "D(i, name)", i is an integer that shows the length (row number) of the dataset, and names is a list of strings show all available column entries
    
    D returns an array of (m,n) sampled from the dataset based on given indice(s) i and name(s), where m=len(i) and n>=1, if multiple names are given, then n should be the column-concatenation of the vectors by each of the names
    
    e.g., if name "pos" returns a three-column vector (x,y,z), and name "len" returns a two-column vector that (l1,l2),
    then interpret(['pos', 'len']) should returns a five-column vector (x,y,z,l1,l2)
    '''
    return lambda i,x: __dataT__[_get_all_cols(x)].T[i], __rows__, names

def get_topology():
    '''
    get the topology of the dataset in the format of trail path (connect all trail edges together) and deviation edges
    
    note that this function is for this dataset only and cannot be reused to other dataset due to the preassumption of the data in __topology__
    
    ----- returns -----
    trail_path, deviation_edge, to_trail_edge_indice, from_trail_edge_indice
    
    where trail_path and deviation_edge are list of integers,
    to_trail_edge_indice(i,j) is a callable which gets the trail edge indice from the input indice (i-th trail path, j-th edge)
    from_trail_edge_indice(i) is a callable which gets the i,j trail path indice from the trail edge indice
    '''
    node_num_per_floor = 16
    edge_num_per_trail = node_num // node_num_per_floor - 1 # 10
#     print(edge_num_per_trail)
    
    trail_edges=__topology__[:trail_num]
    deviation_edges=__topology__[trail_num:]
        
    trail_path=[[trail_edges[j * node_num_per_floor + i] for j in range(edge_num_per_trail)] for i in range(node_num_per_floor)]
    trail_path=[[*[s for s,e in trail],trail[-1][1]] for trail in trail_path]
    
    return trail_path, deviation_edges, lambda i,j: (j * node_num_per_floor + i), lambda i_edge: (i_edge%node_num_per_floor, i_edge//node_num_per_floor)