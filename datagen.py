from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import pdb
import pickle
import numpy as np

# Eperiment
dataID = sys.argv[1]

# Number of nodes
nNodes = 20

# Path gain exponent
pl = 2.2

# Rayleigh distribution scale
alpha = 1 

# Batch size
batch_size = 64

# Training iterations
tr_iter = 10000

# Testing iterations
te_iter = 100


# Build random geometric graph
def build_adhoc_network( nNodes, r=1, pl=2.2 ):
    transmitters = np.random.uniform(low=-nNodes/r, high=nNodes/r, size=(nNodes,2))
    receivers = transmitters + np.random.uniform(low=-nNodes/4,high=nNodes/4, size=(nNodes,2))

    L = np.zeros((nNodes,nNodes))

    for i in np.arange(nNodes):
        for j in np.arange(nNodes):
            d = np.linalg.norm(transmitters[i,:]-receivers[j,:])
            L[i,j] = np.power(d,-pl)

    return( dict(zip(['tx', 'rx'],[transmitters, receivers] )), L )

# Simuate Fading
def sample_graph(batch_size, A, nNodes, alpha=1.):
    samples = np.random.rayleigh(alpha, (batch_size, nNodes, nNodes))
    #samples = (samples + np.transpose(samples,(0,2,1)))/2
    PP = samples[None,:,:] * A
    return PP[0]

# Training Data
def generate_data(batch_size, alpha, A, nNodes):
    tr_H = []
    te_H = []
    
    for indx in range(tr_iter):
        # sample training data 
        H = sample_graph(batch_size, A, nNodes, alpha )
        tr_H.append( H )

    for indx in range(tr_iter):
        # sample test data 
        H = sample_graph(batch_size, A, nNodes, alpha )
        te_H.append( H )

    return( dict(zip(['train_H', 'test_H'],[tr_H, te_H] ) ) )
        
def main():
    coord, A = build_adhoc_network( nNodes )
    
    # Create data path
    if not os.path.exists('data/'+dataID):
        os.makedirs('data/'+dataID)

    # Coordinates of nodes
    f = open('data/'+dataID+'/coordinates.pkl', 'wb')  
    pickle.dump(coord, f)         
    f.close()

    # Geometric graph
    f = open('data/'+dataID+'/A.pkl', 'wb')  
    pickle.dump(A, f)          
    f.close()
    
    # Training data
    data = generate_data(batch_size, alpha, A, nNodes)
    f = open('data/'+dataID+'/H.pkl', 'wb')  
    pickle.dump(data, f)         
    f.close()

    

if __name__ == '__main__':
    rn = np.random.randint(2**20)
    rn1 = np.random.randint(2**20)
    np.random.seed(rn1)

    main()
