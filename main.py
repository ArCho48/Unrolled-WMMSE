from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys
import pdb
import time
import math
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.logging.set_verbosity(tf.logging.INFO)

from model import UWMMSE

# Experiment 
dataID = sys.argv[1]
exp = sys.argv[2]
if len( sys.argv ) > 3:
    mode = sys.argv[3]

# Maximum available power at each node
Pmax = 1.0

# Noise power
var_db = -91
var = 10**(var_db/10)

# Features
feature_dim = 1

# Batch size
batch_size = 64

# Layers UWMMSE = 4 (default)  WMMSE = 100 (default)
layers = 4 if exp == 'uwmmse' else 100

# Learning rate
learning_rate=1e-3

# Number of epochs
nEpoch = 20

    
# Create Model Instance
def create_model( session, exp='uwmmse' ):
    # Create
    model = UWMMSE( Pmax=Pmax, var=var, feature_dim=feature_dim, batch_size=batch_size, layers=layers, learning_rate=learning_rate, exp=exp )
    
    # Initialize variables ( To train from scratch )
    session.run(tf.compat.v1.global_variables_initializer())
    
    return model

# Train
def mainTrain():        
    # Data
    H = pickle.load( open( 'data/'+dataID+'/H.pkl', 'rb' ) )
    
    #Training data
    train_H = H['train_H']
    
    #Test data
    test_H = H['test_H']
    
    # Initiate TF session
    with tf.compat.v1.Session(config=config) as sess:
    
        # WMMSE experiment
        if exp == 'wmmse':
        
            # Create model 
            model = create_model( sess, exp )
            
            # Test
            test_iter = len(test_H)
                    
            print( '\nWMMSE Started\n' )

            t = 0.
            test_rate = 0.0
            sum_rate = []
            
            for batch in range(test_iter):
                batch_test_inputs = test_H[batch]                
                start = time.time()
                avg_rate, batch_rate, batch_power = model.eval( sess, inputs=batch_test_inputs )
                t += (time.time() - start)
                test_rate += -avg_rate
                sum_rate.append( batch_rate )
                
            test_rate /= test_iter

            # Average per-iteration test time
            t = t / test_iter
            
            log = "Test_rate = {:.3f}, Time = {:.3f} sec\n"
            print(log.format( test_rate, t))
            
        # Unrolled WMMSE experiment
        else:
            
            # Create model 
            model = create_model( sess )

            if mode == 'train':
                # Create model path
                if not os.path.exists('models/'+dataID):
                    os.makedirs('models/'+dataID)
                    
                #Training loop
                print( '\nUWMMSE Training Started\n' )
                max_rate = 0.
                train_iter = len(train_H)
                
                #nEpoch = 1
                for epoch in range(nEpoch):
                    start = time.time()
                    train_rate = 0.0
                
                    for it in range(train_iter):
                        batch_train_inputs = train_H[it]
                        step_rate, batch_rate, power  = model.train( sess, inputs=batch_train_inputs ) 
                        if np.isnan(step_rate) or np.isinf(step_rate) :
                            pdb.set_trace()
                        train_rate += -step_rate
                    train_rate /= train_iter
                    
                    log = "Epoch {}/{}, Average Sum_rate = {:.3f}, Time = {:.3f} sec\n"
                    print(log.format( epoch+1, nEpoch, train_rate, time.time() - start) )
                    
                    # Save model with best average sum-rate
                    if train_rate > max_rate:
                        max_rate = train_rate
                        model.save(sess, path='models/'+dataID+'/uwmmse-model', global_step=(epoch+1))
                    
                    # Shuffle
                    shuffled_indices = np.random.permutation(train_iter)
                    train_H = [train_H[indx] for indx in shuffled_indices]

                    
                print( 'Training Complete' )

            # Test
            test_iter = len(test_H)       

            # Restore best saved model
            model.restore(sess, path='models/'+dataID+'/')

            print( '\nUWMMSE Testing Started\n' )

            t = 0.
            test_rate = 0.0
            sum_rate = []

            for batch in range(test_iter):
                batch_test_inputs = test_H[batch]
                start = time.time()
                avg_rate, batch_rate, batch_power = model.eval( sess, inputs=batch_test_inputs )
                if np.isnan(avg_rate) or np.isinf(avg_rate):
                    pdb.set_trace()
                t += (time.time() - start)
                sum_rate.append( batch_rate )
                test_rate += -avg_rate
            
            
            test_rate /= test_iter
            
            ## Average per-iteration test time   
            t = t / test_iter

            log = "Test_rate = {:.3f}, Time = {:.3f} sec\n"
            print(log.format( test_rate, t))

            
if __name__ == "__main__":        
    import sys

    rn = np.random.randint(2**20)
    rn1 = np.random.randint(2**20)
    tf.set_random_seed(rn)
    np.random.seed(rn1)

    mainTrain()
