import pdb
import tensorflow as tf

# UWMMSE
class UWMMSE(object):
        # Initialize
        def __init__( self, Pmax=1., var=7e-10, feature_dim=3, batch_size=64, layers=4, learning_rate=1e-3, max_gradient_norm=5.0, exp='uwmmse' ):
            self.Pmax              = tf.cast( Pmax, tf.float64 )
            self.var               = var
            self.feature_dim       = feature_dim
            self.batch_size        = batch_size
            self.layers            = layers
            self.learning_rate     = learning_rate
            self.max_gradient_norm = max_gradient_norm
            self.exp               = exp
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.build_model()

        # Build Model
        def build_model(self):
            self.init_placeholders()
            self.build_network()
            self.build_objective()
            
        def init_placeholders(self):
            # CSI [Batch_size X Nodes X Nodes]
            self.H = tf.compat.v1.placeholder(tf.float64, shape=[None, None, None], name="H")
            
            # NSI [Batch_size X Nodes X Features]
            #self.x = tf.compat.v1.placeholder(tf.float64, shape=[None, None, self.feature_dim], name="x")
            
            # Node Weights [Batch_size X Nodes]
            #self.alpha = tf.compat.v1.placeholder(tf.float64, shape=[None, None], name="alpha")
            
            # Boolean for Training/Inference 
            #self.phase = tf.compat.v1.placeholder_with_default(False, shape=(), name='phase')
        
        
        # Building network
        def build_network(self):
            # Squared H 
            self.Hsq = tf.math.square(self.H)
            
            # Diag H
            dH =  tf.linalg.diag_part( self.H ) 
            self.dH = tf.matrix_diag( dH )
            
            # Retrieve number of nodes for initializing V
            self.nNodes = tf.shape( self.H )[-1]

            # Maximum V = sqrt(Pmax)
            Vmax = tf.math.sqrt(self.Pmax)

            # Initial V
            V = Vmax * tf.ones([self.batch_size, self.nNodes], dtype=tf.float64)
            
            self.pow_alloc = []
            
            # Iterate over layers l
            for l in range(self.layers):
                with tf.variable_scope('Layer{}'.format(l+1)):
                    # Compute U^l
                    U = self.U_block( V )
                    
                    # Compute W^l
                    W_wmmse = self.W_block( U, V )
                    
                    # Learn a^l
                    a = self.gcn('a')

                    # Learn b^l
                    b = self.gcn('b')

                    # Compute Wcap^l = a^l * W^l + b^l
                    W = tf.math.add( tf.math.multiply( a, W_wmmse ), b )
                    
                    # Learn mu^l
                    mu = tf.get_variable( name='mu', initializer=tf.constant(0., shape=(), dtype=tf.float64))

                    # Compute V^l
                    if self.exp == 'wmmse':
                        V = self.V_block( U, W_wmmse, 0. )
                    else:
                        V = self.V_block( U, W, mu )
                    
                    # Saturation non-linearity  ->  0 <= V <= Vmax
                    V = tf.math.minimum(V, Vmax) + tf.math.maximum(V, 0) - V

            # Final V
            self.pow_alloc = V
        
        def U_block(self, V):
            # H_ii * v_i
            num = tf.math.multiply( tf.matrix_diag_part(self.H), V )
            
            # sigma^2 + sum_j( (H_ji)^2 * (v_j)^2 )
            den = tf.reshape( tf.matmul( tf.transpose( self.Hsq, perm=[0,2,1] ), tf.reshape( tf.math.square( V ), [-1, self.nNodes, 1] ) ), [-1, self.nNodes] ) + self.var 
            
            # U = num/den
            return( tf.math.divide( num, den ) )

        # Sum-rate = z
        def W_block(self, U, V):
            # 1 - u_i * H_ii * v_i
            den = 1. - tf.math.multiply( tf.matrix_diag_part(self.H), tf.math.multiply( U, V ) )
            
            # W = 1/den
            return( tf.math.reciprocal( den ) )

        # Weighted Sum-rate = a * z
        def W_block1(self, U, V):
            # 1 - u_i * H_ii * v_i
            den = 1. - tf.math.multiply( tf.matrix_diag_part(self.H), tf.math.multiply( U, V ) )
            
            # W = alpha/den
            return( tf.math.divide( self.alpha, den ) )        


        def gcn(self, name):
            # 2 Layers
            L = 2
            
            # Hidden dim = 5
            input_dim = [self.feature_dim,5]
            output_dim = [5,1]
            
            ## NSI [Batch_size X Nodes X Features]
            x = tf.ones([self.batch_size, self.nNodes, 1], dtype=tf.float64)
            #x = self.x
                        
            with tf.variable_scope('gcn_'+name):
                for l in range(L):
                    with tf.variable_scope('gc_l{}'.format(l+1)):
                        # Weights
                        w1 = tf.get_variable( name='w1', shape=(input_dim[l], output_dim[l]), initializer=tf.initializers.glorot_uniform(), dtype=tf.float64)
                        w0 = tf.get_variable( name='w0', shape=(input_dim[l], output_dim[l]), initializer=tf.initializers.glorot_uniform(), dtype=tf.float64)
                        
                        ## Biases
                        b1 = tf.get_variable( name='b1', initializer=tf.constant(0.1, shape=(output_dim[l],), dtype=tf.float64) )
                        b0 = tf.get_variable( name='b0', initializer=tf.constant(0.1, shape=(output_dim[l],), dtype=tf.float64) )
                        
                        # XW
                        x1 = tf.matmul(x, w1)
                        x0 = tf.matmul(x, w0)
                        
                        # diag(A)XW0 + AXW1
                        x1 = tf.matmul(self.H, x1)  
                        x0 = tf.matmul(self.dH, x0)
                        
                        ## AXW + B
                        x1 = tf.add(x1, b1)
                        x0 = tf.add(x0, b0)
                        
                        # Combine
                        x = x1 + x0
                        
                        # activation(AXW + B)
                        if l == 0:
                            x = tf.nn.relu(x)  
                        else:
                            x = tf.nn.sigmoid(x)

                # Coefficients (a / b) [Batch_size X Nodes]
                output = tf.squeeze(x)
            
            return output
        
        def V_block(self, U, W, mu):
            # H_ii * u_i * w_i
            num = tf.math.multiply( tf.matrix_diag_part(self.H), tf.math.multiply( U, W ) )
            
            # mu + sum_j( (H_ij)^2 * (u_j)^2 *w_j )
            den = tf.math.add( tf.reshape( tf.matmul( self.Hsq, tf.reshape( tf.math.multiply( tf.math.square( U ), W ), [-1, self.nNodes, 1] ) ), [-1, self.nNodes] ), mu)
            
            # V = num/den
            return( tf.math.divide( num, den ) )        
                                                                                
        def build_objective(self):
            # (H_ii)^2 * (v_i)^2
            num = tf.math.multiply( tf.matrix_diag_part(self.Hsq), tf.math.square( self.pow_alloc ) )
            
            # sigma^2 + sum_j j ~= i ( (H_ji)^2 * (v_j)^2 ) 
            den = tf.reshape( tf.matmul( tf.transpose( self.Hsq, perm=[0,2,1] ), tf.reshape( tf.math.square( self.pow_alloc ), [-1, self.nNodes, 1] ) ), [-1, self.nNodes] ) + self.var - num 
            
            # rate
            rate = tf.math.log( 1. + tf.math.divide( num, den ) ) / tf.cast( tf.math.log( 2.0 ), tf.float64 )
            
            # Sum Rate = sum_i ( log(1 + SINR) )
            self.utility = tf.reduce_sum( rate, axis=1 )
            
            # Weighted Sum Rate
            #rate = tf.math.multiply( self.alpha, rate )
            #self.utility = tf.reduce_sum( rate, axis=1 )
            
            # Minimization objective
            self.obj = -tf.reduce_mean( self.utility )
            
            if self.exp == 'uwmmse':
                self.init_optimizer()

        def init_optimizer(self):
            # Gradients and SGD update operation for training the model
            self.trainable_params = tf.compat.v1.trainable_variables()
            
            #Learning Rate Decay
            #starter_learning_rate = self.learning_rate
            #self.learning_rate_decayed = tf.train.exponential_decay(starter_learning_rate, global_step=self.global_step, decay_steps=5000, decay_rate=0.99, staircase=True)
            
            # SGD with Momentum
            #self.opt = tf.train.GradientDescentOptimizer( learning_rate=learning_rate )
            #self.opt = tf.train.MomentumOptimizer(learning_rate=self.learning_rate_decayed, momentum=0.9, use_nesterov=True )

            # Adam Optimizer
            self.opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)

            # Compute gradients of loss w.r.t. all trainable variables
            gradients = tf.gradients(self.obj, self.trainable_params)

            # Clip gradients by a given maximum_gradient_norm
            clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
            
            # Update the model
            self.updates = self.opt.apply_gradients(
                zip(clip_gradients, self.trainable_params), global_step=self.global_step)
                
        def save(self, sess, path, var_list=None, global_step=None):
            saver = tf.compat.v1.train.Saver(var_list)
            save_path = saver.save(sess, save_path=path, global_step=global_step)

        def restore(self, sess, path, var_list=None):
            saver = tf.compat.v1.train.Saver(var_list)
            saver.restore(sess, save_path=tf.train.latest_checkpoint(path))

        def train(self, sess, inputs ):
            input_feed = dict()
            input_feed[self.H.name] = inputs
            #input_feed[self.x.name] = features
            #input_feed[self.alpha.name] = alpha
            
            # Training Phase
            #input_feed[self.phase.name] = True
 
            output_feed = [self.obj, self.utility, self.pow_alloc, self.updates]
                            
            outputs = sess.run(output_feed, input_feed)
            
            return outputs[0], outputs[1], outputs[2]


        def eval(self, sess, inputs ):
            input_feed = dict()
            input_feed[self.H.name] = inputs
            #input_feed[self.x.name] = features
            #input_feed[self.alpha.name] = alpha

            # Training Phase
            #input_feed[self.phase.name] = False

            output_feed = [self.obj,self.utility, self.pow_alloc] 
                           
            outputs = sess.run(output_feed, input_feed)
            
            return outputs[0], outputs[1], outputs[2]
