import tensorflow as tf
import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import math_ops

def objective_function_batch(pred_score, a, b, alpha, p_hat, P_hat, Y):
    pred_score = pred_score[:, 1] 
    batch_size = Y.get_shape().as_list()[0]
    I_pos =  tf.math.maximum(np.zeros((batch_size,1), dtype =np.float32), Y)
    I_neg =  tf.abs(tf.math.maximum(np.zeros((batch_size,1), dtype =np.float32), -Y))
    obj = (1-p_hat)*tf.reduce_mean(I_pos*tf.reshape((pred_score-a)**2, (-1, 1)) + p_hat*I_neg*tf.reshape((pred_score-b)**2, (-1, 1))) + 2*(1+alpha)*tf.reduce_mean( p_hat*tf.reshape(pred_score, (-1, 1))*I_neg - (1-p_hat)*tf.reshape(pred_score, (-1, 1))*I_pos) - P_hat*alpha**2
    return obj
    
    
def PPD_SG(objective, eta, W, W0, a,b, a0, b0, alpha, gamma):
    update_ops = []

    grad_w = tf.gradients(objective, W)
    grad_a = tf.gradients(objective, a)
    grad_b = tf.gradients(objective, b)
    
    grad_v = grad_w + grad_a + grad_b
    V = W + [a, b]
    V0 = W0 + [a0, b0]
    
    grad_alpha = tf.gradients(objective, alpha)[0]
    
    update_ops_bn = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops_bn):
        
        for p, p0, g in zip(V, V0, grad_v):
            new_p = p - eta*(g + (1/gamma)*(p-p0))
            update_ops.append(p.assign(new_p))
        new_alpha = alpha + eta*grad_alpha
        update_ops.append(alpha.assign(new_alpha))
        
    return update_ops

   
def PPD_ADAGRAD(objective, stage_idx, T0, eta, W, W0, a, b, a0, b0, alpha, gamma, factor=1.2):
    update_ops = []

    grad_w = tf.gradients(objective, W)
    grad_a = tf.gradients(objective, a)
    grad_b = tf.gradients(objective, b)
    grad_alpha = tf.gradients(objective, alpha)[0]
    
    grad_v = grad_w + grad_a + grad_b + [-grad_alpha]
    V = W + [a, b] + [alpha]
    V0 = W0 + [a0, b0] + [0]
    d = np.sum([np.prod(v.get_shape().as_list()) for v in W])
    
    epsilon = 0.5
    accumulators =  [tf.Variable(tf.zeros(w.get_shape().as_list()) , dtype=tf.float32, name='acc') for w in V] #[K.zeros(w.get_shape().as_list()) for w in V] #
    grad_accumulators = [tf.Variable(tf.zeros(w.get_shape().as_list()) , dtype=tf.float32, name='grad_acc') for w in V]
    M_s = T0*math_ops.sqrt(factor**(stage_idx-1))
    max_i = tf.Variable(0, dtype=tf.float32, name='max_i')
    sum_gt = tf.Variable(0, dtype=tf.float32, name='sum_gt')
      
    update_ops_bn = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops_bn):
      
      count = 0
      for p, g, a, g_a, p0 in zip(V, grad_v, accumulators, grad_accumulators, V0):
            
          if count != len(V) -1:
             new_g = g + (1/gamma)*(p-p0)
          else:
             new_g = g
             
          new_g = g + (1/gamma)*(p-p0)     
          new_a = a + math_ops.square(new_g) 
          update_ops.append(a.assign(new_a))
          
          new_g_a = g_a + new_g  # gradient accumlators 
          update_ops.append(g_a.assign(new_g_a))
          
          tmp_max_i =  math_ops.maximum(max_i, K.sqrt(math_ops.reduce_max(new_a)))  
          tmp_sum_gt = K.sum(K.sqrt(new_a)) + sum_gt
              
          update_ops.append(max_i.assign(tmp_max_i))
          update_ops.append(sum_gt.assign(tmp_sum_gt))
          
          new_p = - eta * (new_g_a)/ ( K.sqrt(new_a) + epsilon) + p0 
          update_ops.append(p.assign(new_p))
          
          count += 1
          
    return_value = M_s* math_ops.sqrt((max_i + epsilon)*(sum_gt/(d+3)))
    return update_ops, return_value, accumulators, grad_accumulators, max_i, sum_gt, M_s
