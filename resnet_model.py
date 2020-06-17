import numpy as np
import tensorflow as tf

_batch_norm_decay = 0.999
_batch_norm_epsilon = 0.001

def _fully_connected(input_layer, num_labels):
    with tf.name_scope('fully_connected'):
     fc_h = tf.layers.dense(input_layer, num_labels)
    return fc_h

def _elu(x):
    return tf.nn.elu(x) if _activations=='elu' else tf.nn.relu(x)

def _batch_norm(input_layer, phase_train):
    bn_layer = tf.layers.batch_normalization(input_layer, training=phase_train, fused=None, momentum=_batch_norm_decay, epsilon=_batch_norm_epsilon)
    return bn_layer

def _avg_pool( x, pool_size, stride):
    with tf.name_scope('avg_pool'):
      x = tf.layers.average_pooling2d(x, pool_size, stride, 'SAME')
    return x

def _global_avg_pool(x):
    with tf.name_scope('global_avg_pool'):
      assert x.get_shape().ndims == 4
      x = tf.reduce_mean(x, [1, 2])
    return x

def _residual_v1(x,
                   kernel_size,
                   in_filter,
                   out_filter,
                   stride,
                   activate_before_residual=False, phase_train=True):
    """Residual unit with 2 sub layers, using Plan A for shortcut connection."""

    del activate_before_residual
    with tf.name_scope('residual_v1'):
      orig_x = x
      
      with tf.variable_scope('sub_1'):
          x = _conv(x, kernel_size, out_filter, stride)
          x = _batch_norm(x, phase_train)
          x = _elu(x)
      with tf.variable_scope('sub_2'):
          x = _conv(x, kernel_size, out_filter, 1)
          x = _batch_norm(x, phase_train)
         
      if in_filter != out_filter:
        orig_x = _avg_pool(orig_x, stride, stride)
        pad = (out_filter - in_filter) // 2
        orig_x = tf.pad(orig_x, [[0, 0], [0, 0], [0, 0], [pad, pad]])

      x = _elu(tf.add(x, orig_x))
      return x

def _conv(x, kernel_size, filters, strides, is_atrous=False):
    """Convolution."""
    padding = 'SAME'
    if not is_atrous and strides > 1:
      pad = kernel_size - 1
      pad_beg = pad // 2
      pad_end = pad - pad_beg
      x = tf.pad(x, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
      padding = 'VALID'
    return tf.layers.conv2d(
        inputs=x,
        kernel_size=kernel_size,
        filters=filters,
        strides=strides,
        padding=padding,
        use_bias=False)


def resnet_inference(input_tensor_batch, num_layers, num_classes=10, activations='elu', phase_train=True):
    '''
    inputs:
        input_tensor_batch: input placeholder
        num_layers: 56 or 20
        num_classes: 100 or 10 
        activations: relu or elu
        phase_train: True if training, False if testing
    return:
        logits (before softmax)
    '''
    global _activations
    _activations = activations
    
    print ('Activation:[%s]'%activations)
    
    n =  (num_layers - 2) // 6 # number of blocks 
    x = input_tensor_batch
    with tf.variable_scope('resblock_%d'%(0)):
        x = _conv(x, 3, 16, 1)
        x = _batch_norm(x, phase_train=phase_train)
        x = _elu(x)
        
       
    with tf.variable_scope('res_block_%d'%(1)):
        for j in range(n):
            with tf.variable_scope('layer_%d'%(j)):
                if j == 0:
                    # First block in a stage, filters and strides may change.
                    x = _residual_v1(x, 3, 16, 16, 1, phase_train)
                else:
                    # Following blocks in a stage, constant filters and unit stride.
                    x = _residual_v1(x, 3, 16, 16, 1, phase_train)
                    
    with tf.variable_scope('res_block_%d'%(2)):
        for j in range(n):
            with tf.variable_scope('layer_%d'%(j)):
                if j == 0:
                    # First block in a stage, filters and strides may change.
                    x = _residual_v1(x, 3, 16, 32, 2, phase_train)
                else:
                    # Following blocks in a stage, constant filters and unit stride.
                    x = _residual_v1(x, 3, 32, 32, 1, phase_train)                  
                    
                    
    with tf.variable_scope('res_block_%d'%(3)):
        for j in range(n):
            with tf.variable_scope('layer_%d'%(j)):
                if j == 0:
                    # First block in a stage, filters and strides may change.
                    x = _residual_v1(x, 3, 32, 64, 2, phase_train)
                else:
                    # Following blocks in a stage, constant filters and unit stride.
                    x = _residual_v1(x, 3, 64, 64, 1, phase_train)
                    
    x = _global_avg_pool(x)
    x = _fully_connected(x, num_classes)
    num_params = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])
    print ('buliding ResNet-%d'%num_layers )
    print ('parameters: [%d]'%num_params)
    return x
