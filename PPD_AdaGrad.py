# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
from datetime import datetime
from resnet_model import resnet_inference
import cifar_input as cifar_data
import my_utils
import optimizer as opt


tf.reset_default_graph()
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model', 'resnet', '''which model to train: resnet or convnet''')
tf.app.flags.DEFINE_string('activation', 'elu', '''activation function to use: relu or elu''')
tf.app.flags.DEFINE_integer('K', 1000, '''Number of stages''')
tf.app.flags.DEFINE_integer('random_seed', 123, '''random seed for initialization''')
tf.app.flags.DEFINE_integer('train_batch_size', 128, '''batch_size''')
tf.app.flags.DEFINE_integer('t0', 200, '''T0 for stagewise training''')
tf.app.flags.DEFINE_float('lr', 0.1, '''learning rate to train the models''')
tf.app.flags.DEFINE_integer('split_index', 4, '''index where to partition the dataset''')
tf.app.flags.DEFINE_float('keep_index', 0.1, '''portion of data to keep ''')
tf.app.flags.DEFINE_integer('dataset', 10, '''dataset to evalute: 10 or 100 or 2''')
tf.app.flags.DEFINE_integer('resnet_layers', 20, '''number of layers to use in ResNet: 56 or 20; if convnet, make it to 3''')
tf.app.flags.DEFINE_boolean('use_avg', False, '''if True, use avg to evaluate''')
tf.app.flags.DEFINE_boolean('is_tune', False, '''if True, split train dataset (50K) into 45K, 5K as train/validation data''')
tf.app.flags.DEFINE_boolean('is_crop_flip', False, '''if True, make train_data random_crop_flip''')
tf.app.flags.DEFINE_boolean('use_L2', False, '''whether to use L2 regularizer''')


tf.set_random_seed(FLAGS.random_seed)

# Import CIFAR data
if FLAGS.dataset != 2 and FLAGS.is_stl10 == False:
    (train_data, train_labels), (test_data, test_labels) = cifar_data.load_data(FLAGS.dataset, FLAGS.is_tune, FLAGS.is_crop_flip)
    split_index = FLAGS.split_index if FLAGS.dataset==10 else FLAGS.split_index
    train_labels[train_labels<=split_index] = -1 # [0, ....]
    test_labels[test_labels<=split_index] = -1
    train_labels[train_labels>=split_index+1] = 1 # [0, ....]
    test_labels[test_labels>=split_index+1] = 1
    
    train_ids = list(range(train_data.shape[0]))
    np.random.seed(123)
    np.random.shuffle(train_ids)
    train_data = train_data[train_ids]
    train_labels = train_labels[train_ids ]
    
    # delete some samples
    num_neg = np.where(train_labels==-1)[0].shape[0]
    idx_neg_tmp = np.where(train_labels==-1)[0][:int(num_neg*FLAGS.keep_index)]
    idx_pos_tmp = np.where(train_labels==1)[0]
    train_data = train_data[idx_neg_tmp.tolist() + idx_pos_tmp.tolist() ] 
    train_labels = train_labels[idx_neg_tmp.tolist() + idx_pos_tmp.tolist() ] 

    pos_count = np.count_nonzero(train_labels == 1)
    neg_count = np.count_nonzero(train_labels == -1)
    print ('Pos:Neg: [%d : %d]'%(np.count_nonzero(train_labels == 1), np.count_nonzero(train_labels == -1)))



batch_size = FLAGS.train_batch_size
inference = resnet_inference 

# tuned paramaters
initial_learning_rate = FLAGS.lr
T0 = FLAGS.t0
gamma_ = 2000 # weakly convex parameters
factor = 9

img_size = train_data.shape[1]
channel_size = train_data.shape[-1]
X = tf.placeholder(tf.float32, [batch_size, img_size, img_size, channel_size])
Y = tf.placeholder(tf.float32, [batch_size, 1])
phase_train = tf.placeholder(tf.bool, name='phase_train')

logits = inference(X, num_classes=2, num_layers=FLAGS.resnet_layers, activations=FLAGS.activation, phase_train=phase_train) # when resnet you need to pass number of layers 
pred_score = tf.nn.softmax(logits)


W = [var for var in tf.trainable_variables ()]
a = tf.Variable([0], dtype=tf.float32, name='a') 
b = tf.Variable([0], dtype=tf.float32, name='b') 
alpha = tf.Variable([0], dtype=tf.float32, name='alpha')

# placeholders
p = tf.placeholder(tf.float32, shape=(1,))
p_hat = tf.placeholder(tf.float32, shape=(1,))
P_hat = tf.placeholder(tf.float32, shape=(1,))
W0 = [tf.placeholder(tf.float32, shape=w.get_shape().as_list()) for w in W]
a0 = tf.placeholder(tf.float32, shape=(1,), name='a0') 
b0 = tf.placeholder(tf.float32, shape=(1,), name='b0') 
eta = tf.placeholder(tf.float32, shape=(1,))
# gamma needs to be given 
gamma = tf.Variable(gamma_, dtype=tf.float32, name='gamma')
stage_idx = tf.placeholder(tf.float32)

# objective function 
objective = opt.objective_function_batch(pred_score, a, b, alpha, p_hat, P_hat, Y) 
train_op, return_value, accumulators, grad_accumulators, max_, sum_, M_s = opt.PPD_ADAGRAD(objective, stage_idx, T0, eta, W, W0, a, b, a0, b0, alpha, gamma, factor=factor)

# init
init = tf.global_variables_initializer()

# shuffle data 
train_ids = list(range(train_data.shape[0]))
np.random.seed(None)
np.random.shuffle(train_ids)
train_data = train_data[train_ids]
train_labels = train_labels[train_ids]
test_auc = []
test_iter = []
total_iter = 0
num_batch = train_labels.shape[0]//batch_size

# Start training
with tf.Session() as sess:  
    sess = tf.Session()


    sess.run(init)
    print ('\nStart training...')
    
    W_avg = [np.zeros(w.get_shape().as_list()) for w in W]
    a_avg = 0
    b_avg = 0
    alpha_avg = 0
    
    W_avg_acc = sess.run(W)
    a_avg_acc = 0
    b_avg_acc = 0
    
    assign_W = [tf.placeholder(tf.float32, w.get_shape().as_list()) for w in W]
    update_W_ops = [var.assign(assign_W[idx]) for idx, var in enumerate(W) if len(var.get_shape().as_list()) != 1] 

    for k in range(1, FLAGS.K+1):
        
        if total_iter == 80000:
            break
        
        T_k = T0*(np.sqrt(factor)**(k-1))
        sess.run(alpha.assign([alpha_avg]))
        a_avg = a_avg_acc
        b_avg = b_avg_acc
        W_avg = W_avg_acc

        # reset max_i, sum_gt at each stage 
        update_ops1 = [var.assign(np.zeros(var.get_shape().as_list())) for var in accumulators]
        update_ops2 = [var.assign(np.zeros(var.get_shape().as_list())) for var in grad_accumulators] 
        sess.run(update_ops1)
        sess.run(update_ops2)
        sess.run([max_.assign(0),  sum_.assign(0)])
        
        for t in range(1, int(T_k)+1):
            total_iter += 1

            # stop 2
            if total_iter == 80000:
                break
        
            idx = total_iter % num_batch
            if idx == 0: # shuffle dataset every epoch 
                np.random.shuffle(train_ids)
                train_data = train_data[train_ids]
                train_labels = train_labels[train_ids]
                idx += 1

            offset = (idx-1) * batch_size
            batch_x, batch_y = (train_data[offset:offset+batch_size], train_labels[offset:offset+batch_size][:, np.newaxis])
            
            # initialization
            if total_iter == 1:
 
                T_pos = sum([1 for y_ in batch_y if y_ > 0])
                T_neg = sum([1 for y_ in batch_y if y_ < 0])
                p_hat_ = T_pos/(T_pos + T_neg)
                y_hat = (T_pos)/batch_size 
                P_hat_ = sum([(1-y_hat)**2 for y_ in batch_y if y_ > 0 ] )/(batch_size-1) 
            else:
                T_pos = T_pos + sum([1 for y_ in batch_y if y_ > 0])
                T_neg = T_neg + sum([1 for y_ in batch_y if y_ < 0])
                p_hat_ = T_pos/(T_pos + T_neg)
                y_hat = (( (total_iter-1)*batch_size )*y_hat + sum([1 for y in batch_y if y >0])  )/((total_iter)*batch_size)
                P_hat_ = p_hat_*(1-p_hat_)

            eta_k = initial_learning_rate*(1/np.sqrt(factor)**(k-1))
            
            feed_dict = {}
            VARs = [X, Y, p_hat, P_hat, eta, phase_train] + W0 + [a0, b0] + [stage_idx]
            VALUEs = [batch_x, batch_y, [p_hat_], [P_hat_], [eta_k], True] + W_avg + [[a_avg], [b_avg]] + [k+1]
            for var, value in zip(VARs, VALUEs):
                feed_dict[var] = value 
            
            # optimization
            _ = sess.run(train_op, feed_dict=feed_dict)

            # evaluation
            if total_iter % 400 == 0 or total_iter == 1:
                images, labels = test_data, test_labels
                num_batches = images.shape[0]//batch_size
                pred_probs = []
                for step in range(num_batches):
                    offset = step * batch_size
                    vali_data_batch = images[offset:offset+batch_size]
                    vali_label_batch = labels[offset:offset+batch_size][:, np.newaxis]
                    score = sess.run(pred_score, feed_dict={X: vali_data_batch, phase_train:False})
                    score = score[:, 1].flatten()
                    pred_probs.extend(score.tolist())
                    
                auc = my_utils.AUC(test_labels[:num_batches*batch_size], pred_probs)
                test_auc.append(auc)
                test_iter.append(total_iter)
                print ('%s: [%d] pos_ratio:%.4f, auc:%.4f, eta_k:%.4f, T_k:%d'%(datetime.now(), total_iter, p_hat_, auc, eta_k, T_k))

            # accumlated average 
            W_local, a_local, b_local = sess.run([W, a, b])
            W_avg_acc = [W_avg_acc[idx]+W_local[idx] for idx, w in enumerate(W)]
            a_avg_acc += a_local[0]
            b_avg_acc += b_local[0]     
            
        T_k = t
        W_avg_acc = [w/T_k for w in W_avg_acc]
        a_avg_acc /= T_k
        b_avg_acc /= T_k 
                
        extra_batch_num = (3**k) if (3**k) < num_batch else num_batch
        num_batch_tmp = train_data.shape[0]//(batch_size*extra_batch_num)
        idx_extra = 1 
        
        # shuffle data
        np.random.shuffle(train_ids)
        train_data = train_data[train_ids]
        train_labels = train_labels[train_ids]
    
        offset = (idx_extra-1) * batch_size*extra_batch_num
        batch_x_extra  = train_data[offset:offset+batch_size*extra_batch_num]
        batch_y_extra = train_labels[offset:offset+batch_size*extra_batch_num]
        pos_count = sum([1 for y_ in batch_y_extra if y_ > 0])
        neg_count = sum([1 for y_ in batch_y_extra if y_ < 0])

        W_values = sess.run(W)
        # evaluate at current average solution
        assign_dict = {}
        for p, v in zip(assign_W, W_avg_acc):
           assign_dict[p]=v
        sess.run(update_W_ops, assign_dict)
        
        score = []
        for i in range(1, extra_batch_num+1):
            offset = (i-1) * batch_size
            s_out = sess.run(pred_score, feed_dict={X: batch_x_extra[offset:offset+batch_size], phase_train:False})
            score.extend(s_out)

        # change weights back to current weights 
        assign_dict = {}
        for p, v in zip(assign_W, W_values):
            assign_dict[p]=v
        sess.run(update_W_ops, assign_dict) 
        
        
        # count score 
        score_pos = 0
        score_neg = 0
        for idx, sc_  in enumerate(score):
            if batch_y_extra[idx] > 0:
                score_pos += score[idx][-1]
            else:
                score_neg += score[idx][-1]
                
        alpha_avg = (score_neg)/neg_count - (score_pos)/pos_count

        T_pos = T_pos + sum([1 for y_ in batch_y_extra if y_ > 0])
        T_neg = T_neg + sum([1 for y_ in batch_y_extra if y_ < 0])
        p_hat_ = T_pos/(T_pos + T_neg)
        y_hat = (( (total_iter-1)*batch_size )*y_hat + sum([1 for y in batch_y_extra if y >0])  )/((total_iter)*batch_size)
        P_hat_ = p_hat_*(1-p_hat_)
        print ('Restart...')

        