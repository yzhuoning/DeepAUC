import tensorflow as tf
from sklearn import metrics


def cross_entropy_loss_with_l2(logits, labels, W=[], weight_decay=0.0005, use_L2=True):
    labels = tf.cast(labels, tf.int64)
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    print ('use_L2: [{}]'.format(use_L2))
    if use_L2:
        var_list_no_bias = [var for var in W if len(var.get_shape().as_list()) != 1] # no bias added
        l2_loss = tf.add_n([tf.nn.l2_loss(var) for idx, var in enumerate(var_list_no_bias)])
        loss_op = loss_op + l2_loss*weight_decay*2
    return loss_op
    
def AUC(y, pred):
    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
    return metrics.auc(fpr, tpr)
