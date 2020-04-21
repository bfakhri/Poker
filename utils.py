###############################################
# Defines utility functions for other scripts #
###############################################

import glob
from Dataset import CardDataset
import tensorflow as tf

################# BOOK KEEPING #################
def find_latest_model(model_dir):
    print('Searching for Saved Models in: ', model_dir)
    file_list = [f for f in glob.glob(model_dir+'*')]
    max_step = 0
    max_step_model = None
    for f in file_list:
        print('Evaluating ', f)
        step_str = f.split('_step_')[-1].split('/')[0]
        print('step_str: ', step_str)
        try:
            step = int(step_str)
            if(step > max_step):
                max_step = step
                max_step_model = f
        except: 
            continue

    print('Found latest model: ', max_step_model, max_step)
    return max_step_model, max_step


################# MODEL METRICS #################
def accuracy(y, y_hat, thresh=0.5):
    y_hat_bool = y_hat > thresh
    y_bool = y > thresh
    acc = tf.reduce_mean(tf.cast(tf.equal(y_bool, y_hat_bool), tf.float32))
    return acc

def accuracy_topk(y, y_hat):
    ''' Assuming you know the number of cards present, how accurate? '''
    num_cards = tf.cast(tf.reduce_sum(y, axis=-1), tf.int32)
    sorted_idxs = tf.argsort(y_hat, axis=-1, direction='DESCENDING')
    # Get the top num_cards predicitons
    summer = 0
    for i in range(y.shape[0]):
        top_idxs = sorted_idxs[i,0:num_cards[i]]
        y_top = tf.gather(y[i,:], top_idxs)
        summer += tf.reduce_sum(y_top)
    acc = tf.cast(summer, tf.float32)/tf.cast(tf.reduce_sum(num_cards), tf.float32)
    return acc

def precision_recall(y, y_hat, thresh=0.5):
    y_hat_bool = y_hat > thresh
    y_bool = y > thresh
    TP = tf.reduce_sum(tf.cast(tf.logical_and(y_bool, y_hat_bool), tf.float32))
    precision = TP/tf.reduce_sum(y_hat)
    recall = TP/tf.reduce_sum(tf.cast(y_bool, tf.float32))
    return precision, recall

# What does the deck look like? 

card_ds = CardDataset()
def cards_from_preds(preds, k):
    ''' 
    Given predictions, returns cards for the top k predictions 
    Input: assumes preds are of shape (52,)
    '''
    sorted_idxs = tf.argsort(preds, direction='DESCENDING')
    top_k_idxs = sorted_idxs[:k]
    top_k_cards = card_ds.flatidx_to_card(top_k_idxs)
    return top_k_cards

