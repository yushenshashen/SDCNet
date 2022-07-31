#!/usr/bin/env python

"""
Usage:
nohup python get_sdcnet_loewe_load_pretrainmodel.py  >> logs/log_sdcnet_loewe_load_pretrainmodel.txt 2>&1 &
python get_sdcnet_loewe_load_pretrainmodel.py -modelfile ../trained_model/sdcnet_loewe/best_model.ckpt
"""

import time
import os
import numpy as np
import pandas as pd
# import networkx as nx
import scipy.sparse as sp
from itertools import islice, combinations
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc, f1_score, accuracy_score, precision_score, recall_score
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
# Train on CPU (hide GPU) due to memory constraints
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.compat.v1.disable_v2_behavior() 

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
config.allow_soft_placement = True
config.log_device_placement = True
config.gpu_options.per_process_gpu_memory_fraction = 0.3

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 1, 'Number of epochs to train.')
flags.DEFINE_integer('embedding_dim', 320, 'Number of the dim of embedding')
flags.DEFINE_float('dropout', 0.2, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('val_test_size', 0.1, 'the rate of validation and test samples.')
flags.DEFINE_string('modelfile', '../trained_model/sdcnet_loewe/best_model.ckpt', 'the path of the trained model file')

#some usefull funs
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)

# 1. load the data
import numpy as np
import pandas as pd
data = pd.read_csv('../data/oneil_dataset_loewe.txt', sep='\t', header=0)
data.columns = ['drugname1','drugname2','cell_line','synergy']
# data.columns = ['drugname1','drugname2','cell_line', 'zip', 'bliss', 'loewe', 'hsa']

drugslist = sorted(list(set(list(data['drugname1']) + list(data['drugname2'])))) #38
drugscount = len(drugslist)
cellslist = sorted(list(set(data['cell_line']))) 
cellscount = len(cellslist)

features = pd.read_csv('../data/oneil_drug_informax_feat.txt',sep='\t', header=None)

drug_feat = sp.csr_matrix( np.array(features) )
drug_feat = sparse_to_tuple(drug_feat.tocoo())
num_drug_feat = drug_feat[2][1]
num_drug_nonzeros = drug_feat[1].shape[0]


# if not os.path.isdir('results'):
#     os.makedirs('results')

# if not os.path.isdir('logs'):
#     os.makedirs('logs')

resultspath = '../results_oneil/results_sdcnet_loewe3'
if not os.path.isdir(resultspath):
    os.makedirs(resultspath)

#begin the cross validation
all_indexs = []
all_edges = []
for idx1 in range(drugscount):
    for idx2 in range(drugscount):
        all_indexs.append([idx1,idx2])
        if idx1 < idx2:
            all_edges.append([idx1, idx2])

diags_edges = []
diags_indexs = []
for idx in range(drugscount):
    diags_edges.append([idx, idx])
    diags_indexs.append( all_indexs.index([idx, idx]) )

num_folds = 10
all_stats = np.zeros((num_folds, 6))
merged_stats = np.zeros((num_folds, 6))
# for foldidx in range(1, num_folds):
foldidx = 0
print('processing fold ', foldidx)

d_net1_norm = {}
d_net2_norm = {}
d_net1_orig = {}
d_net2_orig = {}
d_pos_weights = {}
d_train_edges = {}
d_train_indexs = {}
d_train_labels = {}
d_test_edges = {}
d_test_labels = {}
d_new_edges = {}
d_net3_edges = {}
d_valid_edges = {}
d_valid_labels = {}
for cellidx in range(cellscount):
    # cellidx = 0
    cellname = cellslist[cellidx]
    print('processing ', cellname)
    each_data = data[data['cell_line']==cellname]
    net1_data = each_data[each_data['synergy'] >= 10]
    net2_data = each_data[each_data['synergy'] < 0] 
    net3_data = each_data[(each_data['synergy'] >= 0) & (each_data['synergy'] < 10)]
    # if net2_data.shape[0] < 10: 
    #     num_need = 10 - net2_data.shape[0]
    #     net2_data = pd.concat([net2_data, net3_data[:num_need]])
    print(net1_data.shape, net2_data.shape, net3_data.shape)
    d_net1 = {}
    for each in net1_data.values:
        drugname1, drugname2, cell_line, synergy = each
        key = drugname1+ '&' + drugname2
        d_net1[key] = each
        key = drugname2+ '&' + drugname1
        d_net1[key] = each
    d_net2 = {}
    for each in net2_data.values:
        drugname1, drugname2, cell_line, synergy = each
        key = drugname1+ '&' + drugname2
        d_net2[key] = each
        key = drugname2 + '&' + drugname1
        d_net2[key] = each

    adj_net1_mat = np.zeros((drugscount, drugscount))
    adj_net2_mat = np.zeros((drugscount, drugscount))

    for i in range(drugscount):
        for j in range(drugscount):
            drugname1 = drugslist[i]
            drugname2 = drugslist[j]
            key1 = drugname1 + '&' + drugname2
            key2 = drugname2 + '&' + drugname1
            if key1 in d_net1.keys() or key2 in d_net1.keys():
                adj_net1_mat[i, j] = 1
            elif key1 in d_net2.keys() or key2 in d_net2.keys():
                adj_net2_mat[i, j] = 1

    adj_net1 = sp.csr_matrix(adj_net1_mat)
    adj_net2 = sp.csr_matrix(adj_net2_mat)

    net1_edges = sparse_to_tuple(sp.triu(adj_net1))[0]
    net2_edges = sparse_to_tuple(sp.triu(adj_net2))[0]

    #split the train and test edges
    num_test = int(np.floor(net1_edges.shape[0] * FLAGS.val_test_size))
    net1_edge_idx = list(range(net1_edges.shape[0]))
    np.random.seed(1)
    np.random.shuffle(net1_edge_idx)
    if foldidx == 0:
        net1_test_edge_idx = net1_edge_idx[(foldidx - 1) * num_test: ]
    else:
        net1_test_edge_idx = net1_edge_idx[(foldidx -1 )* num_test: foldidx *num_test]
    net1_valid_edge_idx = net1_edge_idx[foldidx * num_test: (foldidx+1)*num_test]
    net1_test_edges = net1_edges[ net1_test_edge_idx ]
    net1_valid_edges = net1_edges[ net1_valid_edge_idx ]
    net1_train_edge_idx = [ x for x in net1_edge_idx if x not in net1_test_edge_idx + net1_valid_edge_idx ]
    net1_train_edges = net1_edges[net1_train_edge_idx]
    net1_train_data = np.ones(net1_train_edges.shape[0])
    net1_adj_train = sp.csr_matrix( (net1_train_data, (net1_train_edges[:, 0], net1_train_edges[:, 1])), shape= adj_net1.shape )
    net1_adj_train = net1_adj_train + net1_adj_train.T
    net1_adj_norm = preprocess_graph(net1_adj_train)
    net1_adj_orig = net1_adj_train.copy() #this the label
    net1_adj_orig = sparse_to_tuple(sp.csr_matrix(net1_adj_orig))

    ##net2
    ##net2
    net2_edge_idx = list(range(net2_edges.shape[0]))
    #1.the number of negative samples are split into equal subsets
    # num_test2 = int(np.floor(net2_edges.shape[0] * FLAGS.val_test_size))

    #2. the number of negative sample is equal to positive samples
    num_test2 = num_test

    np.random.seed(2)
    np.random.shuffle(net2_edge_idx)
    if foldidx == 0:
        net2_test_edge_idx = net2_edge_idx[(foldidx - 1) * num_test2: ]
    else:
        net2_test_edge_idx = net2_edge_idx[(foldidx -1 )* num_test2: foldidx *num_test2]
    net2_valid_edge_idx = net2_edge_idx[foldidx * num_test2: (foldidx+1)*num_test2]
    # net2_valid_test_edge_idx = np.random.choice(net2_edge_idx, num_test * 2)
    # net2_test_edge_idx = net2_valid_test_edge_idx[:num_test]
    # net2_valid_edge_idx = net2_valid_test_edge_idx[num_test:]
    net2_test_edges = net2_edges[ net2_test_edge_idx ]
    net2_valid_edges = net2_edges[ net2_valid_edge_idx ]
    net2_train_edge_idx = [ x for x in net2_edge_idx if x not in net2_test_edge_idx + net2_valid_edge_idx ]
    net2_train_edges = net2_edges[net2_train_edge_idx]
    ##
    net1_train_edges_symmetry = np.array([  [x[1],x[0]] for x in net1_train_edges ])
    net2_train_edges_symmetry = np.array([  [x[1],x[0]] for x in net2_train_edges ])
    net1_train_edges = np.concatenate([net1_train_edges, net1_train_edges_symmetry])
    net2_train_edges = np.concatenate([net2_train_edges, net2_train_edges_symmetry])
    # net1_train_index = [ all_indexs.index(x) for x in np.concatenate([net1_train_edges, net1_train_edges_symmetry]).tolist() ]
    # net2_train_index = [ all_indexs.index(x) for x in np.concatenate([ net2_train_edges, net2_train_edges_symmetry]).tolist()]
    # train_indexs = [net1_train_index, net2_train_index ]
    test_edges = np.concatenate([net1_test_edges, net2_test_edges])
    y_test = [1] * net1_test_edges.shape[0] + [0] * net2_test_edges.shape[0]
    valid_edges = np.concatenate([net1_valid_edges, net2_valid_edges])
    y_valid = [1] * net1_valid_edges.shape[0] + [0] * net2_valid_edges.shape[0]
    train_edges = np.concatenate([net1_train_edges, net2_train_edges])
    y_train = [1] * net1_train_edges.shape[0] + [0] * net2_train_edges.shape[0]
    train_indexs = [ all_indexs.index(x) for x in train_edges.tolist() ]
    each_pos_weight = len(net2_train_edges) / len(net1_train_edges)
    # each_pos_weight =  len(net1_train_edges) / len(net2_train_edges)
    d_pos_weights[cellidx] = each_pos_weight
    d_net1_norm[cellidx] = net1_adj_norm
    d_net1_orig[cellidx] = net1_adj_orig
    d_test_edges[cellidx] = test_edges
    d_test_labels[cellidx] = y_test
    d_train_edges[cellidx] = train_edges
    d_train_indexs[cellidx] = train_indexs
    d_train_labels[cellidx] = y_train
    d_valid_edges[cellidx] = valid_edges
    d_valid_labels[cellidx] = y_valid

placeholders = {
    'features': tf.sparse_placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=()),
}
placeholders.update({'net1_adj_norm_'+str(cellidx) : tf.sparse_placeholder(tf.float32) for cellidx in range(cellscount)})

# Create model
from models.model_mult import sdcnet
model = sdcnet(placeholders, num_drug_feat, FLAGS.embedding_dim, num_drug_nonzeros, name='sdcnet', use_cellweights=True, use_layerweights=True,  fncellscount =cellscount )
#optimizer
from models.optimizer_mult import Optimizer
with tf.name_scope('optimizer'):
    opt = Optimizer(preds= model.reconstructions, d_labels= d_train_labels, model=model, lr= FLAGS.learning_rate, d_pos_weights = d_pos_weights, d_indexs = d_train_indexs )

# Initialize session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=1)

best_model_file = FLAGS.modelfile
# best_model_file = '../trained_model/results_sdcnet_loewe/best_model.ckpt'
# best_model_file = resultspath + '/best_model_' + str(foldidx) +'.ckpt'
# best_acc = 0
# for epoch in range(FLAGS.epochs):
#     # epoch =  0
#     feed_dict = dict()
#     feed_dict.update({placeholders['features']: drug_feat})
#     feed_dict.update({placeholders['dropout']: FLAGS.dropout})
#     feed_dict.update({placeholders['net1_adj_norm_'+str(cellidx)] : d_net1_norm[cellidx] for cellidx in range(cellscount)})
#     _, avg_cost = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)
#     if epoch % 10 == 0:
#         print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost))
#     feed_dict.update({placeholders['dropout']: 0})
#     res = sess.run( model.reconstructions, feed_dict=feed_dict)

#     merged_preds = []
#     merged_labels = []
#     # cells_stats = np.zeros((cellscount, 6))
#     for cellidx in range(cellscount):
#         # cellidx = 0
#         preds_all = res[cellidx][ tuple( d_valid_edges[cellidx].T )].tolist()
#         # preds_all = [sigmoid(x) for x in preds_all]
#         preds_all_binary = [ 1 if x>=0.5 else 0 for x in preds_all ]
#         labels_all = d_valid_labels[cellidx]
#         merged_preds += preds_all
#         merged_labels += labels_all
#     #     #cal
#     #     roc_score = roc_auc_score(labels_all, preds_all)
#     #     precision, recall, _ = precision_recall_curve(labels_all, preds_all_binary)
#     #     auprc_score = auc(recall, precision)
#     #     accuracy = accuracy_score(preds_all_binary, labels_all)
#     #     f1 = f1_score(labels_all, preds_all_binary)
#     #     precision = precision_score(labels_all, preds_all_binary, zero_division=0)
#     #     recall = recall_score(labels_all, preds_all_binary)
#     #     cells_stats[cellidx] = [ roc_score, accuracy, auprc_score, f1, precision, recall ]
    
#     # ave_auc, ave_auprc, ave_acc, ave_f1, ave_precision, ave_recall = cells_stats.mean(axis=0)

#     merged_preds_binary = [1 if x >= 0.5 else 0 for x in merged_preds ]
#     merged_auc = roc_auc_score(merged_labels, merged_preds)
#     precision, recall, _ = precision_recall_curve(merged_labels,merged_preds_binary)
#     merged_auprc = auc(recall, precision)
#     merged_acc = accuracy_score(merged_preds_binary, merged_labels)
#     merged_f1 = f1_score(merged_labels, merged_preds_binary)
#     merged_precision = precision_score(merged_labels,merged_preds_binary, zero_division=0)
#     merged_recall = recall_score(merged_labels, merged_preds_binary)

#     if best_acc < merged_acc:
#         best_acc = merged_acc
#     # if best_acc < ave_avcc:
#     #     best_acc = ave_acc
#         saver.save(sess, best_model_file)

saver.restore(sess, best_model_file )

feed_dict = dict()
feed_dict.update({placeholders['features']: drug_feat})
feed_dict.update({placeholders['dropout']: FLAGS.dropout})
feed_dict.update({placeholders['net1_adj_norm_'+str(cellidx)] : d_net1_norm[cellidx] for cellidx in range(cellscount)})

##test predict
feed_dict.update({placeholders['dropout']: 0})
res = sess.run( model.reconstructions , feed_dict=feed_dict)

merged_preds = []
merged_labels = []
cells_stats = np.zeros((cellscount, 6))
for cellidx in range(cellscount):
    cellname = cellslist[cellidx]
    preds_all = res[cellidx][ tuple( d_test_edges[cellidx].T )].tolist()
    # preds_all = [sigmoid(x) for x in preds_all]
    preds_all_binary = [ 1 if x>=0.5 else 0 for x in preds_all ]
    labels_all = d_test_labels[cellidx]
    merged_preds += preds_all
    merged_labels += labels_all

    roc_score = roc_auc_score(labels_all, preds_all)
    precision, recall, _ = precision_recall_curve(labels_all, preds_all)
    auprc_score = auc(recall, precision)
    accuracy = accuracy_score(preds_all_binary, labels_all)
    f1 = f1_score(labels_all, preds_all_binary)
    precision = precision_score(labels_all, preds_all_binary, zero_division=0)
    recall = recall_score(labels_all, preds_all_binary)
    t = [ roc_score, accuracy, auprc_score, f1, precision, recall ]
    cells_stats[cellidx] = t

ave_auc, ave_auprc, ave_acc, ave_f1, ave_precision, ave_recall = cells_stats.mean(axis=0)

test_mean = cells_stats.mean(axis=0)
all_stats[foldidx] = test_mean

test_cell_stats = pd.DataFrame(cells_stats)
test_cell_stats.index = cellslist
test_std = test_cell_stats.std(axis=0)
test_cell_stats.loc['mean'] = test_mean
test_cell_stats.loc['std'] = test_std
test_cell_stats.to_csv(resultspath+ '/cell_stats_'+str(foldidx)+'.txt',sep='\t',header=None,index=True)

##get merged stats
merged_preds_binary = [ 1 if x >=0.5 else 0 for x in merged_preds ]
merged_auc = roc_auc_score(merged_labels, merged_preds)
precision, recall, _ = precision_recall_curve(merged_labels,merged_preds)
merged_auprc = auc(recall, precision)
merged_acc = accuracy_score(merged_preds_binary, merged_labels)
merged_f1 = f1_score(merged_labels, merged_preds_binary)
merged_precision = precision_score(merged_labels,merged_preds_binary, zero_division=0)
merged_recall = recall_score(merged_labels, merged_preds_binary)
merged_stats[foldidx] = [ merged_auc, merged_auprc, merged_acc, merged_f1, merged_precision, merged_recall ]
pd.DataFrame([ merged_auc, merged_auprc, merged_acc, merged_f1, merged_precision, merged_recall ]).to_csv(resultspath + '/stats_'+str(foldidx)+'.txt',sep='\t',header=None, index=None)

print('cv is over!')
##all stats
all_stats = pd.DataFrame(all_stats)
stats_mean = all_stats.mean(axis=0)
stats_std = all_stats.std(axis=0)
all_stats.loc['mean'] = stats_mean
all_stats.loc['std'] = stats_std
all_stats.to_csv(resultspath+'/all_stats.txt', sep='\t', header=None, index=True)

##all stats
merged_stats = pd.DataFrame(merged_stats)
stats_mean = merged_stats.mean(axis=0)
stats_std = merged_stats.std(axis=0)
merged_stats.loc['mean'] = stats_mean
merged_stats.loc['std'] = stats_std
merged_stats.to_csv(resultspath+'/merged_stats.txt', sep='\t', header=None, index=True)
