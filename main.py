import pandas as pd
from log import *
from parser import *
from eval import *
from utils import *
from train import *
from module import NLB
import resource
import torch.nn as nn
import statistics
import torch
import os
import numpy as np

os.environ["OMP_NUM_THREADS"] = '32'

torch.set_num_threads(32)

args, sys_argv = get_args()

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_EPOCH = args.n_epoch
ATTN_NUM_HEADS = args.attn_n_head
DROP_OUT = args.drop_out
ATTN_DROPOUT = args.attn_dropout
DATA = args.data
NUM_HOP = args.n_hop
LEARNING_RATE = args.lr
TOLERANCE = args.tolerance
VERBOSITY = args.verbosity
SEED = args.seed
TIME_DIM = args.time_dim
REPLACE_PROB = args.replace_prob
SELF_DIM = args.self_dim
NODE_TO_GPU = True
NLB_NODE = args.nlb_node
EVAL_BS = args.bs

EDGE_FEAT_LOAD_TO_GPU_PARTITION_SIZE = int(5e7)

assert(NUM_HOP < 2) # only up to 1 hop is supported
logger, get_checkpoint_path, get_ngh_store_path, get_self_rep_path, get_prev_raw_path, best_model_path, best_model_ngh_store_path, best_decoder_path, get_decoder_checkpoint_path, runtime_id = set_up_logger(args, sys_argv)


# Load data and sanity check
g_df = pd.read_csv('./DATA/{}/edges.csv'.format(DATA))
g_df = g_df.rename(columns={'Unnamed: 0': 'idx', 'src': 'u', 'dst': 'i', 'time': 'ts'})

ts_l = g_df.ts.values
assert((ts_l == np.sort(ts_l)).all())
src_l = g_df.u.values.astype(int)
tgt_l = g_df.i.values.astype(int)
max_idx = max(src_l.max(), tgt_l.max())
min_idx = min(src_l.min(), tgt_l.min())



if DATA == 'GDELT':
  edge_feat_in_gpu = False
  edge_feats = torch.load('./DATA/GDELT/edge_features.pt')
  edge_feats.requires_grad = False
  node_feats = torch.load('./DATA/GDELT/node_features.pt')
  node_feats.requires_grad = False
elif DATA == 'MAG':
  edge_feats = None
  NODE_TO_GPU = False
  node_feats = torch.load('./DATA/MAG/node_features.pt')
  node_feats.requires_grad = False
elif DATA == 'WIKI' or DATA == 'REDDIT':
  edge_feats = torch.load('./DATA/{}/edge_features.pt'.format(DATA))
  edge_feats.requires_grad = False
  edge_feat_in_gpu = True
  NODE_TO_GPU = True
  node_feats = None
else:
  node_feats = None
  edge_feats = None
  edge_feat_in_gpu = True

if min_idx == 0:
  # move the node ids to start from 1
  src_l = src_l + 1
  tgt_l = tgt_l + 1
  g_df.u = g_df.u + 1
  g_df.i = g_df.i + 1
  if node_feats is not None:
    padding_n = torch.zeros_like(node_feats[:1])
    node_feats = torch.cat((padding_n, node_feats), 0)
  max_idx = max(src_l.max(), tgt_l.max())
  min_idx = min(src_l.min(), tgt_l.min())

if edge_feats is not None:
  if edge_feats.dtype == torch.bool:
    edge_feats = edge_feats.type(torch.float32)
if node_feats is not None:
  if node_feats.dtype == torch.bool:
    node_feats = node_feats.type(torch.float32)
  n_feat = node_feats.numpy()

  assert(n_feat.shape[0] == max_idx + 1)  # the nodes need to map one-to-one to the node feat matrix
else:
  n_feat = None

e_idx_l = g_df.idx.values.astype(int)
label_l = g_df.ts.values
ts_l = g_df.ts.values



assert(np.unique(np.stack([src_l, tgt_l])).shape[0] == max_idx)  # all nodes except node 0 should appear and be compactly indexed

# split and pack the data by generating valid train/val/test mask according to the "mode"
val_time, test_time = list(np.quantile(g_df.ts, [0.70, 0.85]))
transductive_auc = []
transductive_ap = []
inductive_auc = []
inductive_ap = []
test_times = []
inference_times = []
early_stoppers = []
total_time = []

for run in range(args.run):
  SEED += run
  set_random_seed(SEED)
  if args.mode == 't':
    logger.info('Transductive training...')
    valid_train_flag = (ts_l <= val_time)
    valid_val_flag = (ts_l <= test_time) * (ts_l > val_time)
    valid_test_flag = ts_l > test_time

  else:
    assert(args.mode == 'i')
    logger.info('Inductive training...')
    # pick some nodes to mask (i.e. reserved for testing) for inductive setting
    total_node_set = set(np.unique(np.hstack([src_l,tgt_l])))
    num_total_unique_nodes = len(total_node_set)
    mask_node_set = set(random.sample(set(src_l[ts_l > val_time]).union(set(tgt_l[ts_l > val_time])), int(0.1 * num_total_unique_nodes)))
    mask_src_flag = g_df.u.map(lambda x: x in mask_node_set).values
    mask_tgt_flag = g_df.i.map(lambda x: x in mask_node_set).values
    none_mask_node_flag = (1 - mask_src_flag) * (1 - mask_tgt_flag)
    valid_train_flag = (ts_l <= val_time) * (none_mask_node_flag > 0.5)
    valid_val_flag = (ts_l <= test_time) * (ts_l > val_time) * (none_mask_node_flag > 0.5)  # both train and val edges can not contain any masked nodes
    all_train_val_flag = ts_l <= test_time
    inductive_train_val_flag = (ts_l <= test_time) * (none_mask_node_flag <= 0.5)
    valid_test_flag = (ts_l > test_time) * (none_mask_node_flag < 0.5)  # test edges must contain at least one masked node
    logger.info('Sampled {} nodes (10 %) which are masked in training and reserved for testing...'.format(len(mask_node_set)))
    edge_feats_inductive = None
    if edge_feats is not None:
      edge_feats_inductive = edge_feats[valid_train_flag + valid_val_flag + valid_test_flag]
  # split data according to the mask
  train_src_l, train_tgt_l, train_ts_l, train_e_idx_l, train_label_l = src_l[valid_train_flag], tgt_l[valid_train_flag], ts_l[valid_train_flag], e_idx_l[valid_train_flag], label_l[valid_train_flag]
  val_src_l, val_tgt_l, val_ts_l, val_e_idx_l, val_label_l = src_l[valid_val_flag], tgt_l[valid_val_flag], ts_l[valid_val_flag], e_idx_l[valid_val_flag], label_l[valid_val_flag]
  test_src_l, test_tgt_l, test_ts_l, test_e_idx_l, test_label_l = src_l[valid_test_flag], tgt_l[valid_test_flag], ts_l[valid_test_flag], e_idx_l[valid_test_flag], label_l[valid_test_flag]
  all_nodes = np.unique(np.hstack([train_src_l, train_tgt_l, val_src_l, val_tgt_l, test_src_l, test_tgt_l]))

  if args.mode == 'i':
    e_idx_l_inductive = np.arange(len(train_src_l) + len(val_src_l) + len(test_src_l))
    train_e_idx_l = e_idx_l_inductive[:len(train_src_l)]
    val_e_idx_l = e_idx_l_inductive[len(train_src_l):-len(test_src_l)]
    test_e_idx_l = e_idx_l_inductive[-len(test_src_l):]

    all_train_val_src_l, all_train_val_tgt_l, all_train_val_ts_l, all_train_val_e_idx_l, all_train_val_label_l = src_l[all_train_val_flag], tgt_l[all_train_val_flag], ts_l[all_train_val_flag], e_idx_l[all_train_val_flag], label_l[all_train_val_flag]
    inductive_train_val_src_l, inductive_train_val_tgt_l, inductive_train_val_ts_l, inductive_train_val_e_idx_l, inductive_train_val_label_l = src_l[inductive_train_val_flag], tgt_l[inductive_train_val_flag], ts_l[inductive_train_val_flag], e_idx_l[inductive_train_val_flag], label_l[inductive_train_val_flag]
  train_data = train_src_l, train_tgt_l, train_ts_l, train_e_idx_l, train_label_l
  val_data = val_src_l, val_tgt_l, val_ts_l, val_e_idx_l, val_label_l
  train_val_data = (train_data, val_data)

  # model initialization
  device =  torch.device('cuda:{}'.format(str(args.gpu)) if torch.cuda.is_available() else 'cpu')
  if n_feat is not None:
    feat_dim = n_feat.shape[1]
  else:
    feat_dim = 0
  if edge_feats is not None:
    e_feat_dim = edge_feats.shape[1]
  else:
    e_feat_dim = 0

  time_dim = TIME_DIM
  model_dim = feat_dim + e_feat_dim + time_dim
  hidden_dim = e_feat_dim + time_dim
  num_raw = 2 + e_feat_dim
  memory_dim = num_raw
  num_neighbors = [1]
  for i in range(NUM_HOP):
    num_neighbors.extend([int(NUM_NEIGHBORS[i])])

  total_start = time.time()
  nlb = NLB(feat_dim, e_feat_dim, memory_dim, max_idx + 1, time_dim=TIME_DIM, n_head=ATTN_NUM_HEADS, num_neighbors=num_neighbors, dropout=DROP_OUT, attn_dropout=ATTN_DROPOUT,
    get_checkpoint_path=get_checkpoint_path, get_ngh_store_path=get_ngh_store_path, get_self_rep_path=get_self_rep_path, get_prev_raw_path=get_prev_raw_path, verbosity=VERBOSITY,
  n_hops=NUM_HOP, replace_prob=REPLACE_PROB, self_dim=SELF_DIM, device=device, nlb_node=NLB_NODE, data_name=DATA)
  nlb.to(device)
  if args.mode == 'i':
    nlb.set_edge_features(edge_feats_inductive, train_e_idx_l, val_e_idx_l, test_e_idx_l, edge_feat_load_to_gpu_partition_size=EDGE_FEAT_LOAD_TO_GPU_PARTITION_SIZE, to_gpu=edge_feat_in_gpu)
  else:
    nlb.set_edge_features(edge_feats, train_e_idx_l, val_e_idx_l, test_e_idx_l, edge_feat_load_to_gpu_partition_size=EDGE_FEAT_LOAD_TO_GPU_PARTITION_SIZE, to_gpu=edge_feat_in_gpu)

  nlb.set_node_features(node_feats, to_gpu=NODE_TO_GPU)
 
  nlb.reset_store()

  optimizer = torch.optim.Adam(nlb.parameters(), lr=LEARNING_RATE)
  criterion = torch.nn.BCEWithLogitsLoss()
  early_stopper = EarlyStopMonitor(tolerance=TOLERANCE)

  # start train and val phases
  train_val(train_val_data, nlb, args.mode, BATCH_SIZE, NUM_EPOCH, criterion, optimizer, early_stopper, logger, model_dim, n_hop=NUM_HOP, seed=SEED, eval_bs=EVAL_BS, all_nodes=all_nodes)

  # final testing
  print("_*"*50)
  if args.mode == 'i':
    nlb.reset_store(keep_self_rep=True)
    nlb.set_edge_features(edge_feats, all_train_val_e_idx_l, None, None, edge_feat_load_to_gpu_partition_size=EDGE_FEAT_LOAD_TO_GPU_PARTITION_SIZE, to_gpu=edge_feat_in_gpu)
  #   # No more training, but need to load all previous interactions into the hash table for the final test
    build_history_interactions('rebuild for {} nodes'.format(args.mode), nlb, all_train_val_src_l, all_train_val_tgt_l, all_train_val_ts_l, all_train_val_label_l, all_train_val_e_idx_l, bs=EVAL_BS, device=device)
    nlb.set_edge_features(edge_feats_inductive, train_e_idx_l, val_e_idx_l, test_e_idx_l, edge_feat_load_to_gpu_partition_size=EDGE_FEAT_LOAD_TO_GPU_PARTITION_SIZE, to_gpu=edge_feat_in_gpu)
  test_start = time.time()
  test_acc, test_ap, test_f1, test_auc, predict_total_time = eval_one_epoch('test for {} nodes'.format(args.mode), nlb, test_src_l, test_tgt_l, test_ts_l, test_label_l, test_e_idx_l, bs=EVAL_BS, phase='test', mode=args.mode, device=device, all_nodes=all_nodes, eval_neg_samples=args.eval_neg_samples)
  test_end = time.time()
  logger.info('Test statistics: {} all nodes -- acc: {}, auc: {}, ap: {}, inference time: {}, total test time: {}'.format(args.mode, test_acc, test_auc, test_ap, predict_total_time, test_end - test_start))
  test_new_new_acc, test_new_new_ap, test_new_new_auc, test_new_old_acc, test_new_old_ap, test_new_old_auc = [-1]*6
  if args.mode == 'i':
    inductive_auc.append(test_auc)
    inductive_ap.append(test_ap)
  else:
    transductive_auc.append(test_auc)
    transductive_ap.append(test_ap)
  test_times.append(test_end - test_start)
  inference_times.append(predict_total_time)
  early_stoppers.append(early_stopper.best_epoch + 1)
  # save model
  logger.info('Saving NLB model ...')
  torch.save(nlb.state_dict(), best_model_path[:-4] + str(run) + '.pth')
  logger.info('NLB model saved')

  # save one line result
  save_oneline_result('log/', args, [test_acc, test_auc, test_ap, test_new_new_acc, test_new_new_ap, test_new_new_auc, test_new_old_acc, test_new_old_ap, test_new_old_auc])
  total_end = time.time()
  print("NLB experiment statistics:")
  if args.mode == "t":
    nlb_results(logger, transductive_auc, "transductive_auc")
    nlb_results(logger, transductive_ap, "transductive_ap")
  else:
    nlb_results(logger, inductive_auc, "inductive_auc")
    nlb_results(logger, inductive_ap, "inductive_ap")
  
  nlb_results(logger, test_times, "test_times")
  nlb_results(logger, inference_times, "inference_times")
  nlb_results(logger, early_stoppers, "early_stoppers")
  total_time.append(total_end - total_start)
  nlb_results(logger, total_time, "total_time")
