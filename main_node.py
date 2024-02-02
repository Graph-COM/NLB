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
import os

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
MODEL_PATH = args.model_path
NODE_TO_GPU = True
NLB_NODE = args.nlb_node
EVAL_BS = args.bs

EDGE_FEAT_LOAD_TO_GPU_PARTITION_SIZE = int(5e7)

assert(NUM_HOP < 2) # only up to 1 hop is supported
logger, get_checkpoint_path, get_ngh_store_path, get_self_rep_path, get_prev_raw_path, best_model_path, best_model_ngh_store_path, best_decoder_path, get_decoder_checkpoint_path, runtime_id = set_up_logger(args, sys_argv, 'node')


g_df = pd.read_csv('./DATA/{}/edges_with_node_labels.csv'.format(DATA))
g_df = g_df.rename(columns={'Unnamed: 0': 'idx', 'src': 'u', 'dst': 'i', 'time': 'ts'})
g_df.sort_values(by=['idx'], inplace=True)

src_label_l = g_df.src_label.values.astype(int)
tgt_label_l = g_df.tgt_label.values.astype(int)

src_l = g_df.u.values.astype(int)
tgt_l = g_df.i.values.astype(int)
max_idx = max(src_l.max(), tgt_l.max())

if DATA == 'GDELT':
  edge_feat_in_gpu = False
  edge_feats = torch.load('./DATA/GDELT/edge_features_remap.pt')
  node_feats = torch.load('./DATA/GDELT/node_features_remapped.pt')
  e_idx_l = np.arange(len(g_df)).astype(int)
elif DATA == 'MAG':
  edge_feats = None
  node_feats = torch.load('./DATA/MAG/node_features_remapped.pt')
  NODE_TO_GPU = False
  e_idx_l = np.arange(len(g_df)).astype(int)
else:
  e_idx_l = g_df.idx.values.astype(int)
  edge_feats = torch.load('./DATA/{}/edge_features_remap.pt'.format(DATA))
  node_feats = None
  edge_feat_in_gpu = True
  
e_idx_total = g_df.idx.values.astype(int)
if edge_feats is not None:
  if edge_feats.dtype == torch.bool:
    edge_feats = edge_feats.type(torch.float32)
  e_feat = edge_feats.numpy()
else:
  e_feat = None
if node_feats is not None:
  if node_feats.dtype == torch.bool:
    node_feats = node_feats.type(torch.float32)
  n_feat = node_feats.numpy()
  assert(n_feat.shape[0] == max_idx + 1)  # the nodes need to map one-to-one to the node feat matrix
else:
  n_feat = None

ts_l_total = g_df.ts.values
ts_l = ts_l_total
assert((ts_l == np.sort(ts_l)).all())

assert(np.unique(np.stack([src_l, tgt_l])).shape[0] == max_idx)  # all nodes except node 0 should appear and be compactly indexed
assert(0 not in np.unique(np.stack([src_l, tgt_l])).astype(int))  # all nodes except node 0 should appear and be compactly indexed

# split and pack the data by generating valid train/val/test mask according to the "mode"
val_time, test_time = list(np.quantile(ts_l, [0.70, 0.85]))
print(val_time, test_time)
transductive_auc = []
transductive_ap = []
transductive_f1 = []
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
    print(valid_train_flag.sum())
    print(valid_val_flag.sum())
    print(valid_test_flag.sum())
  else:
    raise NotImplementedError
  # split data according to the mask
  train_src_l, train_tgt_l, train_ts_l, train_e_idx_l, train_src_label_l, train_tgt_label_l = src_l[valid_train_flag], tgt_l[valid_train_flag], ts_l[valid_train_flag], e_idx_l[valid_train_flag], src_label_l[valid_train_flag], tgt_label_l[valid_train_flag]
  val_src_l, val_tgt_l, val_ts_l, val_e_idx_l, val_src_label_l, val_tgt_label_l = src_l[valid_val_flag], tgt_l[valid_val_flag], ts_l[valid_val_flag], e_idx_l[valid_val_flag], src_label_l[valid_val_flag], tgt_label_l[valid_val_flag]
  test_src_l, test_tgt_l, test_ts_l, test_e_idx_l, test_src_label_l, test_tgt_label_l = src_l[valid_test_flag], tgt_l[valid_test_flag], ts_l[valid_test_flag], e_idx_l[valid_test_flag], src_label_l[valid_test_flag], tgt_label_l[valid_test_flag]
  train_data = train_src_l, train_tgt_l, train_ts_l, train_e_idx_l, train_src_label_l, train_tgt_label_l
  val_data = val_src_l, val_tgt_l, val_ts_l, val_e_idx_l, val_src_label_l, val_tgt_label_l

  data = src_l, tgt_l, ts_l_total, e_idx_total, src_label_l, tgt_label_l

  train_val_data = (train_data, val_data)

  # model initialization
  device =  torch.device('cuda:{}'.format(str(args.gpu)) if torch.cuda.is_available() else 'cpu')
  
  if e_feat is not None:
    e_feat_dim = e_feat.shape[1]
  else:
    e_feat_dim = 0
  if n_feat is not None:
    feat_dim = n_feat.shape[1]
  else:
    feat_dim = 0
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
  n_hops=NUM_HOP, replace_prob=REPLACE_PROB, self_dim=SELF_DIM, device=device, nlb_node=NLB_NODE)
  

  state_dict = torch.load(MODEL_PATH,map_location='cpu')

  nlb.load_state_dict(state_dict)
  nlb.to(device)
  nlb.set_edge_features(edge_feats, e_idx_total, edge_feat_load_to_gpu_partition_size=EDGE_FEAT_LOAD_TO_GPU_PARTITION_SIZE, to_gpu=edge_feat_in_gpu)
  nlb.set_node_features(node_feats, to_gpu=NODE_TO_GPU)

  nlb.reset_store()
  optimizer = torch.optim.Adam(nlb.parameters(), lr=LEARNING_RATE)
  criterion = torch.nn.BCELoss()
  early_stopper = EarlyStopMonitor(tolerance=TOLERANCE)
  
  
  num_classes = max(src_label_l.max(), tgt_label_l.max()) + 1
  decoder_loss_criterion = torch.nn.CrossEntropyLoss()
  
  decoder = NodeClassificationModel(SELF_DIM, SELF_DIM, num_classes, get_checkpoint_path=get_decoder_checkpoint_path)
  decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)
  decoder = decoder.to(device)


  #####################################
  #### Get Embedding first before training
  # generate temporal embeddings
  # emb_file_name = DATA + '.pt'
  emb, rows, node_edges = generate_temporal_embeddings(data, nlb, args.mode, BATCH_SIZE, logger, model_dim, runtime_id, n_hop=NUM_HOP, seed=SEED)
  node_ts = rows[:, 0]
  train_node_flag = (node_ts <= val_time)
  val_node_flag = (node_ts <= test_time) * (node_ts > val_time)
  test_node_flag = node_ts > test_time

  train_emb, val_emb, test_emb = emb[train_node_flag], emb[val_node_flag], emb[test_node_flag]
  train_rows, val_rows, test_rows = rows[train_node_flag], rows[val_node_flag], rows[test_node_flag]
  train_node_edges, val_node_edges, test_node_edges = node_edges[train_node_flag], node_edges[val_node_flag], node_edges[test_node_flag]
  train_val_embs = (train_emb, val_emb)
  train_val_rows = (train_rows, val_rows)
  train_val_node_edges = (train_node_edges, val_node_edges)
  
  train_val_node_with_embed(train_val_embs, train_val_rows, train_val_node_edges, BATCH_SIZE, args.mode, NUM_EPOCH, criterion, optimizer, early_stopper, logger,
    decoder, decoder_optimizer, decoder_loss_criterion, device, seed=SEED, num_classes=num_classes, eval_bs=EVAL_BS)
  

  # final testing
  print("_*"*50)
  test_start = time.time()
  test_acc, test_ap, test_f1, test_auc, predict_total_time = eval_node_with_embed('test for {} nodes'.format(args.mode),
      decoder, test_emb, test_rows, test_node_edges, device=device, num_classes=num_classes, bs=EVAL_BS)
  test_end = time.time()
  logger.info('Test statistics: {} all nodes -- acc: {}, auc: {}, ap: {}, f1: {}, inference time: {}, test time: {}'.format(args.mode, test_acc, test_auc, test_ap, test_f1, predict_total_time, test_end - test_start))
  test_new_new_acc, test_new_new_ap, test_new_new_auc, test_new_old_acc, test_new_old_ap, test_new_old_auc = [-1]*6
  transductive_auc.append(test_auc)
  transductive_ap.append(test_ap)
  transductive_f1.append(test_f1)
  test_times.append(test_end - test_start)
  inference_times.append(predict_total_time)
  early_stoppers.append(early_stopper.best_epoch + 1)
  # save model
  logger.info('Saving NLB decoder ...')
  torch.save(decoder.state_dict(), best_decoder_path)
  logger.info('NLB decoder saved')

  # save one line result
  save_oneline_result('log/', args, [test_f1, 0, 0, 0, 0, 0, 0, 0, 0])
  # save walk_encodings_scores
  total_end = time.time()
  print("NLB experiment statistics:")
  if args.mode == "t":
    nlb_results(logger, transductive_auc, "transductive_auc")
    nlb_results(logger, transductive_ap, "transductive_ap")
    nlb_results(logger, transductive_f1, "transductive_f1")
  
  nlb_results(logger, test_times, "test_times")
  nlb_results(logger, inference_times, "inference_times")
  nlb_results(logger, early_stoppers, "early_stoppers")
  total_time.append(total_end - total_start)
  nlb_results(logger, total_time, "total_time")
