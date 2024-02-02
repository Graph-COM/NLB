import torch
import torch.nn as nn
import logging
import time
import numpy as np
import random
import math
from GAT import GAT
from torch.utils.data import WeightedRandomSampler
from parser import *
args, sys_argv = get_args()

class NLB(torch.nn.Module):
  def __init__(self, n_feat_dim, e_feat_dim, memory_dim, total_nodes, get_checkpoint_path=None, get_ngh_store_path=None, get_self_rep_path=None, get_prev_raw_path=None, time_dim=50, n_head=4, num_neighbors=['1', '32'],
      dropout=0.1, attn_dropout=0.1, verbosity=1, seed=1, n_hops=2, replace_prob=0.9, self_dim=100, device=None, nlb_node=False):
    super(NLB, self).__init__()
    self.logger = logging.getLogger(__name__)
    self.dropout = dropout
    self.feat_dim = n_feat_dim # node feature dimension
    self.e_feat_dim = e_feat_dim # edge feature dimension
    self.time_dim = time_dim  # default to be time feature dimension
    self.self_dim = self_dim
    # embedding layers and encoders
    self.nlb_node = nlb_node
    self.time_encoder = self.init_time_encoder() # fourier
    self.device = device

    
    # final projection layer
    self.out_layer = EdgePredictor(self.self_dim)
    self.get_checkpoint_path = get_checkpoint_path
    self.get_ngh_store_path = get_ngh_store_path
    self.get_self_rep_path = get_self_rep_path
    self.get_prev_raw_path = get_prev_raw_path
    self.num_neighbors = num_neighbors
    self.n_hops = n_hops
    self.ngh_id_idx = 0
    self.e_feat_idx = 2
    self.ts_raw_idx = 1

    self.memory_dim = memory_dim
    self.verbosity = verbosity
    
    self.attn_dim = self.feat_dim + self.self_dim + self.time_dim + self.e_feat_dim
    self.gat = GAT(1, [n_head], [self.attn_dim, self.self_dim], add_skip_connection=False, bias=False,
                 dropout=dropout, attn_dropout=attn_dropout, log_attention_weights=True)
    self.total_nodes = total_nodes
    self.replace_prob = replace_prob
    self.self_rep_linear = nn.Linear(self.self_dim + self.time_dim + self.e_feat_dim, self.self_dim, bias=False)
    self.self_aggregator = self.init_self_aggregator() # RNN
    self.merger = MergeLayer(self.self_dim, self_dim, self_dim, self_dim)
  
  def set_seed(self, seed):
    self.seed = seed

  def set_edge_features(self, edge_feats, train_e_idx_l=None, val_e_idx_l=None, test_e_idx_l=None, to_gpu=False, edge_feat_load_to_gpu_partition_size=5e7):
    self.n_edges = {"train": 0, "val": 0, "test": 0}
    if edge_feats is None:
      self.edge_feats_partitions = None
      return
    if to_gpu:
      self.edge_raw_embed = torch.nn.Embedding.from_pretrained(edge_feats, freeze=True)
      self.edge_raw_embed = self.edge_raw_embed.to(self.device)
      self.edge_feats_partitions = None
    else:
      self.edge_feats_partitions = {}
      self.phases = ['train', 'val', 'test']
      e_idxs = [train_e_idx_l, val_e_idx_l, test_e_idx_l]
      self.n_partitions = {}
      self.edge_feat_load_to_gpu_partition_size = edge_feat_load_to_gpu_partition_size
      for i in range(3):
        e_idx = e_idxs[i]
        if e_idx is None:
          break
        self.n_edges[self.phases[i]] = len(e_idx)
        self.n_partitions[self.phases[i]] = math.ceil(self.n_edges[self.phases[i]]/edge_feat_load_to_gpu_partition_size)
        edge_feat = edge_feats[e_idx]
        self.edge_feats_partitions[self.phases[i]] = [None for j in range(self.n_partitions[self.phases[i]])]
        for j in range(self.n_partitions[self.phases[i]]):
          self.edge_feats_partitions[self.phases[i]][j] = edge_feat[j * 
            self.edge_feat_load_to_gpu_partition_size: min(self.n_edges[self.phases[i]], 
              (j+1) * self.edge_feat_load_to_gpu_partition_size)]
      self.curr_edge_partition = -1
      self.phase = 'train'
  
  def reset_edge_feat_partition_to_cpu(self):
    if self.edge_feats_partitions is None:
      return
    for i in range(3):
      for j in range(self.n_partitions.get(self.phases[i], 0)):
        self.edge_feats_partitions[self.phases[i]][j] = self.edge_feats_partitions[self.phases[i]][j].cpu()
    self.curr_edge_partition = -1

  def check_idx_and_load_edge_feat_partition_to_gpu(self, idx=0):
    if self.edge_feats_partitions is None:
      return
    if idx >= (self.curr_edge_partition + 1) * self.edge_feat_load_to_gpu_partition_size:
      self.curr_edge_partition += 1
      if self.curr_edge_partition < self.n_partitions[self.phase]:
        if self.curr_edge_partition > 0:
          self.edge_feats_partitions[self.phase][self.curr_edge_partition - 1] = self.edge_feats_partitions[self.phase][self.curr_edge_partition - 1].cpu()
        self.edge_feats_partitions[self.phase][self.curr_edge_partition] = self.edge_feats_partitions[self.phase][self.curr_edge_partition].to(self.device)

  def set_node_features(self, node_feats, to_gpu=False):
    if node_feats is None:
      self.node_raw_embed = None
      return
    self.node_raw_embed = node_feats
    if to_gpu:
      self.node_raw_embed = self.node_raw_embed.to(self.device)
  
  def clear_store(self):
    self.neighborhood_store = None

  def reset_store(self, keep_self_rep=False):
    ngh_stores = []
    self.num_neighbors_stored = 0
    ngh_store_type = torch.float32 # by default use float
    if self.e_feat_dim == 0: # this is a special case for dataset MAG. The ID are too big to be represented by float
      ngh_store_type = torch.int32
    for i in self.num_neighbors:
      self.num_neighbors_stored += i
      max_e_idx = self.total_nodes * i
      ngh_store = torch.zeros(max_e_idx, self.memory_dim, dtype=ngh_store_type, device=self.device, requires_grad=False)
      ngh_stores.append(ngh_store)
    self.neighborhood_store = ngh_stores
    if keep_self_rep:
      return
    self.self_and_edge_rep = torch.zeros(self.total_nodes, self.self_dim*2 + self.e_feat_dim, device=self.device, requires_grad=False)
    self.prev_raw = torch.zeros(self.total_nodes, 2, dtype=torch.int32, device=self.device, requires_grad=False)

  def set_neighborhood_store(self, neighborhood_store):
    self.neighborhood_store = neighborhood_store

  def set_self_rep(self, self_and_edge_rep, prev_raw):
    self.self_and_edge_rep = self_and_edge_rep
    self.prev_raw = prev_raw

  def set_device(self, device):
    self.device = device

  def log_time(self, desc, start, end):
    if self.verbosity > 1:
      self.logger.info('{} for the minibatch, time eclipsed: {} seconds'.format(desc, str(end-start)))
  
  def position_bits(self, bs, hop):
    return torch.ones(bs * self.num_neighbors[hop], device=self.device, dtype=torch.int32) * hop

  def fetch_edge_feat(self, e_idx_th):
    if self.e_feat_dim == 0:
      return None
    if self.edge_feats_partitions != None:
      e_id_remap = e_idx_th - self.curr_edge_partition * self.edge_feat_load_to_gpu_partition_size
      e_feat = self.edge_feats_partitions[self.phase][self.curr_edge_partition][e_id_remap]
    else:
      e_feat = self.edge_raw_embed(e_idx_th)
    return e_feat

  def contrast(self, src_l_cut, tgt_l_cut, bad_l_cut, cut_time_l, e_idx_l=None):
    predict_start = time.time()
    batch_size = len(src_l_cut)
    
    updated_embeddings, updated_mem_h0, updated_mem_h1, _ = self.updated_embeddings(batch_size, src_l_cut, tgt_l_cut, bad_l_cut, cut_time_l)
    p_score, n_score = self.forward(updated_embeddings)
    predict_end = time.time()
    predict_time = predict_end - predict_start
    e_feat = self.fetch_edge_feat(e_idx_l)
    self.update_memory(src_l_cut, tgt_l_cut, e_feat, cut_time_l, updated_embeddings, updated_mem_h1, batch_size)
    return p_score, n_score, predict_time

  def updated_embeddings(self, batch_size, src_th, tgt_th, bad_th, cut_time_th, e_feat=None):
    idx_th = torch.cat((src_th, tgt_th, bad_th), 0)
    batch_idx = torch.arange(batch_size * 3, device=self.device)
    self.neighborhood_store[0][idx_th, 0] = idx_th.type(self.neighborhood_store[0].dtype)

    ngh_id, updated_mem_h0 = self.batch_fetch_temporal_neighbors(idx_th, cut_time_th.repeat(3), hop=0)
    feature_dim = self.memory_dim
    updated_mem = updated_mem_h0.view(3 * batch_size, self.num_neighbors[0], -1)
    ngh_id = ngh_id.view(3 * batch_size, self.num_neighbors[0])
    updated_mem_h1 = None
    if self.n_hops > 0:
      h1_pos_bit = self.position_bits(3 * batch_size, hop=1)
      ngh_id_h1,updated_mem_h1 = self.batch_fetch_temporal_neighbors(idx_th, cut_time_th.repeat(3), hop=1)      
      ngh_id = torch.cat((ngh_id, ngh_id_h1.view(3 * batch_size, self.num_neighbors[1])), -1)
      updated_mem = torch.cat((
        updated_mem,
        updated_mem_h1.view(3 * batch_size, self.num_neighbors[1], -1)), 1)
      if not torch.is_floating_point(self.neighborhood_store[0]):
        updated_mem_h1 = torch.cat((ngh_id_h1.unsqueeze(1), updated_mem_h1[:, self.ngh_id_idx + 1:].int()), -1)
    if self.n_hops > 1:
      raise NotImplementedError
    updated_mem = updated_mem.view(-1, feature_dim + self.time_dim)
    e_and_t_feats = updated_mem[:, self.e_feat_idx:]
    ngh_id = ngh_id.flatten().long()
    ngh_exists = torch.nonzero(ngh_id, as_tuple=True)[0]
    ngh_count = torch.count_nonzero(ngh_id.view(3 * batch_size, -1), dim=-1)
    if self.node_raw_embed is not None:
      storage_device = self.node_raw_embed.device
    else:
      storage_device = self.self_and_edge_rep.device
    ngh_id_for_fetch = ngh_id.index_select(0, ngh_exists)
    
    updated_mem = updated_mem.index_select(0, ngh_exists)
    e_and_t_feats = e_and_t_feats.index_select(0, ngh_exists)

    sparse_idx = torch.repeat_interleave(batch_idx, ngh_count)

    if self.node_raw_embed is not None:
      node_features = self.node_raw_embed[ngh_id_for_fetch]
      if not node_features.is_cuda:
        node_features = node_features.to(self.device, non_blocking=True)
    

    ngh_self_rep = self.updated_self_rep(ngh_id)
    self_reps = ngh_self_rep.index_select(0, batch_idx * self.num_neighbors_stored)


    ngh_self_rep = ngh_self_rep.index_select(0, ngh_exists)
    if self.node_raw_embed is not None:
      hidden_states = torch.cat((node_features, ngh_self_rep, e_and_t_feats), -1)
    else:
      hidden_states = torch.cat((ngh_self_rep, e_and_t_feats), -1)
      node_features = None
    ngh_and_batch_id = torch.cat((ngh_id_for_fetch.unsqueeze(1), sparse_idx.unsqueeze(1)), -1)
    
    embeddings = self.aggregate(ngh_and_batch_id, hidden_states, batch_size, self_reps)
   
    return embeddings, updated_mem_h0, updated_mem_h1, node_features

  def updated_self_rep(self, node_id):
    node_id_l = node_id
    self_store = self.prev_raw[node_id_l]
    oppo_id = self_store[:, self.ngh_id_idx].long()
    ts_raw = self_store[:,self.ts_raw_idx]
    
    ts_feat = self.time_encoder(ts_raw.float())
    prev_self_and_edge_rep = self.self_and_edge_rep[node_id_l].to(self.device, non_blocking=True)

    prev_self_rep = prev_self_and_edge_rep[:, :self.self_dim]

    updated_self_rep = self.self_aggregator(torch.cat((prev_self_and_edge_rep, ts_feat), -1), prev_self_rep) # self.self_rep_linear(
    return updated_self_rep

  # this is used for building up the downsampled temporal neighbors for the inductive learning
  def update_cache_only(self, src_th, tgt_th, cut_time_th, e_idx_l=None):
    batch_size = len(src_th)
    ori_idx = torch.cat((src_th, tgt_th), 0)
    cut_time_th = cut_time_th.repeat(2)
    opp_th = torch.cat((tgt_th, src_th), 0)
    e_feat = self.fetch_edge_feat(e_idx_l)
    t_and_e_feat = cut_time_th.unsqueeze(1).int()
    if e_feat is not None:
        e_feat = e_feat.repeat(2, 1)
        t_and_e_feat = torch.cat((t_and_e_feat, e_feat), -1)
      
    # Update neighbors
    batch_id = torch.arange(batch_size * 2, device=self.device)
    if self.n_hops > 0:
      # Update second hop neighbors
      if self.n_hops > 1:
        raise NotImplementedError

      candidate_temporal_neighbors = torch.cat((opp_th.unsqueeze(1).int(), t_and_e_feat), -1)
      self.update_temporal_neighbors(ori_idx, candidate_temporal_neighbors, 1)
    # Update self
    candidate_temporal_neighbors = torch.cat((ori_idx.unsqueeze(1).int(), t_and_e_feat), -1)
    self.update_temporal_neighbors(ori_idx, candidate_temporal_neighbors, 0)

  def update_memory(self, src_th, tgt_th, e_feat, cut_time_th, updated_mem_h0, updated_mem_h1, batch_size):
    ori_idx = torch.cat((src_th, tgt_th), 0)
    cut_time_th = cut_time_th.repeat(2)
    opp_th = torch.cat((tgt_th, src_th), 0)
    self.prev_raw[ori_idx] = torch.cat((opp_th.unsqueeze(1).int(), cut_time_th.unsqueeze(1).int()), dim = 1)

    t_and_e_feat = cut_time_th.unsqueeze(1).int()
    if e_feat is not None:
      e_feat = e_feat.repeat(2, 1)
      t_and_e_feat = torch.cat((t_and_e_feat, e_feat), -1)
    
    # Update neighbors
    batch_id = torch.arange(batch_size * 2, device=self.device)
    if self.n_hops > 0:
      updated_mem_h1 = updated_mem_h1.detach()[:2 * batch_size * self.num_neighbors[1]]
      # Update second hop neighbors
      if self.n_hops > 1:
        raise NotImplementedError

      candidate_temporal_neighbors = torch.cat((opp_th.unsqueeze(1).int(), t_and_e_feat), -1)
      self.update_temporal_neighbors(ori_idx, candidate_temporal_neighbors, 1)
    # Update self
    updated_mem_h0 = updated_mem_h0.detach()[:batch_size * self.num_neighbors[0] * 2]
    reversed_reps = torch.cat((updated_mem_h0[batch_size:2*batch_size], updated_mem_h0[:batch_size]), 0)
    self_and_edge_reps = torch.cat((updated_mem_h0[:2*batch_size], reversed_reps), -1)
    if e_feat is not None:
      self_and_edge_reps = torch.cat((self_and_edge_reps, e_feat), dim = 1)
    if self.node_raw_embed is not None:
      storage_device = self.node_raw_embed.device
    else:
      storage_device = self.self_and_edge_rep.device
    self_and_edge_reps = self_and_edge_reps.to(storage_device)
    self.self_and_edge_rep[ori_idx[:2*batch_size]] = self_and_edge_reps[:2*batch_size]
    candidate_temporal_neighbors = torch.cat((ori_idx.unsqueeze(1).int(), t_and_e_feat), -1)
    self.update_temporal_neighbors(ori_idx, candidate_temporal_neighbors, 0)

  def temporal_neighbor_hash(self, ngh_id, hop):
    ngh_id = ngh_id.long()
    if self.nlb_node:
      return ((ngh_id * (self.seed % 100) + ngh_id * ngh_id * ((self.seed % 100) + 1)) % self.num_neighbors[hop]).long()
    return ((ngh_id * (int(random.random() * 100)) + ngh_id * ngh_id * (int(random.random() * 100) + 1)) % self.num_neighbors[hop]).long()

  def update_temporal_neighbors(self, self_id, candidate_temporal_neighbors, hop):
    if self.num_neighbors[hop] == 0:
      return
    ngh_id = candidate_temporal_neighbors[:, self.ngh_id_idx]
    idx = self_id * self.num_neighbors[hop] + self.temporal_neighbor_hash(ngh_id, hop)
    is_occupied = torch.logical_and(self.neighborhood_store[hop][idx,self.ngh_id_idx] != 0, self.neighborhood_store[hop][idx,self.ngh_id_idx] != ngh_id)
    should_replace =  (is_occupied * torch.rand(is_occupied.shape[0], device=self.device)) < self.replace_prob
    idx *= should_replace
    idx *= ngh_id != 0
    self.neighborhood_store[hop][idx] = candidate_temporal_neighbors

  def batch_fetch_temporal_neighbors(self, ori_idx, curr_time, hop):
    ngh = self.neighborhood_store[hop].view(self.total_nodes, self.num_neighbors[hop], self.memory_dim)[ori_idx].view(ori_idx.shape[0] * (self.num_neighbors[hop]), self.memory_dim)

    curr_time = curr_time.repeat_interleave(self.num_neighbors[hop])
    ngh_id = ngh[:,self.ngh_id_idx]
    ngh_ts_raw = ngh[:,self.ts_raw_idx]
    ts_feat = self.time_encoder(ngh_ts_raw.float())

    msk = ngh_ts_raw < curr_time
    ngh_info = torch.cat((ngh, ts_feat), -1)
    return ngh_id, ngh_info# * msk.unsqueeze(1).repeat(1, ngh_info.shape[1])


  def forward(self, embeddings):
    return self.out_layer(embeddings)


  def aggregate(self, ngh_and_batch_id, feat, bs, self_rep=None):
    edge_idx = ngh_and_batch_id.T
    embed, _, attn_score = self.gat((feat, edge_idx, 3*bs))
    if self_rep is not None:
      embed = self.merger(embed, self_rep)
    return embed

  def init_time_encoder(self):
    return TimeEncode(self.time_dim)

  def init_self_aggregator(self):
    return FeatureEncoderGRU(self.self_dim*2 + self.time_dim + self.e_feat_dim, self.self_dim, self.dropout)

class FeatureEncoderGRU(torch.nn.Module):
  def __init__(self, input_dim, output_dim, dropout_p=0.0):
    super(FeatureEncoderGRU, self).__init__()
    self.gru = nn.GRUCell(input_dim, output_dim)
    self.dropout = nn.Dropout(dropout_p)
    self.output_dim = output_dim

  def forward(self, input_features, hidden_state, use_dropout=False):
    encoded_features = self.gru(input_features, hidden_state)
    # if use_dropout:
    encoded_features = self.dropout(encoded_features)
    
    return encoded_features


# class TimeEncode(torch.nn.Module):

#   def __init__(self, dim):
#     super(TimeEncode, self).__init__()
#     self.dim = dim
#     self.w = torch.nn.Linear(1, dim)
#     self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dim, dtype=np.float32))).reshape(dim, -1))
#     self.w.bias = torch.nn.Parameter(torch.zeros(dim))

#   def forward(self, t):
#     output = torch.cos(self.w(t.reshape((-1, 1))))
#     return output

class TimeEncode(torch.nn.Module):
  def __init__(self, expand_dim, factor=5):
    super(TimeEncode, self).__init__()

    self.time_dim = expand_dim
    self.factor = factor
    self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim))).float())
    self.phase = torch.nn.Parameter(torch.zeros(self.time_dim).float())


  def forward(self, ts):
    # ts: [N, 1]
    batch_size = ts.size(0)

    ts = ts.view(batch_size, 1)  # [N, 1]
    map_ts = ts * self.basis_freq.view(1, -1) # [N, time_dim]
    map_ts += self.phase.view(1, -1) # [N, time_dim]
    harmonic = torch.cos(map_ts)

    # return torch.zeros_like(ts)
    return harmonic #self.dense(harmonic)
class MergeLayer(torch.nn.Module):
  def __init__(self, dim1, dim2, dim3, dim4):
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1 + dim2, dim4)
    self.fc2 = torch.nn.Linear(dim3, dim4)
    self.act = torch.nn.ReLU()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)


  def forward(self, x1, x2):
    x = torch.cat([x1, x2], dim=-1)
    h = self.act(self.fc1(x))
    z = self.fc2(h)
    return z

class OutLayer(torch.nn.Module):
  def __init__(self, dim1, dim2, dim3):
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1, dim2)
    self.fc2 = torch.nn.Linear(dim2, dim3)
    self.act = torch.nn.ReLU()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x):
    h = self.act(self.fc1(x))
    z = self.fc2(h)
    return z
  

class EdgePredictor(torch.nn.Module):

  def __init__(self, dim_in):
    super(EdgePredictor, self).__init__()
    self.dim_in = dim_in
    self.src_fc = torch.nn.Linear(dim_in, dim_in)
    self.dst_fc = torch.nn.Linear(dim_in, dim_in)
    self.out_fc = torch.nn.Linear(dim_in, 1)

  def forward(self, h, neg_samples=1):
    num_edge = h.shape[0] // (neg_samples + 2)
    h_src = self.src_fc(h[:num_edge])
    h_pos_dst = self.dst_fc(h[num_edge:2 * num_edge])
    h_neg_dst = self.dst_fc(h[2 * num_edge:])
    h_pos_edge = torch.nn.functional.relu(h_src + h_pos_dst)
    h_neg_edge = torch.nn.functional.relu(h_src.tile(neg_samples, 1) + h_neg_dst)
    return self.out_fc(h_pos_edge).squeeze_(dim=-1), self.out_fc(h_neg_edge).squeeze_(dim=-1)

