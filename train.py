import torch
import numpy as np
from tqdm import tqdm
import math
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from eval import *
import logging
import random
import time
from utils import *
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib.ticker').disabled = True
import os
from pyJoules.handler.csv_handler import CSVHandler
from pyJoules.energy_meter import measure_energy
from parser import *

args, sys_argv = get_args()

csv_handler = CSVHandler('{}_{}_energy.csv'.format(args.data, args.nlb_node))

@measure_energy(handler=csv_handler)
def train_one_epoch(model, seed, seeds, logger, epoch, num_batch, bs, num_instance, train_e_idx_l, idx_list,
  train_src_l, train_tgt_l, train_bad_l, train_ts_l, optimizer, device, criterion, train_time, n_hop,
  mode, val_src_l, val_tgt_l, val_ts_l, val_src_label_l, val_e_idx_l, eval_bs, early_stopper, all_nodes):
  train_start = time.time()
  model.set_seed(seed)
  set_random_seed(seed)
  seeds.append(seed)
  model.clear_store()
  model.reset_store()
  
  acc, ap, f1, auc, m_loss = [], [], [], [], []
  logger.info('start {} epoch'.format(epoch))
  model.reset_edge_feat_partition_to_cpu()
  model.phase = 'train'
  should_break = False

  for k in tqdm(range(num_batch)):
    # generate training mini-batch
    s_idx = k * bs
    e_idx = min(num_instance, s_idx + bs)
    if s_idx == e_idx:
      continue
    e_start = train_e_idx_l[s_idx]
    model.check_idx_and_load_edge_feat_partition_to_gpu(e_start)
    batch_idx = idx_list[s_idx:e_idx] # shuffle training samples for each batch
    # np.random.shuffle(batch_idx)
    src_l_cut, tgt_l_cut, bad_l_cut = train_src_l[batch_idx].to(dtype=torch.long, device=device, non_blocking=True), train_tgt_l[batch_idx].to(dtype=torch.long, device=device, non_blocking=True), train_bad_l[batch_idx].to(dtype=torch.long, device=device, non_blocking=True)
    ts_l_cut = train_ts_l[batch_idx].to(dtype=torch.float, device=device, non_blocking=True)
    e_l_cut = train_e_idx_l[batch_idx].to(dtype=torch.long, device=device, non_blocking=True)
    size = len(src_l_cut)

    # feed in the data and learn from error
    optimizer.zero_grad()
    model.train()
    pos_prob, neg_prob, predict_time = model.contrast(src_l_cut, tgt_l_cut, bad_l_cut, ts_l_cut, e_l_cut)   # the core training code
    pos_label = torch.ones(size, dtype=torch.float, device=device, requires_grad=False)
    neg_label = torch.zeros(size, dtype=torch.float, device=device, requires_grad=False)

    loss = criterion(pos_prob, pos_label) + criterion(neg_prob, neg_label)
    loss.backward()
    optimizer.step()
    # collect training results
    m_loss.append(loss.item())
  train_end = time.time()
  train_time.append(train_end - train_start)
  nlb_results(logger, train_time, "train_time")
  # validation phase use all information
  val_start = time.time()
  val_acc, val_ap, val_f1, val_auc, predict_total_time = eval_one_epoch('val for {} nodes'.format(mode), model, val_src_l,
                            val_tgt_l, val_ts_l, val_src_label_l, val_e_idx_l, bs=eval_bs, phase="val", mode=mode, device=device, all_nodes=all_nodes)
  val_end = time.time()
  logger.info('epoch: {}:'.format(epoch))
  logger.info('epoch mean loss: {}'.format(np.mean(m_loss)))
  logger.info('train acc: {}, val acc: {}'.format(np.mean(acc), val_acc))
  logger.info('train auc: {}, val auc: {}'.format(np.mean(auc), val_auc))
  logger.info('train ap: {}, val ap: {}'.format(np.mean(ap), val_ap))
  logger.info('train time: {}, val time: {}'.format(train_end - train_start, predict_total_time))
  # early stop check and checkpoint saving
  if early_stopper.early_stop_check(val_ap):
    logger.info('No improvment over {} epochs, stop training'.format(early_stopper.max_round))
    logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
    best_checkpoint_path = model.get_checkpoint_path(early_stopper.best_epoch)
    model.load_state_dict(torch.load(best_checkpoint_path))
    best_ngh_store = []
    model.clear_store()
    for i in range(n_hop + 1):
      best_ngh_store_path = model.get_ngh_store_path(early_stopper.best_epoch, i)
      best_ngh_store.append(torch.load(best_ngh_store_path))
    model.set_neighborhood_store(best_ngh_store)
    best_self_rep_path = model.get_self_rep_path(early_stopper.best_epoch)
    best_prev_raw_path = model.get_prev_raw_path(early_stopper.best_epoch)
    best_self_rep = torch.load(best_self_rep_path)
    best_prev_raw = torch.load(best_prev_raw_path)
    model.set_self_rep(best_self_rep, best_prev_raw)
    model.set_seed(seeds[early_stopper.best_epoch])
    logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
    model.eval()
    should_break = True
  else:
    for i in range(n_hop + 1):
      torch.save(model.neighborhood_store[i], model.get_ngh_store_path(epoch, i))
    torch.save(model.state_dict(), model.get_checkpoint_path(epoch))
    torch.save(model.self_and_edge_rep, model.get_self_rep_path(epoch))
    torch.save(model.prev_raw, model.get_prev_raw_path(epoch))
    
    # delete models from the earlier epochs
    legacy_epoch = epoch - early_stopper.max_round
    if legacy_epoch >= 0:
      for i in range(n_hop + 1):
        os.remove(model.get_ngh_store_path(legacy_epoch, i))
      os.remove(model.get_checkpoint_path(legacy_epoch))
      os.remove(model.get_self_rep_path(legacy_epoch))
      os.remove(model.get_prev_raw_path(legacy_epoch))
  
  return should_break

def train_val(train_val_data, model, mode, bs, epochs, criterion, optimizer, early_stopper, logger, model_dim, n_hop=1, seed=2023, eval_bs=50, all_nodes=None):
  # unpack the data, prepare for the training
  train_data, val_data = train_val_data
  train_src_l, train_tgt_l, train_ts_l, train_e_idx_l, train_src_label_l = train_data
  val_src_l, val_tgt_l, val_ts_l, val_e_idx_l, val_src_label_l = val_data
  num_instance = len(train_src_l)
  device = model.device
  if mode == 't':
    train_bad_l = np.random.randint(1, model.total_nodes, size=num_instance)
  else:
    train_bad_l = np.random.choice(all_nodes, size=num_instance)
  train_bad_l = torch.from_numpy(train_bad_l)
  train_src_l = torch.from_numpy(train_src_l)
  train_tgt_l = torch.from_numpy(train_tgt_l)
  train_ts_l = torch.from_numpy(train_ts_l)
  train_e_idx_l = torch.from_numpy(train_e_idx_l)
  num_batch = math.ceil(num_instance / bs)
  logger.info('num of training instances: {}'.format(num_instance))
  logger.info('num of batches per epoch: {}'.format(num_batch))
  idx_list = np.arange(num_instance)
  seeds = []
  seed = seed
  train_time = []
  for epoch in range(epochs):
      should_break = train_one_epoch(model, seed, seeds, logger, epoch, num_batch, bs, num_instance, train_e_idx_l, idx_list,
        train_src_l, train_tgt_l, train_bad_l, train_ts_l, optimizer, device, criterion, train_time, n_hop,
        mode, val_src_l, val_tgt_l, val_ts_l, val_src_label_l, val_e_idx_l, eval_bs, early_stopper, all_nodes)
      if should_break:
        break
  csv_handler.save_data()

def generate_temporal_embeddings(data, model, mode, bs, logger, model_dim, emb_file_stub, n_hop=2, seed=2023):
  logger.info('Generating temporal embeddings...')
  emb = list()
  data_row = list()
  node_edges = list()
  # unpack the data, prepare for the training
  src_l, tgt_l, ts_l, e_idx_l, src_label_l, tgt_label_l = data
  device = model.device
  num_instance = len(src_l)
  num_batch = math.ceil(num_instance / bs)
  logger.info('num of temporal instances: {}'.format(num_instance))
  logger.info('num of batches: {}'.format(num_batch))
  idx_list = np.arange(num_instance)
  seeds = []
  seed = seed
  train_time = []
  model.eval()
  train_start = time.time()
  model.set_seed(seed)
  set_random_seed(seed)
  seeds.append(seed)
  model.reset_store()

  model.reset_edge_feat_partition_to_cpu()
  
  model.phase = 'train'
  for k in tqdm(range(num_batch)):
      # generate training mini-batch
    s_idx = k * bs
    e_idx = min(num_instance, s_idx + bs)
    if s_idx == e_idx:
      continue
    batch_idx = idx_list[s_idx:e_idx] # shuffle training samples for each batch
    np.random.shuffle(batch_idx)
    src_l_cut, tgt_l_cut = src_l[batch_idx], tgt_l[batch_idx]
    ts_l_cut = ts_l[batch_idx]
    e_l_cut = e_idx_l[batch_idx]
    e_start = e_idx_l[s_idx]
    src_label_l_cut = src_label_l[batch_idx]
    tgt_label_l_cut = tgt_label_l[batch_idx]
    model.check_idx_and_load_edge_feat_partition_to_gpu(e_start)

    size = len(src_l_cut)

    src_th = torch.from_numpy(src_l_cut).to(dtype=torch.long, device=device)
    tgt_th = torch.from_numpy(tgt_l_cut).to(dtype=torch.long, device=device)

    cut_time_th = torch.from_numpy(ts_l_cut).to(dtype=torch.float, device=device)
    e_idx_th = torch.from_numpy(e_l_cut).to(dtype=torch.long, device=device)
    e_feats = model.fetch_edge_feat(e_idx_th)
    with torch.no_grad():
      collect_emb_start = time.time()
      embeddings, updated_mem_h0, updated_mem_h1, n_feats = model.updated_embeddings(size, src_th, tgt_th, tgt_th, cut_time_th, e_feats)
      collect_emb_end = time.time()
      ## Update memory after prediction made
      model.update_memory(src_th, tgt_th, e_feats, cut_time_th, embeddings, updated_mem_h1, size)
    decoder_input = embeddings[:2 * size]

    label_l_cut = np.concatenate((src_label_l_cut, tgt_label_l_cut), 0)
    labels_batch_torch = torch.from_numpy(label_l_cut).type(torch.LongTensor).to(device)

    condition = labels_batch_torch != -1 
    labels_batch_torch = labels_batch_torch[condition]
    label_l_cut = labels_batch_torch.cpu().numpy()
    decoder_input = decoder_input[condition]
    emb.append(decoder_input.cpu())
    node_id_to_save = torch.cat((src_th, tgt_th), dim=0)[condition]
    e_idx_to_save = torch.cat((e_idx_th, e_idx_th), dim=0)[condition]
    cut_time_to_save = torch.cat((cut_time_th, cut_time_th), dim=0)[condition]
    node_edges.append(torch.cat((node_id_to_save.unsqueeze(1).cpu(),e_idx_to_save.unsqueeze(1).cpu()), dim = 1)) 
    data_row.append(torch.cat((cut_time_to_save.unsqueeze(1).float().cpu(), labels_batch_torch.unsqueeze(1).float().cpu()), dim = 1))
  emb = torch.cat(emb, dim=0)
  rows = torch.cat(data_row, dim=0)
  node_edges = torch.cat(node_edges, dim=0)
  model.reset_edge_feat_partition_to_cpu()
  return emb, rows, node_edges



def train_val_node_with_embed(train_val_emb, train_val_data, train_val_node_edges, bs, mode, epochs, criterion, optimizer, early_stopper, logger,
  decoder, decoder_optimizer, decoder_loss_criterion, device, seed=2023, num_classes=1, eval_bs=50):
  # unpack the data, prepare for the training
  train_data, val_data = train_val_data
  train_node_edges, val_node_edges = train_val_node_edges
  train_ts_th, train_label_th = train_data[:, 0], train_data[:, 1] 
  train_emb, val_emb = train_val_emb

  num_instance = len(train_ts_th)  
  num_batch = math.ceil(num_instance / bs)
  logger.info('num of training instances: {}'.format(num_instance))
  logger.info('num of batches per epoch: {}'.format(num_batch))
  idx_list = np.arange(num_instance)
  seeds = []
  seed = seed
  train_time = []
  for epoch in range(epochs):
    train_start = time.time()

    acc, ap, f1, auc, m_loss, labels, preds = [], [], [], [], [], [], []
    np.random.shuffle(idx_list)  # shuffle the training samples for every epoch
    logger.info('start {} epoch'.format(epoch))
    decoder.train()
    loss = 0
    for k in tqdm(range(num_batch)):
       # generate training mini-batch
      s_idx = k * bs
      e_idx = min(num_instance, s_idx + bs)
      
      if s_idx == e_idx:
        continue
      batch_idx = idx_list[s_idx:e_idx] # shuffle training samples for each batch
      np.random.shuffle(batch_idx)
      ts_th_cut = train_ts_th[batch_idx].to(dtype=torch.float, device=device)
      emb_cut = train_emb[batch_idx].to(device=device)
      label_th_cut = train_label_th[batch_idx].to(dtype=torch.long, device=device)
      decoder_optimizer.zero_grad()
      decoder.train()
      predict_start = time.time()
      pred_score = decoder(emb_cut)
      predict_end = time.time()
      predict_time = predict_end - predict_start
      decoder_loss = decoder_loss_criterion(pred_score, label_th_cut)
      decoder_loss.backward()
      decoder_optimizer.step()
      # collect training results
      with torch.no_grad():
        decoder.eval()
        if num_classes == 2:
          pred_score = pred_score.softmax(dim=1)[:, 1].cpu().numpy()
          pred_label = pred_score > 0.5
          preds.append(pred_score)
        else:
          pred_score = pred_score.cpu().numpy()
          pred_label = np.argmax(pred_score, axis=1)
          preds.append(pred_label)
        true_label = label_th_cut.cpu().numpy()
        m_loss.append(decoder_loss.item())
        labels.append(true_label)

    true_l = np.concatenate(labels, -1).astype(int)
    pred_l = np.concatenate(preds, -1)
    if num_classes == 2:
      ap.append(average_precision_score(true_l, pred_l))
      auc.append(roc_auc_score(true_l, pred_l))
    else:
      ap.append(0)
      auc.append(0)
    train_end = time.time()
    train_time.append(train_end - train_start)
    nlb_results(logger, train_time, "train_time")
    # validation phase use all information
    val_start = time.time()
    val_acc, val_ap, val_f1, val_auc, predict_total_time = eval_node_with_embed('val for {} nodes'.format(mode),
      decoder, val_emb, val_data, val_node_edges, device=device, num_classes=num_classes, bs=eval_bs)
    val_end = time.time()
    logger.info('epoch: {}:'.format(epoch))
    logger.info('epoch mean loss: {}'.format(np.mean(m_loss)))
    logger.info('train acc: {}, val acc: {}'.format(np.mean(acc), val_acc))
    logger.info('train auc: {}, val auc: {}'.format(np.mean(auc), val_auc))
    logger.info('train ap: {}, val ap: {}'.format(np.mean(ap), val_ap))
    logger.info('train f1: {}, val f1: {}'.format(np.mean(f1), val_f1))
    logger.info('train time: {}, val time: {}'.format(train_end - train_start, predict_total_time))
    if num_classes == 2:
      early_stop_metric = val_auc
    else:
      early_stop_metric = val_f1
    torch.save(decoder.state_dict(), decoder.get_checkpoint_path(epoch))
