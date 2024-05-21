import math
import torch
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import time


def build_history_interactions(hint, model, src, tgt, ts, label, e_id, bs=2000, device=None):
  val_acc, val_ap, val_f1, val_auc = [], [], [], []
  model.reset_edge_feat_partition_to_cpu()
  with torch.no_grad():
    model = model.eval()
    TEST_BATCH_SIZE = bs
    num_test_instance = len(src)
    # 
    b_max = math.ceil(num_test_instance / TEST_BATCH_SIZE)
    b_min = 0
    predict_total_time = 0
    for k in range(b_min, b_max):
      s_idx = k * TEST_BATCH_SIZE
      e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
      if s_idx == e_idx:
        continue
      e_start = e_id[s_idx]
      model.check_idx_and_load_edge_feat_partition_to_gpu(e_start)
      batch_idx = np.arange(s_idx, e_idx)
      # np.random.shuffle(batch_idx)
      src_l_cut = src[batch_idx]
      tgt_l_cut = tgt[batch_idx]
      ts_l_cut = ts[batch_idx]
      e_l_cut = e_id[batch_idx] if (e_idx is not None) else None

      size = len(src_l_cut)
      bad_l_cut = np.random.randint(1, model.total_nodes, size=size)
      src_l_cut = torch.from_numpy(src_l_cut).to(dtype=torch.long, device=device)
      tgt_l_cut = torch.from_numpy(tgt_l_cut).to(dtype=torch.long, device=device)
      bad_l_cut = torch.from_numpy(bad_l_cut).to(dtype=torch.long, device=device)
      ts_l_cut = torch.from_numpy(ts_l_cut).to(dtype=torch.float, device=device)
      e_l_cut = torch.from_numpy(e_l_cut).to(dtype=torch.long, device=device)
      model.update_cache_only(src_l_cut, tgt_l_cut, ts_l_cut, e_idx_l=e_l_cut)


# Phase = [train, val, test]
# mode = [t, i] # transductive or inductive
def eval_one_epoch(hint, model, src, tgt, ts, label, e_id, bs=2000, phase=None, mode=None, device=None, all_nodes=None, eval_neg_samples=1):
  val_acc, val_ap, val_f1, val_auc = [], [], [], []
  if model.edge_feats_partitions != None:
    model.reset_edge_feat_partition_to_cpu()
    if phase == 'val':
      e_id_remap = e_id - model.n_edges["train"] # make the e_id start from 0
    else:
      e_id_remap = e_id - model.n_edges["train"] - model.n_edges["val"] # make the e_id start from 0
  else:
    e_id_remap = e_id
  model.phase = phase
  test_size = len(src)
  if mode == 't':
    bad = np.random.randint(1, model.total_nodes, size=len(src))
  else:
    bad = np.random.choice(all_nodes, size=len(src))
  src = torch.from_numpy(src).to(dtype=torch.long, device=device)
  tgt = torch.from_numpy(tgt).to(dtype=torch.long, device=device)
  bad = torch.from_numpy(bad).to(dtype=torch.long, device=device)
  ts = torch.from_numpy(ts).to(dtype=torch.float, device=device)
  e_id_remap = torch.from_numpy(e_id_remap).to(dtype=torch.long, device=device)

  with torch.no_grad():
    model = model.eval()
    TEST_BATCH_SIZE = bs
    num_test_instance = len(src)
    b_max = math.ceil(num_test_instance / TEST_BATCH_SIZE)
    b_min = 0
    predict_total_time = 0
    for k in range(b_min, b_max):
      s_idx = k * TEST_BATCH_SIZE
      e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
      if s_idx == e_idx:
        continue
      e_start = e_id_remap[s_idx]
      model.check_idx_and_load_edge_feat_partition_to_gpu(e_start)
      batch_idx = np.arange(s_idx, e_idx)
      # np.random.shuffle(batch_idx)
      src_l_cut = src[batch_idx]
      tgt_l_cut = tgt[batch_idx]
      bad_l_cut = bad[batch_idx]
      size = len(src_l_cut)
      if eval_neg_samples > 1:
        if mode == 't':
          bad_l_cut = np.random.randint(1, model.total_nodes, size=size*eval_neg_samples)
        else:
          bad_l_cut = np.random.choice(all_nodes, size=size*eval_neg_samples)
        bad_l_cut = torch.from_numpy(bad_l_cut).to(dtype=torch.long, device=device)
      ts_l_cut = ts[batch_idx]
      e_l_cut = e_id_remap[batch_idx] if (e_idx is not None) else None

      size = len(src_l_cut)
      pos_prob, neg_prob, predict_time = model.contrast(src_l_cut, tgt_l_cut, bad_l_cut, ts_l_cut, e_l_cut, neg_samples=eval_neg_samples)
      predict_total_time += predict_time
      time_start = time.time()
      if eval_neg_samples == 1:
        pred_score = torch.cat((pos_prob, neg_prob), 0).sigmoid().cpu().numpy()
        pred_label = pred_score > 0.5
        true_label = np.concatenate([np.ones(size), np.zeros(size)])

        val_acc.append((pred_label == true_label).mean())
        pred_score[np.isnan(pred_score)] = 0
        val_ap.append(average_precision_score(true_label, pred_score))
        val_auc.append(roc_auc_score(true_label, pred_score))
      else:
        mrr = torch.reciprocal(torch.sum(pos_prob.squeeze() < neg_prob.squeeze().reshape(eval_neg_samples, -1), dim=0) + 1).type(torch.float64).cpu()
      time_end = time.time()
  return np.mean(val_acc), np.mean(val_ap), None, np.mean(val_auc), predict_total_time


def eval_node_with_embed(hint, decoder, emb, data, val_node_edges, bs=2000, device=None, num_classes=1):
  ts_th, label_th = data[:, 0], data[:, 1]

  val_acc, val_ap, val_f1, val_auc, labels, preds = [], [], [], [], [], []
  with torch.no_grad():
    decoder.eval()
    TEST_BATCH_SIZE = bs
    num_test_instance = len(ts_th)
    # 
    b_max = math.ceil(num_test_instance / TEST_BATCH_SIZE)
    b_min = 0
    predict_total_time = 0
    for k in range(b_min, b_max):
      s_idx = k * TEST_BATCH_SIZE
      e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
      if s_idx == e_idx:
        continue
      batch_idx = np.arange(s_idx, e_idx)
      np.random.shuffle(batch_idx)
      ts_th_cut = ts_th[batch_idx].to(dtype=torch.float, device=device)
      emb_cut = emb[batch_idx].to(device=device)
      label_th_cut = label_th[batch_idx].to(dtype=torch.long, device=device)
      pred_score = decoder(emb_cut)
      if num_classes == 2:
        pred_score = pred_score.softmax(dim=1)[:, 1].cpu().numpy()
        pred_label = pred_score > 0.5
        preds.append(pred_score)
      else:
        pred_score = pred_score.cpu().numpy()
        pred_label = np.argmax(pred_score, axis=1)
        preds.append(pred_label)
      true_label = label_th_cut.cpu().numpy()

      val_acc.append((pred_label == true_label).mean())

      labels.append(true_label)
      
    true_l = np.concatenate(labels, -1).astype(int)
    pred_l = np.concatenate(preds, -1)
    if num_classes == 2:
      val_ap = average_precision_score(true_l, pred_l)
      val_auc = roc_auc_score(true_l, pred_l)
      val_f1 = f1_score(true_l, (pred_l > 0.5).astype(int), average='micro')
    else:
      val_ap = 0
      val_auc = 0
      val_f1 = f1_score(true_l, pred_l.astype(int), average='micro')
  return np.mean(val_acc), val_ap, val_f1, val_auc, predict_total_time

