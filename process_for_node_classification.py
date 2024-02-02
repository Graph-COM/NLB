import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import pickle
import torch
import os
import argparse

def get_args():
  parser = argparse.ArgumentParser('Interface for node classification pre-processing')

  # select dataset and training mode
  parser.add_argument('-d', '--data', type=str, help='data sources to process, try WIKI or REDDIT',
            choices=['GDELT', 'REDDIT', 'WIKI', 'MAG', 'UBUNTU', 'WIKITALK'],
            default='WIKI')

  try:
    args = parser.parse_args()
  except:
    parser.print_help()
    sys.exit(0)
  return args, sys.argv

args, sys_argv = get_args()

DATA = args.data

print(DATA)


def remove_node_with_no_edges_from_features_and_labels(DATA):
  g_df = pd.read_csv('./DATA/{}/edges.csv'.format(DATA))
  edge_feat = torch.load('./DATA/{}/edge_features.pt'.format(DATA))

  g_df = g_df.rename(columns={'Unnamed: 0': 'idx', 'src': 'u', 'dst': 'i', 'time': 'ts'})
  l_df = pd.read_csv('./DATA/{}/labels.csv'.format(DATA))
  src_l = g_df.u.values.astype(int)
  tgt_l = g_df.i.values.astype(int)
  min_node = min(src_l.min(), tgt_l.min())
  unique_ids, indices = np.unique(np.stack([src_l, tgt_l]), return_inverse=True) # [6,4,1...] -> [1,4,6,...,1000], [2,1,0...,]
  old_to_new_map = {element: index for index, element in enumerate(unique_ids)} # {1:0, 4:1, 6:2, ...}
  node_feats = None
  if os.path.isfile('./DATA/{}/node_features.pt'.format(DATA)):
    node_feats = torch.load('./DATA/{}/node_features.pt'.format(DATA))

    node_feats = node_feats.index_select(0, torch.from_numpy(unique_ids).to(dtype=torch.long))
    if min_node == 0:
      padding = torch.zeros_like(node_feats[:1])
      node_feats = torch.cat((padding, node_feats), 0)
      print("DATA {} increases node index by 1 and fill index 0 with null".format(DATA))
    print(node_feats.shape)
    torch.save(node_feats, './DATA/{}/node_features_remapped.pt'.format(DATA))

  new_src = indices[:len(src_l)] + 1 # [3,2,1...,]
  new_tgt = indices[len(src_l):] + 1 # [3,2,1...,]

  g_df.u = new_src
  g_df.i = new_tgt


  max_id = indices.max()
  print(max_id)

  l_nodes = l_df.node.values

  l_new_id = [old_to_new_map.get(node, -1) + 1 for node in l_nodes]
  l_df.node = l_new_id
  print('Number of labels before removing nodes that never have and edge:', len(l_df))
  l_df = l_df[l_df['node'] != 0]
  print('Number of labels after removing nodes that never have and edge:', len(l_df))
  l_df.to_csv('./DATA/{}/labels_remapped.csv'.format(DATA), index=False)
  g_df.to_csv('./DATA/{}/edges_remapped.csv'.format(DATA), index=False)
  if node_feats is not None:
    assert(node_feats.shape[0] == max_id + 2)
remove_node_with_no_edges_from_features_and_labels(DATA)

def associate_labels_to_edges(DATA):

  l_df = pd.read_csv('./DATA/{}/labels_remapped.csv'.format(DATA))
  g_df = pd.read_csv('./DATA/{}/edges_remapped.csv'.format(DATA))


  g_df.sort_values(by=['ts', 'u', 'i'], inplace=True)
  l_df.sort_values(by=['time', 'node'], inplace=True)

  if edge_feat is not None:
    inverted_indices_g_df = np.argsort(g_df.idx.values)
    edge_feat = edge_feat.index_select(0, torch.from_numpy(inverted_indices_g_df).to(dtype=torch.long))
    torch.save(edge_feat, './DATA/{}/edge_features_remap.pt'.format(DATA))



  labels = l_df.label.values
  time = l_df.time.values

  # Edges that happen after the labels with the largest timestamp is not useful
  # for prediction the node labels
  max_time = time.max()
  max_t_idx = np.searchsorted(g_df['ts'].values, max_time, side='right')
  print('Number of edges before truncating at the maximum timestamp node labels appear:', len(g_df))
  g_df = g_df[:max_t_idx]
  print('Number of edges after truncating at the maximum timestamp node labels appear:', len(g_df))



  s_idx = 0
  t_idx = 0

  # Initialize src and tgt labels for each edge with -1
  src_label = np.zeros(len(g_df)) - 1
  tgt_label = np.copy(src_label)


  # We keep track of which label is taken to avoid duplicate (node, time, label) tuples.
  taken = np.zeros(len(l_df))
  lidx = 0

  index = -1
  print("Iterating over the edges to find (src_node, time) pairs in the edges that match the (node, time) pair in the labels.")
  for _, row in g_df.iterrows():
    index += 1
    src = row.u
    time = row.ts
    if lidx >= len(l_df):
      print("Finished.")
      break
    li = l_df.iloc[lidx]
    # if the label's timestamp is smaller than the timestamp of the current edge,
    # Or if the label's node id is smaller than the src node id of the current edge,
    # Iterate by moving the label's index.
    # Else break.
    while li.time < time or (li.time == time and li.node < src):
      lidx += 1
      if lidx >= len(l_df):
        break
      li = l_df.iloc[lidx]
    if lidx >= len(l_df):
      print("Finished.")
      break
    # if the label's timestamp is larger than the timestamp of the current edge,
    # Continue.
    if li.time > time:
      continue
    # If the label's timestamp matches with the timestamp of the current edge,
    # And the label's node id matches with the src node id of the current edge,
    # change the current src_label to the current label
    # and move the label's index and mark it taken.
    # Otherwise, thelabel's node id is larger than the src node id of the current edge,
    # then Continue.
    if li.node == src:
      src_label[index] = li.label
      taken[lidx] = 1
      lidx += 1

  g_df['src_label'] = src_label
  # We iterate over only the labels that does not find a match with the src nodes
  l_df_left = l_df[taken == 0]
  print("Number of remaining labels that do not have a match with the src nodes of the graph", len(l_df_left))

  # Sort by tgt
  g_df.sort_values(by=['ts', 'i', 'u'], inplace=True)
  lidx = 0
  taken = np.zeros(len(l_df_left))
  index = -1
  print("Iterating over the edges to find (tgt_node, time) pairs in the edges that match the (node, time) pair in the labels.")
  for _, row in g_df.iterrows():
    index += 1
    tgt = row.i
    time = row.ts
    if lidx >= len(l_df_left):
      print("Finished.")
      break
    li = l_df_left.iloc[lidx]
    # if the label's timestamp is smaller than the timestamp of the current edge,
    # Or if the label's node id is smaller than the tgt node id of the current edge,
    # Iterate by moving the label's index.
    # Else break.
    while li.time < time or (li.time == time and li.node < tgt):
      lidx += 1
      if lidx >= len(l_df_left):
        break
      li = l_df_left.iloc[lidx]
    if lidx >= len(l_df_left):
      print("Finished.")
      break
    # if the label's timestamp is larger than the timestamp of the current edge,
    # Continue.
    if li.time > time:
      continue
    # If the label's timestamp matches with the timestamp of the current edge,
    # And the label's node id matches with the tgt node id of the current edge,
    # change the current tgt_label to the current label
    # and move the label's index and mark it taken.
    # Otherwise, thelabel's node id is larger than the tgt node id of the current edge,
    # then Continue.
    if li.node == tgt:
      tgt_label[index] = li.label
      taken[lidx] = 1
      lidx += 1

  g_df['tgt_label'] = tgt_label
  print("Number of remaining labels that do not have a match with the src and tgt nodes of the graph", len(l_df_left[taken == 0]))
  g_df.to_csv('./DATA/{}/edges_with_node_labels.csv'.format(DATA), index=False)

associate_labels_to_edges(DATA)

# For TGL, it only takes labels without the edges, the following function extract only labels
# that is valid in the edges from the edge label pairs contructed in associate_labels_to_edges.
def extract_labels_from_edge_label_tuples(DATA):
  l_df = pd.read_csv('./DATA/{}/edges_with_node_labels.csv'.format(DATA))
  l_df.sort_values(by=['ts', 'u', 'i'], inplace=True)
  g_df = pd.read_csv('./DATA/{}/edges.csv'.format(DATA))
  g_df.sort_values(by=['time', 'src', 'dst'], inplace=True)
  g_df = g_df[:len(l_df)]
  srcs = g_df["src"].values.tolist()
  dsts = g_df["dst"].values.tolist()
  ts = l_df["ts"].values.tolist()
  ext_roll = l_df["ext_roll"].values.tolist()
  src_label = l_df["src_label"].values.tolist()
  tgt_label = l_df["tgt_label"].values.tolist()

  nodes = np.stack([srcs, dsts], axis = -1).flatten()
  ts = np.stack([ts, ts], axis = -1).flatten()
  ext_roll = np.stack([ext_roll, ext_roll], axis = -1).flatten()
  labels = np.stack([src_label, tgt_label], axis = -1).flatten()

  df = pd.DataFrame(columns=["node","time","label","ext_roll"])
  df.node = nodes
  df.time = ts
  df.ext_roll = ext_roll
  df.label = labels
  print('All src and tgt labels', len(df))
  df = df[df.label != -1]
  print('Only labels that are valid', len(df))

  df.to_csv('./DATA/{}/only_node_labels.csv'.format(DATA))
extract_labels_from_edge_label_tuples(DATA)
