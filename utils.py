import numpy as np
import torch
import os
import random
import statistics
import math

class NodeClassificationModel(torch.nn.Module):

    def __init__(self, dim_in, dim_hid, num_class, get_checkpoint_path=None):
        super(NodeClassificationModel, self).__init__()
        self.fc1 = torch.nn.Linear(dim_in, dim_hid)
        self.fc2 = torch.nn.Linear(dim_hid, num_class)
        self.get_checkpoint_path = get_checkpoint_path

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        return x

class MLP(torch.nn.Module):
  def __init__(self, dim, drop=0.3, output_dim = 1, get_checkpoint_path=None):
    super().__init__()
    self.fc_1 = torch.nn.Linear(dim, 80)
    self.fc_2 = torch.nn.Linear(80, 10)
    self.fc_3 = torch.nn.Linear(10, output_dim)
    self.act = torch.nn.ReLU()
    self.output_dim = output_dim
    self.dropout = torch.nn.Dropout(p=drop, inplace=False)
    self.get_checkpoint_path = get_checkpoint_path

  def forward(self, x):
    x = self.act(self.fc_1(x))
    x = self.dropout(x)
    x = self.act(self.fc_2(x))
    x = self.dropout(x)
    return self.fc_3(x).squeeze(dim=1)

class EarlyStopMonitor(object):
  def __init__(self, max_round=5, higher_better=True, tolerance=1e-4):
    self.max_round = max_round
    self.num_round = 0

    self.epoch_count = 0
    self.best_epoch = 0

    self.last_best = None
    self.higher_better = higher_better
    self.tolerance = tolerance

  def early_stop_check(self, curr_val):
    if not self.higher_better:
      curr_val *= -1
    if self.last_best is None:
      self.last_best = curr_val
    elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
      self.last_best = curr_val
      self.num_round = 0
      self.best_epoch = self.epoch_count
    else:
      self.num_round += 1
    self.epoch_count += 1
    return self.num_round >= self.max_round

class RandEdgeSampler(object):
  def __init__(self, src_list, dst_list):
    src_list = np.flip(np.concatenate(src_list))
    dst_list = np.flip(np.concatenate(dst_list))
    self.src_list, src_idx = np.unique(src_list, return_index=True)
    self.dst_list, dst_idx = np.unique(dst_list, return_index=True)

  def sample(self, size):
    src_index = np.random.randint(0, len(self.src_list), size)
    dst_index = np.random.randint(0, len(self.dst_list), size)
    return self.src_list[src_index], self.dst_list[dst_index]


def set_random_seed(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(seed)
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)


def process_sampling_numbers(num_neighbors, num_layers):
  num_neighbors = [int(n) for n in num_neighbors]
  if len(num_neighbors) == 1:
    num_neighbors = num_neighbors * num_layers
  else:
    num_layers = len(num_neighbors)
  return num_neighbors, num_layers

def nlb_results(logger, arr, name):
  logger.info(name + " " + str(arr))
  logger.info("Mean " + str(100 * statistics.mean(arr)))
  logger.info("Standard deviation " + str(statistics.pstdev(arr)))
  logger.info("95% " + str(1.96 * 100 * statistics.pstdev(arr) / math.sqrt(len(arr))))
  logger.info("--------")