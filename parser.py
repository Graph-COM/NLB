import argparse
import sys


def get_args():
  parser = argparse.ArgumentParser('Interface for NLB')

  # select dataset and training mode
  parser.add_argument('-d', '--data', type=str, help='data sources to use, try WIKI or REDDIT',
            choices=['GDELT', 'REDDIT', 'WIKI', 'MAG', 'UBUNTU', 'WIKITALK'],
            default='WIKI')
  parser.add_argument('-m', '--mode', type=str, default='t', choices=['t', 'i'], help='transductive (t) or inductive (i)')

  # methodology-related hyper-parameters
  parser.add_argument('--n_degree', nargs='*', default=['20'],
            help='a list of neighbor sampling numbers for different hops, when only a single element is input n_layer will be activated')
  parser.add_argument('--n_hop', type=int, default=1, help='number of hops is used')
  parser.add_argument('--bias', default=0.0, type=float, help='the hyperparameter alpha controlling sampling preference with time closeness, default to 0 which is uniform sampling')
  parser.add_argument('--pos_dim', type=int, default=16, help='dimension of the positional embedding')
  parser.add_argument('--self_dim', type=int, default=72, help='dimension of the self representation')
  parser.add_argument('--nlb_node', action='store_true', default=False, help="whether use NLB-node or NLB-edge. NLB-edge by default.")

  parser.add_argument('--attn_n_head', type=int, default=2, help='number of heads used in tree-shaped attention layer, we only use the default here')
  parser.add_argument('--time_dim', type=int, default=1, help='dimension of the time embedding')
  # general training hyper-parameters
  parser.add_argument('--n_epoch', type=int, default=50, help='number of epochs')
  parser.add_argument('--bs', type=int, default=64, help='batch_size')
  parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
  parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability for all dropout layers')
  parser.add_argument('--attn_dropout', type=float, default=0.1, help='dropout probability for attn weights')
  parser.add_argument('--replace_prob', type=float, default=0.9, help='probability for inserting new interactions to downsampled temporal neighbors')
  parser.add_argument('--tolerance', type=float, default=1e-4,
            help='tolerated marginal improvement for early stopper')

  # parameters controlling computation settings but not affecting results in general
  parser.add_argument('--seed', type=int, default=0, help='random seed for all randomized algorithms')
  parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
  parser.add_argument('--verbosity', type=int, default=1, help='verbosity of the program output')
  parser.add_argument('--run', type=int, default=2, help='number of model runs')
  parser.add_argument('--model_path', type=str, default="", help='path to NLB trained model to be loaded')
  parser.add_argument('--eval_neg_samples', type=int, default=1, help='how many negative samples to use at inference. Note: this will change the metric of test set to AP+AUC to MRR!')


  try:
    args = parser.parse_args()
  except:
    parser.print_help()
    sys.exit(0)
  return args, sys.argv
