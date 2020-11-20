import numpy as np
import pandas as pd
import argparse
from mfa import MixtureFA

parser = argparse.ArgumentParser()
parser.add_argument('name', help='e.g. 20Kcells_PC')
parser.add_argument('K', type=int, help='number of mixtures')
parser.add_argument('advi_iter', type=int)
parser.add_argument('input_dir', nargs='?', default='../data/')
parser.add_argument('output_dir', nargs='?', default='outputs/')
args = parser.parse_args()

data = pd.read_csv(args.input_dir + 'scaled_{}.csv'.format(args.name), delimiter=',', index_col=0)
Y = data.values

mfa = MixtureFA(Y=Y, name='ipsc_{}_K{}'.format(args.name, args.K), K=args.K, advi_iter=args.advi_iter, output_dir=args.output_dir)
mfa.fit()
mfa.get_posterior()