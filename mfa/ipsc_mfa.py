import numpy as np
import pandas as pd
import argparse
from mfa import MixtureFA

parser = argparse.ArgumentParser()
parser.add_argument('name', help='e.g. 20Kcells_PC')
parser.add_argument('K', type=int, help='number of mixtures')
args = parser.parse_args()

data_dir = '../data/'
data = pd.read_csv(data_dir + 'scaled_{}.csv'.format(args.name), delimiter=',', index_col=0)
Y = data.values

mfa = MixtureFA(Y=Y, name='ipsc_{}_K{}'.format(args.name, args.K), K=args.K, advi_iter=10)
mfa.fit()
mfa.get_posterior()