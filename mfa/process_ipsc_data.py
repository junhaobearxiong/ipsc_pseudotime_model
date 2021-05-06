import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

input_path = '../data/scaled_20Kcells.csv'
save_path = '../data/'
output_dir = 'outputs/'

np.random.seed(0)

data = pd.read_csv(input_path, delimiter=',', index_col=0)

pca = PCA(n_components=20).fit(data)
data_pcs = pca.transform(data)

plt.figure(figsize=(10, 4))
plt.plot(np.arange(1, 21), pca.explained_variance_ratio_)
plt.xlabel('PC')
plt.ylabel('Explained Variance Ratio')
plt.savefig(output_dir + 'pca_explained_variance.png')

# PC
data_pcs = pd.DataFrame(data_pcs, index=data.index, columns=np.arange(1, 21))
data_pcs.to_csv(save_path + 'scaled_20Kcells_PC.csv')

# subsample
data_sample = data.sample(n=2000)
data_sample.to_csv(save_path + 'scaled_2Kcells.csv')
data_pcs.loc[data_sample.index].to_csv(save_path + 'scaled_2Kcells_PC.csv')