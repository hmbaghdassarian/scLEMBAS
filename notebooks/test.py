#!/usr/bin/env python
# coding: utf-8

# Our expectation with the original LEMBAS is as follows:
# 1) It cannot handle multiple cell types
# 2) It cannot account for the dispersion within a cluster at single-cell resolution, and will output the centroid of a cluster instead.

# Next, we want to see whether scLEMBAS can capture the heterogeneity of cell responses upon ligand exposure. 

# In[1]:


import os

import numpy as np
import pandas as pd

import anndata
import scanpy as sc
from sklearn.neighbors import NearestCentroid
from scipy.spatial.distance import cdist, pdist, squareform

import torch

import matplotlib.pyplot as plt
import seaborn as sns
import plotnine as p9
import patchworklib as pw

import sys

sclembas = '/home/hmbaghda/Projects/scLEMBAS'
sys.path.insert(1, os.path.join(sclembas))
from scLEMBAS import parse_network, io
from scLEMBAS.model.scl import SignalingModel
from scLEMBAS.model.train import TrainSimple
from scLEMBAS.plotting import plot_embedding
from scLEMBAS.preprocess import embed_tf_activity


# In[2]:


n_cores = 12
os.environ["OMP_NUM_THREADS"] = str(n_cores)
os.environ["MKL_NUM_THREADS"] = str(n_cores)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_cores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_cores)
os.environ["NUMEXPR_NUM_THREADS"] = str(n_cores)

seed = 888

device = "cuda" if torch.cuda.is_available() else "cpu"

data_path = '/nobackup/users/hmbaghda/scLEMBAS/analysis'
models_path = os.path.join(data_path, 'processed', 'models')
if not os.path.isdir(models_path):
    os.mkdir(models_path)


# In[3]:


tf_adata = io.read_tfad(file_name = os.path.join(data_path, 'processed', 'ID_tf_activity.h5ad'))


# # Load and Parse Input Signaling Network

# In[4]:


source_label = 'source_genesymbol'
target_label = 'target_genesymbol'
weight_label = 'mode_of_action'
stimulation_label = 'consensus_stimulation'
inhibition_label = 'consensus_inhibition'





tf_labels = tf_adata.var.index.unique().tolist()

ligand_labels = tf_adata.obs['sample'].unique().tolist()
ligand_labels = [(l[0] + l[1:].lower()).replace('-', '') for l in ligand_labels] # mouse naming convention



sn_ppis = pd.read_csv(os.path.join(data_path, 'processed', 'sn_ppis_og.csv'), index_col = 0) # smaller network


# The interactions include the following input ligands:

# In[41]:


all_nodes = sn_ppis[source_label].tolist() + sn_ppis[target_label].tolist()
input_ligands_available = sorted(set(ligand_labels).intersection(all_nodes))
print(*input_ligands_available, sep = ', ')


# # Explore performance on TF activity

# ## Scenario 1:

# In[42]:


model_no = 2


# The simplest scenario is:
# - 1 cell type
# - 1 ligand
# - Binary ligand exposure (0 or 1)
# - Exposure leads to distinct clusters in TF activity space
# 
# To identify the two most distinct clusters, let's calculate the Euclidean distance of the cluster centroids in PC space:

# In[43]:


group_label = 'TF_clusters' # ordered cateogry in obs

pca_rank = tf_adata.uns['pca']['pca_rank']
X_pca = tf_adata.obsm['X_pca'][:, :pca_rank] # PCA to pca_rank components




# Let's also get the within-cluster dispersion, as measured by WCSS. We adapt the below equation of WCSS to calculate just the inner sum for each cluster(not summing across all clusters, which gives one metric of overall dispersion) and normalize to the total number of points in the cluster:
# 
# 

# $$
# \text{WCSS} = \sum_{i=1}^{k} \sum_{\mathbf{x} \in C_i} [d(\mathbf{x}, \mathbf{\mu}_i)]^2
# $$
# 
# 
# - $k$: The number of clusters.
# - $C_i$: The set of points belonging to the $i$-th cluster.
# - $\mathbf{x}$: A data point within a cluster $C_i$.
# - $\mathbf{\mu}_i$: The centroid of the $i$-th cluster, which is the average position of all the points in $C_i$.
# - $d(\mathbf{x}, \mathbf{\mu}_i)$: The distance between a data point $\mathbf{x}$ and the centroid (here calculated as Euclidean distance)
# 
# Our WCSS:
# 
# For each $k$k:
# $$
# \text{WCSS} = \sum_{\mathbf{x} \in C_i} [d(\mathbf{x}, \mathbf{\mu}_i)]^2 / n
# $$
# 
# - $n$: The total number of points in the cluster
# 



# Based on these results, let's say that cluster 9 is unstimulated, and cluster 15 is stimulated. 

# In[44]:


np.random.seed(seed)
selected_ligand = np.random.choice(input_ligands_available, 1)[0]
print('The selected ligand is: ' + selected_ligand)

subset_tf = tf_adata[tf_adata.obs.TF_clusters.isin(['9', '15'])]
subset_tf.obs.TF_clusters.value_counts()


# To avoid any bias, let's randomly subset the larger cluster to match the size of the smaller one:

# In[45]:


sample_size = subset_tf.obs.TF_clusters.value_counts().min()
large_cluster = subset_tf.obs.TF_clusters.value_counts().idxmax()
small_cluster = subset_tf.obs.TF_clusters.value_counts().idxmin()
large_cluster_barcodes = subset_tf.obs[subset_tf.obs.TF_clusters == large_cluster].index
small_cluster_barcodes = subset_tf.obs[subset_tf.obs.TF_clusters == small_cluster].index.tolist()
np.random.seed(seed)
lcb_sub = list(np.random.choice(large_cluster_barcodes, sample_size, replace = False))
subset_tf = subset_tf[lcb_sub + small_cluster_barcodes, :]
subset_tf.obs.TF_clusters.value_counts()


# For now, for speed, let's subset so that there are just 100 "samples" per stimulation condition:

# In[46]:


# sample_size = 100

# barcodes = []
# for cluster_label in subset_tf.obs.TF_clusters.unique():
#     bc = subset_tf.obs[subset_tf.obs.TF_clusters == cluster_label].index
#     np.random.seed(seed)
#     barcodes += list(np.random.choice(bc, sample_size, replace = False))
# subset_tf = subset_tf[barcodes, :]
# subset_tf.obs.TF_clusters.value_counts()


# Next, let's initialize the model:

# In[47]:


ligand_input = pd.DataFrame(subset_tf.obs.TF_clusters.map({'9': 0, '15': 1}))
ligand_input.columns = [selected_ligand]
tf_output = pd.DataFrame(subset_tf.X, index = subset_tf.obs.index, columns = subset_tf.var.index)


# In[48]:


# linear scaling of inputs/outputs
projection_amplitude_in = 3
projection_amplitude_out = 1.2
# other parameters
bionet_params = {'target_steps': 100, 
                 'max_steps': 120, 
                 'exp_factor':50, 
                 'tolerance': 1e-5, 
                 'leak':1e-2} 

# training parameters
lr_params = {'lr_restart_optimizer_epoch': 50, 'lr_decay': 0.75, 'lr_restart_factor': 1}
other_params = {'network_noise_scale': 10, 'gradient_noise_scale': 1e-9}

max_epoch = 5000
maximum_learning_rate = 2e-3
lr_params = {'max_epochs': max_epoch, 
             'lr_restart_epoch': max_epoch + 1, #round(max_epoch/3),
             'maximum_learning_rate': maximum_learning_rate, 
             'minimum_learning_rate': maximum_learning_rate/10,
             'warmup_epochs': 1000,
             'reset_optimizer_epoch': 200}

other_params = {'batch_size': 256, 
                'network_noise_scale': 10, 
                'gradient_noise_scale': 1e-9}

regularization_params = {'param_lambda_L2': 1e-6, 
                         'discriminator_lambda_L2': 1e-5,
                         'moa_lambda_L1': 0.1, #'ligand_lambda_L2': 1e-5, 
                         'uniform_lambda_L2': 1e-4,  
                         'uniform_max': (1/1.2), 
                         'spectral_loss_factor': 1e-5}

spectral_radius_params = {'n_probes_spectral': 5, 
                          'power_steps_spectral': 50, 
                          'subset_n_spectral': 10}
training_params = {**lr_params, **other_params, **regularization_params, **spectral_radius_params}
target_spectral_radius = 0.8


# In[49]:


mod = SignalingModel(net = sn_ppis,
                     X_in = ligand_input,
                     y_out = tf_output, 
                     projection_amplitude_in = projection_amplitude_in, projection_amplitude_out = projection_amplitude_out,
                     weight_label = weight_label, source_label = source_label, target_label = target_label,
                     bionet_params = bionet_params, 
                     dtype = torch.float32, device = device, seed = seed)


# Next, let's train the model:

# In[50]:


# model setup
mod.input_layer.weights.requires_grad = False # don't learn scaling factors for the ligand input concentrations
mod.signaling_network.prescale_weights(target_radius = target_spectral_radius) # spectral radius


# In[51]:


# training loop
trainer = TrainSimple(mod = mod,
                      prediction_optimizer = torch.optim.Adam,
                      prediction_loss_fn = torch.nn.MSELoss(reduction='mean'),
                      hyper_params = training_params,
                      train_split = {'train': 0.8, 'test': 0.2, 'validation': None}, 
                      track_validation=False,
                      track_test=True,
                      train_seed = seed)
mod, cur_loss, cur_eig, stats_df = trainer.train_model(verbose = True)

# store results
io.write_pickled_object(trainer, os.path.join(models_path, str(model_no) + '_trainer.pickle'))
torch.save(obj=mod.state_dict(), f=os.path.join(models_path, 'model_' + str(model_no) + '_state_dict.pth'))
stats_df.to_csv(os.path.join(models_path, str(model_no) + '_stats_df.csv'))

