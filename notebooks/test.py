#!/usr/bin/env python
# coding: utf-8

# Our expectation with the original LEMBAS is as follows:
# 1) It cannot handle multiple cell types
# 2) It cannot account for the dispersion within a cluster at single-cell resolution, and will output the centroid of a cluster instead.

# Next, we want to see whether scLEMBAS can capture single-cell resolution data.

# In[1]:


import os
from itertools import combinations

import numpy as np
import pandas as pd

import anndata
import scanpy as sc
from sklearn.neighbors import NearestCentroid
from scipy.spatial.distance import cdist, pdist, squareform

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import seaborn as sns
import plotnine as p9
import patchworklib as pw

import sys

sclembas = '/home/hmbaghda/Projects/scLEMBAS'
sys.path.insert(1, os.path.join(sclembas))
from scLEMBAS import parse_network, io
from scLEMBAS.model.scl import SignalingModel
from scLEMBAS.model.train import TrainCat, TrainSC
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


# In[5]:


sn_ppis = parse_network.load_network('omnipath', organism = 'mouse', static = True)

sn_ppis = parse_network.correct_network(sn_ppis = sn_ppis,
                                        source_label = source_label, target_label = target_label,
                                        stimulation_label = stimulation_label, inhibition_label = inhibition_label)

sn_ppis = parse_network.extract_network(sn_ppis, curation_effort_thresh = 5, n_references_thresh = 3,
                                        resources = ['HuRI','IntAct','KEGG-MEDICUS','NetPath','Reactome_SignaLink3','SPIKE','SignaLink3','SIGNOR', 
                                                'Baccin2019', 'Ramilowski2015', 'Reactome_LRdb', 'UniProt_LRdb', 'CellChatDB', 'CellPhoneDB', 'connectomeDB2020', 'scConnect'], 
                                        source_label = source_label, target_label = target_label,
                                        drop_self = True, verbose = True)


# Filter for nodes that fall in paths between ligands and receptors (fully connected network):

# In[6]:


tf_labels = tf_adata.var.index.unique().tolist()

ligand_labels = tf_adata.obs['sample'].unique().tolist()
ligand_labels = [(l[0] + l[1:].lower()).replace('-', '') for l in ligand_labels] # mouse naming convention

# filter for paths b/w ligand and tf
fn_1, _ = parse_network.create_connected_network(sn_ppis, ligand_labels, tf_labels, source_label = source_label, target_label = target_label, 
                       path_finder = 'shortest')
fn_2, _ = parse_network.create_connected_network(sn_ppis, ligand_labels, tf_labels, source_label = source_label, target_label = target_label, 
                       path_finder = 'connected')
# of the methods to identify paths, retain the one that has the most interactions
if fn_1.shape[0] > fn_2.shape[0]:
    sn_ppis = fn_1
else:
    sn_ppis = fn_2

del fn_1, fn_2


# Finally, let's format the network as needed for input to building the model:

# In[7]:


sn_ppis = parse_network.format_network(sn_ppis, weight_label, stimulation_label, inhibition_label) 
# sn_ppis.to_csv(os.path.join(data_path, 'processed', 'ID_input_network.csv'))


# In[8]:


print('The signaling network contains {} interactions'.format(sn_ppis.shape[0]))
sn_ppis[[source_label, target_label, weight_label, stimulation_label, inhibition_label]].head()


# The interactions include the following input ligands:

# In[9]:


all_nodes = sn_ppis[source_label].tolist() + sn_ppis[target_label].tolist()
input_ligands_available = sorted(set(ligand_labels).intersection(all_nodes))
print(*input_ligands_available, sep = ', ')


# # Explore performance on TF activity

# In[10]:


group_label = 'TF_clusters' # ordered cateogry in obs


# ## Scenario 3:

# In[11]:


model_no = 3


# The simplest scenario is:
# - 2 cell type
# - 1 ligand
# - Binary ligand exposure (0 or 1)
# - Exposure leads to distinct clusters in TF activity space
# 
# For now, we simply use visualization to identify the most distinct clusters:

# In[12]:


fig, ax = plt.subplots(ncols = 2, figsize = (8, 4))
sc.pl.pca(tf_adata, color='TF_clusters', ax = ax[0], show = False)
ax[0].legend().set_visible(False)
ax[0].set_title('')
sc.pl.umap(tf_adata, color='TF_clusters', ax = ax[1], show = False)
ax[1].set_title('')

fig.suptitle('TF Activity Space')

fig.tight_layout()
# plt.savefig(fname = os.path.join(data_path, 'figures', 'tf_celltype_umap.png'), 
#             transparent = True, 
#             bbox_inches = 'tight')
plt.show()


# Based on these results, we choose clusters 2, 3, 4, and 5:

# In[13]:


max_clusters = ['3', '4', '5', '2']
viz_adata = tf_adata.copy()
viz_adata.obs[group_label + '_color'] = pd.Categorical(viz_adata.obs[group_label], 
                                                       categories = max_clusters, 
                                                       ordered = True)
fig, ax = plt.subplots(ncols = 2, figsize = (8, 4))
sc.pl.pca(viz_adata, color=group_label + '_color', ax = ax[0], show = False)
ax[0].legend().set_visible(False)
ax[0].set_title('')
sc.pl.umap(viz_adata, color=group_label + '_color', ax = ax[1], show = False)
ax[1].set_title('')

fig.suptitle('Max TF Cluster Distance')

fig.tight_layout()
# plt.savefig(fname = os.path.join(data_path, 'figures', 'tf_celltype_umap.png'), 
#             transparent = True, 
#             bbox_inches = 'tight')
plt.show()


# Based on these results, let's say that cluster 9 is unstimulated, and cluster 15 is stimulated. 

# In[14]:


max_clusters = ['3', '4', '5', '2']


# In[15]:


np.random.seed(seed)
selected_ligand = np.random.choice(input_ligands_available, 1)[0]
print('The selected ligand is: ' + selected_ligand)

subset_tf = tf_adata[tf_adata.obs.TF_clusters.isin(max_clusters)]
subset_tf.obs.TF_clusters.value_counts()


# For now, for speed, let's subset so that there are a smaller number of "samples" per condition:

# In[16]:


sample_size = int(2.5e3)

barcodes = []
for cluster_label in subset_tf.obs.TF_clusters.unique():
    bc = subset_tf.obs[subset_tf.obs.TF_clusters == cluster_label].index
    np.random.seed(seed)
    barcodes += list(np.random.choice(bc, sample_size, replace = False))
subset_tf = subset_tf[barcodes, :]
subset_tf.obs.TF_clusters.value_counts()


# Next, let's initialize the model. 
# 
# Let's say clusters 4 and 5 are unstimulated, and 3 and 2 are stimulated
# We assign clusters 3 and 4 as cell Type A, and 5 and 2 as cell type B.

# In[17]:


ligand_input = pd.DataFrame(subset_tf.obs.TF_clusters.map({'4': 0, '5': 0, '3': 1, '2': 1}))
ligand_input.columns = [selected_ligand]


covariates = pd.DataFrame(subset_tf.obs.TF_clusters.map({'3': 'A', '4': 'A',
                                                        '5': 'B', '2': 'B'}))
covariates.columns = ['celltype']


tf_output = pd.DataFrame(subset_tf.X, index = subset_tf.obs.index, columns = subset_tf.var.index)


# In[18]:


subset_tf.obs = pd.concat([covariates, ligand_input, pd.DataFrame({'TF_clusters': subset_tf.obs.TF_clusters})], axis = 1)


# In[19]:


viz_adata = subset_tf.copy()
for col in viz_adata.obs.columns:
    viz_adata.obs[col] = pd.Categorical(viz_adata.obs[col],
                                        categories = viz_adata.obs[col].unique())


fig, ax = plt.subplots(ncols = 3, figsize = (12, 4))
sc.pl.pca(viz_adata, color='celltype', ax = ax[0], show = False)
ax[0].set_title('Cell Type')

sc.pl.pca(viz_adata, color=selected_ligand, ax = ax[1], show = False)
ax[1].set_title('Ligand Stimulation')

sc.pl.pca(viz_adata, color='TF_clusters', ax = ax[2], show = False)
ax[2].set_title('Clusters')


fig.tight_layout()
plt.show()


# In[20]:


# linear scaling of inputs/outputs
projection_amplitude_in = 3
projection_amplitude_out = 1.2
# other parameters
bionet_params = {'target_steps': 100, 
                 'max_steps': 120, 
                 'exp_factor':50, 
                 'tolerance': 1e-5, 
                 'leak':1e-2, 
                'cat_max_norm': 1} 
vae_params = {'vae_batch_momentum': 0.01, 'vae_layer_norm': False, 'vae_dropout_rate': 0.1,
              'vae_activation_fn': nn.LeakyReLU,
              'vae_n_hidden_nodes': [1024, 768, 512], 
              'vae_var_min': 1e-4}
bionet_params = {**bionet_params, **vae_params}

# training parameters
me = 2000
lr_params = {'max_epochs': 2000, 'maximum_learning_rate': 2e-3, 'minimum_learning_rate': 2e-4,
                 'lr_restart_epoch': int(me/5), 'reset_optimizer_epoch': 200, 
                'lr_decay': 0.9, 'lr_restart_factor': 1, 'warmup_epochs': int(me/10)}

other_params = {'train_batch_size': 2056, 'test_batch_size': 512, 'validation_batch_size': 512, 
                    'network_noise_scale': 10, 'gradient_noise_scale': 1e-9}

regularization_params = {'input_lambda_L2': 1e-6, 'hidden_state_lambda_L2': 1e-6, 'bias_lambda_L2': 1e-6, 
                             'output_lambda_L2': 1e-6, 
                         'discriminator_lambda_L2': 1e-5,
                         'moa_lambda_L1': 0.1, #'ligand_lambda_L2': 1e-5, 
                         'uniform_lambda_L2': 1e-4,  
                         'uniform_max': (1/1.2), 
                         'spectral_loss_factor': 1e-5, 
                        'vae_lambda_l2': 1e-5, 
                        'vae_scaling_KL': 1e-2}
spectral_radius_params = {'n_probes_spectral': 5, 
                          'power_steps_spectral': 50, 
                          'subset_n_spectral': 10}

training_params = {**lr_params, **other_params, **regularization_params, **spectral_radius_params}
target_spectral_radius = 0.8

discriminator_params = {'batch_momentum': 0.01,
 'layer_norm': False,
 'dropout_rate': 0.1,
 'activation_fn': nn.LeakyReLU,
 'n_hidden_nodes': [16, 16, 16],
 'maximum_learning_rate': 2e-3,
 'minimum_learning_rate':2e-4,
 'lr_restart_epoch': int(me/5),
 'reset_optimizer_epoch': 200,
 'lr_decay': 0.9,
 'lr_restart_factor': 1,
 'warmup_epochs': int(me/10),
 'optimizer': torch.optim.Adam,
 'discriminator_lambda_L2': 1e-05, 
                       'discriminator_penalty_weight': 1}


# In[33]:


mod = SignalingModel(net = sn_ppis,
                     X_in = ligand_input,
                     y_out = tf_output, 
                     expr = subset_tf.to_df(), 
                     covariates = subset_tf.obs,
                     categorical_covariate_keys = ['celltype'],
                     projection_amplitude_in = projection_amplitude_in, projection_amplitude_out = projection_amplitude_out,
                     weight_label = weight_label, source_label = source_label, target_label = target_label,
                     bionet_params = bionet_params, 
                     dtype = torch.float32, device = device, seed = seed)


# In[22]:


# get pca on on selected features
subset_tf = subset_tf[:, mod.y_out.columns]
embed_tf_activity(subset_tf)


# # Start dev

# In[23]:


# # model setup
# mod.input_layer.weights.requires_grad = False # don't learn scaling factors for the ligand input concentrations
# mod.signaling_network.prescale_weights(target_radius = target_spectral_radius) # spectral radius

# # training loop
# trainer = TrainSC(mod = mod,
#                    prediction_optimizer = torch.optim.Adam,
#                    prediction_loss_fn = torch.nn.MSELoss(reduction='mean'),
#                   discriminator_params = discriminator_params,
#                    hyper_params = training_params,
#                    train_split = {'train': 0.8, 'test': 0.2, 'validation': None}, 
#                    train_seed = seed, 
#                    track_test = True,
#                    track_validation = False)


# In[24]:


# from scLEMBAS.model.train import *
# self = trainer


# In[25]:


# start_time = time.time()
# e = 0
# cur_lr = self.prediction_optimizer.param_groups[0]['lr']
# self.discriminator['_cur_lr'] = self.discriminator['optimizer'].param_groups[0]['lr']



# cur_loss = []
# cur_eig = []
# cur_loss_with_reg = []
# cur_pearson = []


# # iterate through batches
# if self.mod.seed:
#     utils.set_seeds(self.mod.seed + e)
# for batch, (X_in_, y_out_, covariates_idx_, expr_) in enumerate(self.train_dataloader):
#     break


# # end

# Next, let's train the model:

# In[34]:


# model setup
mod.input_layer.weights.requires_grad = False # don't learn scaling factors for the ligand input concentrations
mod.signaling_network.prescale_weights(target_radius = target_spectral_radius) # spectral radius

# training loop
trainer = TrainSC(mod = mod,
                   prediction_optimizer = torch.optim.Adam,
                   prediction_loss_fn = torch.nn.MSELoss(reduction='mean'),
                  discriminator_params = discriminator_params,
                   hyper_params = training_params,
                   train_split = {'train': 0.8, 'test': 0.2, 'validation': None}, 
                   train_seed = seed, 
                   track_test = True,
                   track_validation = False)
mod, stats_df = trainer.train_model(verbose = True)

torch.save(obj=mod.state_dict(), f=os.path.join(models_path, 'model_' + str(model_no) + '_state_dict.pth'))
io.write_pickled_object(trainer,  os.path.join(models_path, 'trainer_' + str(model_no) + '.pickle'))


# In[ ]:


trainer = io.read_pickled_object(os.path.join(models_path, 'trainer_' + str(model_no) + '.pickle'))
mod = trainer.mod
stats_df = trainer.stats_df.copy()
stats_df['epoch'] = stats_df.index + 1


# # To do: 
# - add discriminator learning rate visualization
# - add individual loss components visualization

# In[27]:


sns.lineplot(stats_df, x = 'epoch', y = 'learning_rate')


# In[28]:


fig, ax = plt.subplots(ncols = 2, figsize = (10, 4))

g = sns.lineplot(data = stats_df, x = 'epoch', y = 'train_loss_mean', ax = ax[0])
g = sns.lineplot(data = stats_df, x = 'epoch', y = 'test_loss_mean', ax = ax[0])
ax[0].set_ylabel('MSE Loss')
ax[0].set_xlabel('Epoch')
ax[0].set_yscale('log')

g = sns.lineplot(data = stats_df, x = 'epoch', y = 'train_pearson_mean', ax = ax[1])
g = sns.lineplot(data = stats_df, x = 'epoch', y = 'test_pearson_mean', ax = ax[1])
ax[1].set_title('Full TF Activity Space')
ax[1].set_ylabel('Sample-Wise Pearson Correlation')
ax[1].set_ylim([-1,1])
ax[1].set_xlabel('Epoch')

fig.suptitle('Best Hyperparameters')
fig.tight_layout()
("")


# Let's see what the train dataset will output:

# In[29]:


# inputs
X_train = mod.df_to_tensor(trainer.X_train)
y_train = mod.df_to_tensor(trainer.y_train)
expr_train = mod.df_to_tensor(mod.expr.loc[trainer.X_train.index, :])
covariates_idx_train = mod.signaling_network.covariates_to_tensor(sample_ids = trainer.X_train.index)

# run prediction
mod.eval()
with torch.inference_mode():
    Y_hat, Y_full, biases = mod(X_in = X_train, covariates_idx = covariates_idx_train, expr = expr_train)
    bias_global, bias_mu, bias_log_sigma_squared = biases

# formatting
y_predicted = pd.DataFrame(Y_hat.cpu().detach().numpy())
y_predicted.index, y_predicted.columns = trainer.y_train.index, trainer.y_train.columns


# In[52]:


pearsons = trainer.get_pearson_correlation(Y_hat, y_train, axis=1, return_mean=False)


# In[53]:


fig, ax = plt.subplots()
sns.histplot(pearsons, ax = ax)
ax.set_xlabel('Feature-wise TF activity Pearons: predicted vs actual')
("")


# In[108]:


pca_mod = subset_tf.uns['pca']['pca_mod']
rank = subset_tf.uns["pca"]['pca_rank']


md = subset_tf.obs
md['condition'] = md['celltype'].str.cat(md['Il5'].astype(str), sep='^')

X_pca = pd.DataFrame(subset_tf.obsm['X_pca'][:, :rank], 
                 index = subset_tf.obs.index, 
                columns = ['PC_{}'.format(i + 1) for i in range(rank)])
clf = NearestCentroid()
clf.fit(X_pca, md['condition'])
X_pca_centroids = pd.DataFrame(clf.centroids_, columns = clf.feature_names_in_, index = clf.classes_)


# In[117]:


fig, ax = plt.subplots(figsize = (10,5), ncols = 2)


viz_df = pd.concat([X_pca.loc[trainer.y_train.index, :], pd.DataFrame(md.loc[trainer.y_train.index, 'condition'])], 
                  axis = 1)
sns.scatterplot(data = viz_df,x = 'PC_1', y = 'PC_2', hue = 'condition', ax = ax[0])
ax[0].set_title('Actual Values')


y_pred_pca = pd.DataFrame(pca_mod.transform(y_predicted), index = y_predicted.index, 
                          columns = ['PC_{}'.format(i + 1) for i in range(pca_mod.n_components)]).iloc[:, :rank]
viz_df = pd.concat([y_pred_pca, pd.DataFrame(md.loc[trainer.y_train.index, 'condition'])], 
                  axis = 1)
sns.scatterplot(data = viz_df,x = 'PC_1', y = 'PC_2', hue = 'condition', ax = ax[1])
ax[1].set_title('Predicted Values')

fig.tight_layout()

