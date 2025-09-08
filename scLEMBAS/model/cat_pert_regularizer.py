"""
Helper functions for building the model.
"""
from typing import Literal
from collections import OrderedDict

import torch
from torch import nn

import scLEMBAS.utilities as utils



class CatPertRegularizer:
    """Regularizes the categorical bias against the perturbation information in the model using various methods"""
    def __init__(self, 
#                  X_in, 
#                  covariates_idx, 
                 bionetwork_class,
#                  batch_size: int, 
                 regularization_scaler = 1,
                 method: Literal['orthogonality', 'info_nce', 'kl_divergence'] = 'orthogonality', 
                 per_label: bool = False, 
                 include_adjacency: bool = False, 
                 temperature: float = 0.1,):
        """
        Parameters
        ----------
        regularization_scaler: Union[float, int]
            scaling value by which to multiply the regularization term
        method : Literal['orthogonality', 'info_nce', 'kl_divergence'], optional
            how to regularize categorical bias against perturbation, by default 'orthogonality'
            - orthogonality: captures linear relationships with a Frobenius norm approach
            - info_nce: captures linear and nonlinear relationships with a contrastive loss approach
            - kl_divergence: penalizes deviation from a uniform similarity distribution between covariates and perturbation samples (minimally informative representation)
        per_label : bool, optional
            whether to apply regularization per label within a categorical group (True), or globall across all labels at once (False), by default False
        include_adjacency : bool, optional
            whether to include the activation function and adjacency matrix in the perturbation term (True) or just use the perturbation directly (False), by default False
        temperature : float, optional
            only relevant if method is info_nce or kl_divergence. A scaling factor applied to the similarity logits 
            before softmax/log-softmax. Lower values sharpen the distribution, increasing emphasis on the highest similarities. 
            Higher values produce a softer distribution. Default is 0.1.
        """
        
        self.include_adjacency = include_adjacency
        self.temperature = temperature
#         self.batch_size = batch_size
        self.method = method

        # attributes of bionetwork modle
        self.from_bionetwork(bionetwork_class)

        self.regularization_scaler = regularization_scaler

        if method in ['info_nce', 'kl_divergence']:
            self._bilinear_initialized = False

        _method_suffix = 'per_label' if per_label else 'global'
        if self.regularization_scaler == 0:
            self.regularizer = self.zero_regularizer
        else:
            self.regularizer = getattr(self, self.method + '_' + _method_suffix)

    def __call__(self):
        return self.regularizer()

    def from_bionetwork(self, bionetwork_class):
        """Retain some needed attributes from the bionetwork class"""
        
        self.device = bionetwork_class.device
        self.dtype = bionetwork_class.dtype
        self.seed = bionetwork_class.seed
        
        self.cat_embeddings = bionetwork_class.cat_embeddings
        self._cat_group_idx = bionetwork_class._cat_group_idx
        self.cat_unmasked_indices = bionetwork_class.cat_unmasked_indices

        if self.include_adjacency:
            self.weights = bionetwork_class.weights
            self.activation = bionetwork_class.activation
            self.bionet_params = bionetwork_class.bionet_params

        self.seed = bionetwork_class.seed
        
    def update_attributes(self, pert_in: torch.Tensor, covariates_idx: torch.Tensor):
        """
        Parameters
        ----------
        pert_in : torch.Tensor
            the ligand concentration inputs. 
            Shape is (samples x ligands) if not including adjacency. Same `X_in` input as to `ProjectInput.forward`
            Shape is (samples x nodes) if including adjacnecy. Same `X_full` output to `ProjectInput.forward`.
        covariates_idx : torch.Tensor
            rows correspond to samples as in X_full. Each column represents one categorical covariate group. Values
            in the columns represent the index mapping of the category label. Basically a rowsubset of `self.covariates_idx`.
            Same input as forward pass
        """
        self.covariates_idx = covariates_idx
        if not self.include_adjacency:
            self.pert_repr = pert_in
        else:
            self.pert_repr = self.activation(self.weights @ pert_in.T, self.bionet_params['leak']).T

        if self.method in ['info_nce', 'kl_divergence'] and not self._bilinear_initialized:
            self._init_bilinear_weights()

    def _init_bilinear_weights(self):
        """Initialize the random weights matrix for bilinear similarity in infoNCE"""
        _, pert_dim = self.pert_repr.shape
        for cat_group_idx in range(self.covariates_idx.shape[1]):
            cat_group = self._cat_group_idx[cat_group_idx]
            emb = self.cat_embeddings[cat_group](self.covariates_idx[:, cat_group_idx])
            emb = emb[:, self.cat_unmasked_indices]
            emb_dim = emb.shape[1]

            utils.set_seeds(self.seed)
            W = torch.randn(emb_dim, pert_dim, device=self.device)
            W = torch.nn.functional.normalize(W, dim=0)
            setattr(self, f"W_bilinear_group{cat_group}", W)
        self._bilinear_initialized = True
            
#     def _adjust_bilinear_weights(self, n_obs: int):
#         # slice bilinear weights to current batch size
#         for cat_group_idx in range(len(self._cat_group_idx)):
#             cat_group = self._cat_group_idx[cat_group_idx]
#             W_full = getattr(self, f"W_init_bilinear_group{cat_group}")
#             emb_dim, max_size = W_full.shape
#             if n_obs > max_size:
#                 raise ValueError(f"Batch size {n_obs} exceeds initialized max_size {max_size}. Reinit with larger batch_size.")
#             W = W_full[:, :n_obs]
#             setattr(self, f"W_bilinear_group{cat_group}", W)

    def zero_regularizer(self):
        return OrderedDict({'cat_bias_pert_loss': torch.tensor(0.0, device = self.device, dtype = self.dtype)})

    def orthogonality_global(self):
        """Enforces global orthogonality between categorical embedding and stimulation spaces.

        Returns
        -------
        : torch.Tensor
            Scalar loss (squared Frobenius norm of correlation between spaces)
        """
        reg_terms = [] 
        for cat_group_idx in range(self.covariates_idx.shape[1]): # iterate through covariates

            # get the embedding, excluding masked stimulation values
            cat_group = self._cat_group_idx[cat_group_idx]
            cat_embedding = self.cat_embeddings[cat_group](self.covariates_idx[:,cat_group_idx]).clone()
            cat_embedding = cat_embedding[:, self.cat_unmasked_indices]

            # column-wise mean centering (rows are batches/samples)
            cat_center = cat_embedding - cat_embedding.mean(dim=0, keepdim=True)
            stim_center = self.pert_repr - self.pert_repr.mean(dim=0, keepdim=True)

            # row-wise normalize to unit length 
            cat_norm = cat_center.norm(dim=1, keepdim=True) + 1e-8
            stim_norm = stim_center.norm(dim=1, keepdim=True) + 1e-8

            c = cat_center / cat_norm
            s = stim_center / stim_norm


            # pairwise cosine similarity between embeddings and stimulations, averaged across batches/samples
            dot_matrix = c.T @ s / c.shape[0] # (dim1 x dim2)

            # frobenius norm of dot product matrix
            reg_terms.append(torch.norm(dot_matrix, p='fro') ** 2) #tot_cos_similarity = tot_cos_similarity + torch.norm(dot_matrix, p='fro') ** 2 

        tot_cos_similarity = torch.mean(torch.stack(reg_terms))
        return OrderedDict({'cat_bias_pert_loss': self.regularization_scaler*tot_cos_similarity})

    def orthogonality_per_label(self):
        """
        Enforces per-label orthogonality between categorical embeddings and stimulation inputs.
        Uses squared Frobenius norm of cosine similarity between embeddings and stimulation vectors,
        computed separately within each category label (if ≥ 2 samples), weighted by sample size.

        Returns
        -------
        OrderedDict
            {'cat_bias_orthogonality_loss': scalar tensor}
        """

        group_weighted_losses = []
        for cat_group_idx in range(self.covariates_idx.shape[1]):
            cat_group = self._cat_group_idx[cat_group_idx]
            cat_ids = self.covariates_idx[:, cat_group_idx]
            cat_embedding_layer = self.cat_embeddings[cat_group]

            weighted_loss_sum = 0.0
            total_samples = 0

            for label in torch.unique(cat_ids):
                label_mask = (cat_ids == label)
                n_label = label_mask.sum().item()

                if n_label < 2:
                    continue  # skip singleton groups

                cat_embedding = cat_embedding_layer(cat_ids[label_mask])[:, self.cat_unmasked_indices]
                stim_sub = self.pert_repr[label_mask]

                # mean center
                cat_center = cat_embedding - cat_embedding.mean(dim=0, keepdim=True)
                stim_center = stim_sub - stim_sub.mean(dim=0, keepdim=True)

                # row-normalize
                cat_norm = cat_center.norm(dim=1, keepdim=True) + 1e-8
                stim_norm = stim_center.norm(dim=1, keepdim=True) + 1e-8

                c = cat_center / cat_norm
                s = stim_center / stim_norm

                # cosine similarity matrix
                dot_matrix = c.T @ s / c.shape[0]
                frob_loss = torch.norm(dot_matrix, p='fro') ** 2

                # weight by number of samples in this label
                weighted_loss_sum += n_label * frob_loss
                total_samples += n_label

            if total_samples > 0:
                group_weighted_losses.append(weighted_loss_sum / total_samples)

        if group_weighted_losses:
            tot_cos_similarity = torch.mean(torch.stack(group_weighted_losses))
            return OrderedDict({'cat_bias_pert_loss': self.regularization_scaler * tot_cos_similarity})
        else:
            return OrderedDict({'cat_bias_pert_loss': torch.tensor(0.0, device = self.device, dtype = self.dtype)})
        
        


    def info_nce_global(self):
        """
        Global InfoNCE loss. 
        **Note, used a random matrix to get bilinear similarity in the same dimension
        between perturbation and categorical embedding.
        """
        _, pert_dim = self.pert_repr.shape
        group_losses = []

        for cat_group_idx in range(self.covariates_idx.shape[1]):
            cat_group = self._cat_group_idx[cat_group_idx]
            cat_ids = self.covariates_idx[:, cat_group_idx]
            emb = self.cat_embeddings[cat_group](cat_ids)
            cat_emb = emb[:, self.cat_unmasked_indices]

            # Normalize both sides
            cat_emb = torch.nn.functional.normalize(cat_emb, dim=1)
            pert = torch.nn.functional.normalize(self.pert_repr, dim=1)

            N, _ = cat_emb.shape

            # Shared W for this group
            W = getattr(self, f"W_bilinear_group{cat_group}")

            # Forward: cat_emb → pert
            proj_cat = cat_emb @ W               # (N, pert_dim)
            logits_fwd = proj_cat @ pert.T       # (N, N)
            logits_fwd = logits_fwd / self.temperature
            labels = torch.arange(N, device=self.device)
            loss_fwd = torch.nn.functional.cross_entropy(logits_fwd, labels)

            # Reverse: pert → cat_emb
            proj_pert = pert @ W.T               # (N, emb_dim)
            logits_rev = proj_pert @ cat_emb.T   # (N, N)
            logits_rev = logits_rev / self.temperature
            loss_rev = torch.nn.functional.cross_entropy(logits_rev, labels)

            # Symmetric loss
            group_loss = (loss_fwd + loss_rev) / 2.0
            group_losses.append(group_loss)

        total_loss = torch.mean(torch.stack(group_losses)) 

        return OrderedDict({'cat_bias_pert_loss': self.regularization_scaler * total_loss})

    
    def info_nce_per_label(self):
        """
        Per-label InfoNCE loss using bilinear similarity between covariate embeddings and
        perturbation encodings. Encourages each covariate group to be locally uninformative
        about perturbation identity.

        Uses sample-size weighted mean across labels and covariate groups.
        """
        # For each covariate group, compute symmetric InfoNCE
        _, pert_dim = self.pert_repr.shape
        group_losses = []

        for cat_group_idx in range(self.covariates_idx.shape[1]):
            cat_group = self._cat_group_idx[cat_group_idx]
            cat_ids = self.covariates_idx[:, cat_group_idx]
            emb_layer = self.cat_embeddings[cat_group]

            # Get embedding dimension
            emb_dim = emb_layer(cat_ids[:1])[:, self.cat_unmasked_indices].shape[1]

            # Use pre-initialized W
            W = getattr(self, f"W_bilinear_group{cat_group}")

            total_samples = 0
            weighted_loss_sum = 0.0

            for label in torch.unique(cat_ids):
                label_mask = (cat_ids == label)
                n = label_mask.sum().item()
                if n < 2:
                    continue  # skip too-small groups

                # Subset
                emb = emb_layer(cat_ids[label_mask])[:, self.cat_unmasked_indices]  # (n, emb_dim)
                pert = self.pert_repr[label_mask]                                         # (n, pert_dim)

                # Normalize
                emb = torch.nn.functional.normalize(emb, dim=1)
                pert = torch.nn.functional.normalize(pert, dim=1)

                # Forward
                proj_cat = emb @ W
                logits_fwd = proj_cat @ pert.T / self.temperature
                labels = torch.arange(n, device=self.device)
                loss_fwd = torch.nn.functional.cross_entropy(logits_fwd, labels)

                # Reverse
                proj_pert = pert @ W.T
                logits_rev = proj_pert @ emb.T / self.temperature
                loss_rev = torch.nn.functional.cross_entropy(logits_rev, labels)

                loss = (loss_fwd + loss_rev) / 2.0

                weighted_loss_sum += n * loss
                total_samples += n

            if total_samples > 0:
                group_losses.append(weighted_loss_sum / total_samples)

        if group_losses:
            total_loss = torch.mean(torch.stack(group_losses))
            return OrderedDict({'cat_bias_pert_loss': self.regularization_scaler * total_loss})
        else:
            return OrderedDict({'cat_bias_pert_loss': torch.tensor(0.0, device = self.device, dtype = self.dtype)})

    def kl_divergence_global(self):
        """
        Global KL divergence against a uniform target over perturbations.
        Encourages categorical embeddings to be maximally uninformative
        about perturbation identity — i.e., uniform similarity over samples.
        
        **Not alternate option was a empirical prior distribution over perturbations, but 
        this is still carrying some sort of "information" on perturbation.


        Returns
        -------
        OrderedDict
            {'cat_bias_pert_loss': scalar tensor}
            
        """
        group_losses = []

        # Forward direction (cat → pert)
        for cat_group_idx in range(self.covariates_idx.shape[1]):
            cat_group = self._cat_group_idx[cat_group_idx]
            cat_ids = self.covariates_idx[:, cat_group_idx]
            cat_emb = self.cat_embeddings[cat_group](cat_ids)[:, self.cat_unmasked_indices]
            cat_emb = torch.nn.functional.normalize(cat_emb, dim=1)

            pert = torch.nn.functional.normalize(self.pert_repr, dim=1)
            N = cat_emb.shape[0]
            W = getattr(self, f"W_bilinear_group{cat_group}")

            # logits and uniform targets
            logits = (cat_emb @ W) @ pert.T / self.temperature
            log_probs = torch.nn.functional.log_softmax(logits, dim=1)
            target = torch.full_like(log_probs, fill_value=1.0 / N)

            kl_forward = (target * (torch.log(target + 1e-8) - log_probs)).sum(dim=1).mean()

            if self.include_adjacency:
                # Reverse direction (pert → cat)
                logits_rev = (pert @ W.T) @ cat_emb.T / self.temperature
                log_probs_rev = torch.nn.functional.log_softmax(logits_rev, dim=1)
                target_rev = torch.full_like(log_probs_rev, fill_value=1.0 / N)

                kl_reverse = (target_rev * (torch.log(target_rev + 1e-8) - log_probs_rev)).sum(dim=1).mean()
                loss = (kl_forward + kl_reverse) / 2.0
            else:
                loss = kl_forward

            group_losses.append(loss)

        total_loss = torch.mean(torch.stack(group_losses))
        return OrderedDict({'cat_bias_pert_loss': self.regularization_scaler * total_loss})

    def kl_divergence_per_label(self):
        """
        Per-label KL divergence using uniform target distribution. Optionally adds
        symmetric reverse KL from perturbation → category embedding.

        Parameters
        ----------
        self.temperature : float
            self.temperature for similarity scaling.

        Returns
        -------
        OrderedDict
            {'cat_bias_pert_loss': scalar tensor}
        """
        group_losses = []

        for cat_group_idx in range(self.covariates_idx.shape[1]):
            cat_group = self._cat_group_idx[cat_group_idx]
            cat_ids = self.covariates_idx[:, cat_group_idx]
            emb_layer = self.cat_embeddings[cat_group]
            W = getattr(self, f"W_bilinear_group{cat_group}")

            total_samples = 0
            weighted_loss_sum = 0.0

            for label in torch.unique(cat_ids):
                label_mask = (cat_ids == label)
                n = label_mask.sum().item()
                if n < 2:
                    continue

                cat_emb = emb_layer(cat_ids[label_mask])[:, self.cat_unmasked_indices]
                cat_emb = torch.nn.functional.normalize(cat_emb, dim=1)

                pert = self.pert_repr[label_mask]
                pert = torch.nn.functional.normalize(pert, dim=1)

                logits = (cat_emb @ W) @ pert.T / self.temperature
                log_probs = torch.nn.functional.log_softmax(logits, dim=1)
                target = torch.full_like(log_probs, fill_value=1.0 / n)

                kl_forward = (target * (torch.log(target + 1e-8) - log_probs)).sum(dim=1).mean()

                if self.include_adjacency:
                    logits_rev = (pert @ W.T) @ cat_emb.T / self.temperature
                    log_probs_rev = torch.nn.functional.log_softmax(logits_rev, dim=1)
                    target_rev = torch.full_like(log_probs_rev, fill_value=1.0 / n)

                    kl_reverse = (target_rev * (torch.log(target_rev + 1e-8) - log_probs_rev)).sum(dim=1).mean()
                    loss = (kl_forward + kl_reverse) / 2.0
                else: 
                    loss = kl_forward

                weighted_loss_sum += n * loss
                total_samples += n

            if total_samples > 0:
                group_losses.append(weighted_loss_sum / total_samples)

        if group_losses:
            total_loss = torch.mean(torch.stack(group_losses))
            return OrderedDict({'cat_bias_pert_loss': self.regularization_scaler * total_loss})
        else:
            return OrderedDict({'cat_bias_pert_loss': torch.tensor(0.0, device = self.device, dtype = self.dtype)})

        

        