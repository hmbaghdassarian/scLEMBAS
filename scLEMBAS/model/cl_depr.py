"""Regularizes lack of separation by perturbation in instances where separation collapses or perturbation signal in data is weak."""

import torch


def bulk_actual(Y_hat, y_out, X_in, covariates_idx, 
                        lambda_scaler: float, 
                        underestimate_only: bool = True, 
                        ):
    """Regularizes each perturbation to encourage separation from control within the category. 

    1) For each category and predicted perturbation condition, calculates the centroid of the 
    perturbation (predicted and actual) and the respective control perturbation (actual only).
    2) Calculates the Euclidean distance between perturbed centroid and the control centroid for 
    predicted and actual data. 
    3) Calculates the loss as the square error between the predicted distance and actual distance.
    4) Takes the mean across conditions. 

    Considerations:
    - prediction contrast anchors to actual controls rather than predicted controls
    - only controls are contrasted, not pairwise
    - centroids are used, rather than individual points

    Parameters
    ----------
    Y_hat : _type_
        the predicted output (samples x features)
    y_out : _type_
        the actual output (samples x features)
    X_in : _type_
        the one-hot encoded perturbations (samples x features)
    covariates_idx : _type_
        the discrete representation of categorical covariate (samples x 1) 
        * can only currently handle one categorical covariate
    lambda_scaler : float, 
        scaling term for the loss
    underestimate_only : bool, optional
        whether to regularize distances only if they are underestimates, by default True

    Returns
    -------
    regularization_loss
        the regularization term
    """
    if lambda_scaler == 0:
        return Y_hat.new_tensor(0.0)
    
    # assert X_in.sum(axis = 1).max() <= 1, 'contrastive loss can currently only handle 1 perturbation per cell'
    # assert covariates_idx.size(1) == 1, "Contrastive regularization is currently only designed for one categorical covariate"

    # flatten covariates
    cats = covariates_idx.squeeze(1)

    control_mask = (X_in.abs().sum(dim=1) == 0)
    pert_mask = ~control_mask
    pert_ids = X_in.argmax(dim=1)  # [N], valid even if row is control (but we’ll mask)

    losses = []
    # all (cell, pert) combos present in this batch (among perturbed rows)
    combos = torch.stack([cats[pert_mask], pert_ids[pert_mask]], dim=1).unique(dim=0)
    for cat, pid in combos:
        cat_mask = (cats == cat)
        ctrl_cat_mask = cat_mask & control_mask
        cp_mask = cat_mask & (pert_ids == pid) & pert_mask # need and pert_mask when using argmax

        n_ctrl = int(ctrl_cat_mask.sum().item())
        n_pert = int(cp_mask.sum().item())

        if n_ctrl == 0 or n_pert == 0:
            continue

    #     # control centroids (pred and true, kept separate)
    #     pred_ctrl_centroid = Y_hat[ctrl_cat_mask].mean(dim=0, keepdim=True)
    #     if detach_ctrl_grad:
    #         pred_ctrl_centroid = pred_ctrl_centroid.detach()
        actual_ctrl_centroid = y_out[ctrl_cat_mask].mean(dim=0, keepdim=True)

        # per (cat, pert) centroids
        pred_pert_centroid = Y_hat[cp_mask].mean(dim=0, keepdim=True)
        actual_pert_centroid = y_out[cp_mask].mean(dim=0, keepdim=True)

        # distances: pred vs pred_ctrl, target vs true_ctrl
        pred_dist = torch.norm(pred_pert_centroid - actual_ctrl_centroid, p=2)
        target_dist = torch.norm(actual_pert_centroid - actual_ctrl_centroid, p=2)

        if not underestimate_only or (underestimate_only and target_dist > pred_dist):
            loss = torch.square(target_dist - pred_dist) # squared deviation with asymmetric weighting
            losses.append(loss)
        else:
            continue

    
    if len(losses) != 0:
        return lambda_scaler * torch.stack(losses).mean()
    else:
        return Y_hat.new_tensor(0.0)
    
def sc_actual(Y_hat, y_out, X_in, covariates_idx, 
                        lambda_scaler: float, 
                        min_percentile: float = 0.3, 
                        ):
    """Regularizes each perturbation to encourage separation from control within the category. 

    1) For each category and predicted perturbation condition, calculates the Euclidean distance of the 
    perturbation (predicted and actual) and the respective control perturbation centroid (actual only).
    2) Identifies a percentile threshold of actual distances and penalizes those predicted distances below that threshold
    2) Calculates the loss as the square error between the predicted distance and actual distance.
    4) Takes the mean across conditions. 

    Considerations:
    - prediction contrast anchors to actual controls rather than predicted controls
    - only controls are contrasted, not pairwise
    - control anchor is the centroid (so not using pairwise distances)

    Parameters
    ----------
    Y_hat : _type_
        the predicted output (samples x features)
    y_out : _type_
        the actual output (samples x features)
    X_in : _type_
        the one-hot encoded perturbations (samples x features)
    covariates_idx : _type_
        the discrete representation of categorical covariate (samples x 1) 
        * can only currently handle one categorical covariate
    lambda_scaler : float, 
        scaling term for the loss
    min_percentile : float, optional
        regularizes the distances that are under the min_percentile of the actual distances, by default True

    Returns
    -------
    regularization_loss
        the regularization term
    """
    if lambda_scaler == 0:
        return Y_hat.new_tensor(0.0)

    # assert X_in.sum(axis = 1).max() <= 1, 'contrastive loss can currently only handle 1 perturbation per cell'
    # assert covariates_idx.size(1) == 1, "Contrastive regularization is currently only designed for one categorical covariate"

    # flatten covariates
    cats = covariates_idx.squeeze(1)

    control_mask = (X_in.abs().sum(dim=1) == 0)
    pert_mask = ~control_mask
    pert_ids = X_in.argmax(dim=1)  # [N], valid even if row is control (but we’ll mask)

    losses = []
    # all (cell, pert) combos present in this batch (among perturbed rows)
    combos = torch.stack([cats[pert_mask], pert_ids[pert_mask]], dim=1).unique(dim=0)
    for cat, pid in combos:
        cat_mask = (cats == cat)
        ctrl_cat_mask = cat_mask & control_mask
        cp_mask = cat_mask & (pert_ids == pid) & pert_mask # need and pert_mask when using argmax

        n_ctrl = int(ctrl_cat_mask.sum().item())
        n_pert = int(cp_mask.sum().item())
        
        if n_ctrl == 0 or n_pert == 0:
            continue

    #     # control centroids (pred and true, kept separate)
    #     pred_ctrl_centroid = Y_hat[ctrl_cat_mask].mean(dim=0, keepdim=True)
    #     if detach_ctrl_grad:
    #         pred_ctrl_centroid = pred_ctrl_centroid.detach()
        actual_ctrl_centroid = y_out[ctrl_cat_mask].mean(dim=0, keepdim=True)
        
        # per (cat, pert) centroids
        pred_pert = Y_hat[cp_mask]
        actual_pert = y_out[cp_mask]
        
        # individual points
        pred_dist = torch.norm(pred_pert - actual_ctrl_centroid, p=2, dim=1)   
        actual_dist = torch.norm(actual_pert - actual_ctrl_centroid, p=2, dim=1) 
        
        min_percentile_thresh = torch.quantile(actual_dist, min_percentile).detach()
        
        under_predicted = pred_dist[pred_dist < min_percentile_thresh] 
        if under_predicted.numel() != 0:
            loss = torch.square(under_predicted - min_percentile_thresh).mean(axis = 0)
            losses.append(loss)
        else:
            continue
            
    if len(losses) != 0:
        return lambda_scaler * torch.stack(losses).mean()
    else:
        return Y_hat.new_tensor(0.0)
  