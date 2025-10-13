from typing import Dict
from collections import OrderedDict

import torch


class ContrastiveLoss:
    """
    Regularizes lack of separation by perturbation in instances where separation collapses 
    or perturbation signal in data is weak.
    """
    
    ALLOWED_METHODS = [
        'sc_actual', 
        'sc_predicted',
        'sc_actual_control', #'sc_actual_bidirectional',
        'sc_predicted_no_bc', 'no_collapse_no_bc',
        'sc_triplet', 'bulk_actual', 'bulk_predicted', 'bulk_triplet']

    def __init__(self,
                 methods: Dict[str, float],
                 underestimate_only=True, min_percentile=0.3, triplet_margin_frac=0.1):
        """Initializes hyperparameters for calculating regularization. 

        Parameters
        ----------
        methods : Dict[str, float]
            a dictionary with methods to use for regularization (keys) and constant to scale the final loss by (values)
        underestimate_only : bool, optional
            whether to regularize distances only if they are underestimates, by default True
            relevant for methods: 'bulk_actual'
        min_percentile : float, optional
            regularizes the distances that are under the min_percentile of the actual distances, 
            by default 0.3 
            must be between [0,1]
            relevant for methods: 'sc_actual', 'sc_triplet'
        triplet_margin_frac : float, optional
            multiplier for calculating the triplet margin, by default 0.1
            relevant for methods: 'bulk_triplet', 'sc_triplet'
        """
        assert len(set(methods).difference(self.ALLOWED_METHODS)) == 0, 'Please only use allowed contrastive loss methods'

        # if 'sc_actual_bidirectional' in methods: 
        #     if len(set(methods).intersection(['sc_actual', 'sc_actual_control'])) != 0:
        #         raise ValueError('Cannot specify both bidirectional and other actual methods')
        
        self.methods = methods
        self.underestimate_only = underestimate_only
        self.min_percentile = min_percentile
        self.triplet_margin_frac = triplet_margin_frac
        
        self._no_collapse_only = len(set(methods).difference(['sc_predicted_no_bc', 'no_collapse_no_bc'])) == 0
        self._need_y_out = len(set(methods).difference(['no_collapse_no_bc'])) != 0
        self._need_no_bc = any(m in self.methods for m in ['sc_predicted_no_bc', 'no_collapse_no_bc'])
        self._need_no_b =  any(m in self.methods for m in ['no_collapse_no_bc'])

        self._need_bulk = any(m in self.methods for m in ['bulk_actual', 'bulk_triplet', 'bulk_predicted'])
        self._need_pred_ctrl_centroid = any(m in self.methods for m in ['bulk_predicted', 'sc_predicted'])
        self._need_triplet = any(m in self.methods for m in ['bulk_triplet', 'sc_triplet'])

        self._need_actual_control = any(m in self.methods for m in ['sc_actual_control', 'sc_actual_bidirectional'])
        self._need_pert_centroid = self._need_bulk or self._need_triplet or self._need_actual_control
        self._need_pred_ctrl = self._need_pred_ctrl_centroid or self._need_actual_control

        self._need_sc = any(m in self.methods for m in ['sc_actual', 'sc_actual_bidirectional', 'sc_triplet', 'sc_predicted', 
                                                        'sc_predicted_no_bc'])
        self._need_neg_bulk = any(m in self.methods for m in ['bulk_actual', 'bulk_triplet'])
        self._need_neg_sc = any(m in self.methods for m in ['sc_actual', 'sc_actual_bidirectional', 'sc_triplet'])


        self.clear_attrs()
        
    def list_methods(self):
        print(self.ALLOWED_METHODS)

    def clear_attrs(self):
        # will hold batch-specific attributes after calling .update()
        self.cats = None
        self.control_mask = None
        self.pert_mask = None
        self.pert_ids = None
        self.combos = None

        self.Y_hat = None
        self.y_out = None
        self.Y_hat_no_bc = None
        self.Y_hat_no_b = None

    def update(self, 
               X_in, 
               covariates_idx, 
               run_adv, 
               y_out: None,
               Y_hat: None, 
               Y_hat_no_bc: None, 
               Y_hat_no_b: None
               ):
        """Cache all batch-specific inputs and masks."""
        # these help keep memory in check

        self.Y_hat = Y_hat
        self.y_out = y_out
        self.Y_hat_no_bc = Y_hat_no_bc
        self.Y_hat_no_b = Y_hat_no_b

        self.run_adv = run_adv

        # assert X_in.sum(axis = 1).max() <= 1, 'contrastive loss can currently only handle 1 perturbation per cell'
        # assert covariates_idx.size(1) == 1, "Contrastive regularization is currently only designed for one categorical covariate"

        # flatten covariates
        self.cats = covariates_idx.squeeze(1)

        self.control_mask = (X_in.sum(dim=1) == 0)
        self.pert_mask = ~self.control_mask
        self.pert_ids = X_in.argmax(dim=1)
        self.combos = torch.stack([self.cats[self.pert_mask], 
                                   self.pert_ids[self.pert_mask]], dim=1).unique(dim=0)
        self.update_calculations()

    def update_calculations(self):
        """Precompute common stats for each methods (cat, pid) combo and store in dictionary."""
        self.combo_cache = {}
        for cat, pid in self.combos:
            cat_mask = (self.cats == cat)
            ctrl_cat_mask = cat_mask & self.control_mask
            cp_mask = cat_mask & (self.pert_ids == pid) & self.pert_mask
            
            n_ctrl = int(ctrl_cat_mask.sum().item())
            n_pert = int(cp_mask.sum().item())

            if n_ctrl == 0 or n_pert == 0:
                self.combo_cache[(cat.item(), pid.item())] = None
            else:

                # method-specific calculations
                actual_ctrl, actual_ctrl_centroid = None, None
                actual_pert, pred_pert = None, None
                pred_ctrl, pred_ctrl_centroid = None, None
                actual_pert_centroid, pred_pert_centroid = None, None
                neg_dist_bulk, actual_dist_bulk = None, None
                neg_dist_sc, actual_dist_sc = None, None
                min_percentile_thresh = None
                pred_pert_no_bc, pred_ctrl_centroid_no_bc = None, None

                if self._need_y_out:
                    actual_ctrl = self.y_out[ctrl_cat_mask]
                    actual_ctrl_centroid = actual_ctrl.mean(dim=0, keepdim=True)
                    actual_pert = self.y_out[cp_mask]
                if not self._no_collapse_only:
                    pred_pert = self.Y_hat[cp_mask]
                
                if self._need_pred_ctrl:
                    pred_ctrl = self.Y_hat[ctrl_cat_mask]
                    if self._need_pred_ctrl_centroid:
                        pred_ctrl_centroid = pred_ctrl.mean(dim=0, keepdim=True).detach()
                if self._need_pert_centroid:
                    actual_pert_centroid = actual_pert.mean(dim=0, keepdim=True)
                if self._need_bulk:
                    pred_pert_centroid = pred_pert.mean(dim=0, keepdim=True)
                    actual_dist_bulk = torch.norm(actual_pert_centroid - actual_ctrl_centroid, p=2)
                    if self._need_neg_bulk:
                        neg_dist_bulk = torch.norm(pred_pert_centroid - actual_ctrl_centroid, p=2)
                if self._need_sc:
                    actual_dist_sc = torch.norm(actual_pert - actual_ctrl_centroid, p=2, dim=1)
                    min_percentile_thresh = torch.quantile(actual_dist_sc, self.min_percentile).detach()
                    if self._need_neg_sc:
                        neg_dist_sc = torch.norm(pred_pert - actual_ctrl_centroid, p=2, dim=1)
                if self._need_no_bc and self.run_adv:
                    pred_pert_no_bc = self.Y_hat_no_bc[cp_mask]
                    pred_ctrl_centroid_no_bc = self.Y_hat_no_bc[ctrl_cat_mask].mean(dim = 0, keepdim = True).detach()
                
                self.combo_cache[(cat.item(), pid.item())] = {
                    "pred_ctrl": pred_ctrl,
                    "actual_ctrl": actual_ctrl, # only used in sc_actual_control
                    "actual_ctrl_centroid": actual_ctrl_centroid,
                    "actual_pert_centroid": actual_pert_centroid,
                    "pred_pert_centroid": pred_pert_centroid,
                    "actual_pert": actual_pert,
                    "pred_pert": pred_pert,
                    "neg_dist_bulk": neg_dist_bulk, 
                    "actual_dist_bulk": actual_dist_bulk,
                    "neg_dist_sc": neg_dist_sc,
                    "actual_dist_sc": actual_dist_sc, 
                    "min_percentile_thresh": min_percentile_thresh, 
                    "pred_ctrl_centroid": pred_ctrl_centroid, 
                    "pred_pert_no_bc": pred_pert_no_bc, 
                    "pred_ctrl_centroid_no_bc": pred_ctrl_centroid_no_bc
                }

                if self._need_no_b and self.run_adv:
                    self.combo_cache[(cat.item(), pid.item())]['cp_mask'] = cp_mask
                    self.combo_cache[(cat.item(), pid.item())]['ctrl_cat_mask'] = ctrl_cat_mask


    def sc_actual(self):
        """
        Regularizes each perturbation prediction to encourage separation from actual control within the category. 

        1) For each category and predicted perturbation condition, calculates the Euclidean distance of the 
        perturbation (predicted and actual) and the respective control perturbation centroid (actual only).
        2) Identifies a percentile threshold of actual distances and penalizes those predicted distances below that threshold
        2) Calculates the loss as the square error between the predicted distance and actual distance.
        4) Takes the mean across conditions. 

        Considerations:
        - prediction contrast anchors to actual controls rather than predicted controls
        - only controls are contrasted, not pairwise
        - control anchor is the centroid (so not using pairwise distances)
        """
        lambda_scaler = self.methods.get('sc_actual', 0.0)
        if lambda_scaler == 0:
            return self.Y_hat.new_tensor(0.0)
        
        losses = []
        for (cat, pid), precalculated in self.combo_cache.items():
            if precalculated is None:
                continue


            pred_dist = precalculated["neg_dist_sc"] 
            min_percentile_thresh = precalculated['min_percentile_thresh']
            
            under_predicted = pred_dist[pred_dist < min_percentile_thresh]
            if under_predicted.numel() > 0:
                losses.append(torch.square(under_predicted - min_percentile_thresh).mean(axis = 0))
        return self._finalize(losses, lambda_scaler)
    
    def sc_actual_control(self):
        """
        Regularizes each control prediction to encourage separation from actual perturbation within the category. 
        The inverse direction of `sc_actual`. Call both methods to get a bidirectional regularization. 

        1) For each category and perturbation condition, calculates the Euclidean distance of the 
        control (predicted and actual) and the respective perturbation centroid (actual only).
        2) Identifies a percentile threshold of actual distances and penalizes those predicted distances below that threshold
        2) Calculates the loss as the square error between the predicted distance and actual distance.
        4) Takes the mean across conditions. 

        Considerations:
        - contrast anchors to actual pertrubations rather than predicted ones
        - only controls are contrasted, not pairwise across all perturbations
        - predicted anchor is the centroid (so not using pairwise distances)
        """
        lambda_scaler = self.methods.get('sc_actual_control', 0.0)
        if lambda_scaler == 0:
            return self.Y_hat.new_tensor(0.0)
        
        losses = []
        for (cat, pid), precalculated in self.combo_cache.items():
            if precalculated is None:
                continue

            actual_pert_centroid = precalculated['actual_pert_centroid']
            actual_dist_sc = torch.norm(precalculated['actual_ctrl'] - actual_pert_centroid, p = 2, dim = 1)
            min_percentile_thresh = torch.quantile(actual_dist_sc, self.min_percentile).detach()

            pred_dist = torch.norm(precalculated['pred_ctrl'] - actual_pert_centroid, p=2, dim=1)
            under_predicted = pred_dist[pred_dist < min_percentile_thresh]
            if under_predicted.numel() > 0:
                    losses.append(torch.square(under_predicted - min_percentile_thresh).mean(axis = 0))

        return self._finalize(losses, lambda_scaler)
    
    # def sc_actual_bidirectional(self):
    #     """Combines `sc_actual` and `sc_actual_control` by taking their sum, ensuring both control and pertrubation predictions
    #     are separated from each other (as in `sc_predicted`) and from respective actual data."""
    #     lambda_scaler = self.methods.get('sc_actual_bidirectional', 0.0)
    #     if lambda_scaler == 0:
    #         return self.Y_hat.new_tensor(0.0)

#         sc_actual_loss = self.sc_actual()
#         sc_actual_control_loss = self.sc_actual_control()
#         return sc_actual_loss + sc_actual_control_loss


    def sc_predicted(self):
        """
        Regularizes each perturbation prediction to encourage separation from predicted control within the category. 

        1) For each category and predicted perturbation condition, calculates the Euclidean distance of the 
        perturbation (predicted and actual) and the respective control perturbation centroid (predicted and actual, respectively).
        2) Identifies a percentile threshold of actual distances and penalizes those predicted distances below that threshold
        2) Calculates the loss as the square error between the predicted distance and actual distance.
        4) Takes the mean across conditions. 

        Considerations:
        - prediction contrast anchors to predicted controls rather than actual controls
        - only controls are contrasted, not pairwise
    - control anchor is the centroid (so not using pairwise distances)
        """
        lambda_scaler = self.methods.get('sc_predicted', 0.0)
        if lambda_scaler == 0:
            return self.Y_hat.new_tensor(0.0)
        
        losses = []
        for (cat, pid), precalculated in self.combo_cache.items():
            if precalculated is None:
                continue

            pred_pert = precalculated['pred_pert']
            pred_ctrl_centroid = precalculated['pred_ctrl_centroid']
            pred_dist = torch.norm(pred_pert - pred_ctrl_centroid, p=2, dim=1)
            
            min_percentile_thresh = precalculated['min_percentile_thresh']
            
            under_predicted = pred_dist[pred_dist < min_percentile_thresh]
            if under_predicted.numel() > 0:
                losses.append(torch.square(under_predicted - min_percentile_thresh).mean(axis = 0))
        return self._finalize(losses, lambda_scaler)
    
    def sc_predicted_no_bc(self):
        """Like sc_predicted, but run on forward pass without categorical bias."""
                
        lambda_scaler = self.methods.get('sc_predicted_no_bc', 0.0)
        if lambda_scaler == 0:
            return self.Y_hat.new_tensor(0.0)
        
        losses = []
        for (cat, pid), precalculated in self.combo_cache.items():
            if precalculated is None:
                continue

            pred_pert = precalculated['pred_pert_no_bc']
            pred_ctrl_centroid = precalculated['pred_ctrl_centroid_no_bc']
            pred_dist = torch.norm(pred_pert - pred_ctrl_centroid, p=2, dim=1)
            
            min_percentile_thresh = precalculated['min_percentile_thresh']
            
            under_predicted = pred_dist[pred_dist < min_percentile_thresh]
            if under_predicted.numel() > 0:
                losses.append(torch.square(under_predicted - min_percentile_thresh).mean(axis = 0))
        return self._finalize(losses, lambda_scaler)
    
    def no_collapse_no_bc(self):
        """Prevents forward pass f(A, X_pert, b_g) from shrinking relative to f(A, X_pert). 
        Does so by changing the perturbaiton predictions of f(A, X_pert_no_bc) (everything else is detached).
        """
        lambda_scaler = self.methods.get('no_collapse_no_bc', 0.0)
        if lambda_scaler == 0:
            return self.Y_hat.new_tensor(0.0)
        
        losses = []
        for (cat, pid), precalculated in self.combo_cache.items():
            if precalculated is None:
                continue

            # forward pass f(A, X_pert, b_g)
            pred_pert_centroid_no_bc = precalculated['pred_pert_no_bc'].mean(dim=0, keepdim=True)
            pred_ctrl_centroid_no_bc = precalculated['pred_ctrl_centroid_no_bc']
            no_bc_dist = torch.norm(pred_pert_centroid_no_bc - pred_ctrl_centroid_no_bc, p=2)

            # forward pass f(A, X_pert)
            # could take first index because without b_g, there is no variance, but adding noise to X_full 
            # in training makes them slightly different, so take centroid
            cp_mask = precalculated['cp_mask']
            ctrl_cat_mask = precalculated['ctrl_cat_mask']
            pred_pert_no_b = self.Y_hat_no_b[cp_mask].mean(dim=0, keepdim=True).detach() #[0, :].detach()
            pred_ctrl_no_b = self.Y_hat_no_b[ctrl_cat_mask].mean(dim=0, keepdim=True).detach() #[0, :].detach()
            no_b_dist = torch.norm(pred_pert_no_b - pred_ctrl_no_b, p = 2)

            if no_b_dist > no_bc_dist:
                losses.append(torch.square(no_b_dist - no_bc_dist))

        return self._finalize(losses, lambda_scaler)
    
    def _finalize(self, losses, lambda_scaler):
        if len(losses) > 0:
            return lambda_scaler * torch.mean(torch.stack(losses))
        return self.Y_hat.new_tensor(0.0)

    def get_loss(self):
        contrastive_losses = OrderedDict()
        for method_name in self.methods:
            if not self.run_adv and method_name in ['no_collapse_no_bc', 'sc_predicted_no_bc']:
                contrastive_losses[method_name] = self.Y_hat.new_tensor(0.0)
                continue

            contrastive_losses[method_name] = getattr(self, method_name)()

        # memory stuff
        self.combo_cache.clear()
        self.clear_attrs()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        return contrastive_losses 

#### TODO: deprecate below

    
    def sc_triplet(self):
        """
        Regularizes each perturbation to encourage separation from control within the category. 
        By using a triplet loss, ensures relative ordering of predicted perturbations w.r.t. 
        actual perturbed and actual control centroids.

        1) For each category and perturbation condition, calculates the Euclidean distance of the 
        predicted perturbed samples to the actual perturbed centroid (positive) 
        and to the actual control centroid (negative).
           - positive = d(pred_pert, actual_pert_centroid)
           - negative = d(pred_pert, actual_ctrl_centroid)

        2) Computes the triplet loss: max(0, positive - negative + margin). 
        The margin is a fraction (margin_frac) of the min_percentile of the 
        actual perturbed–control distances.

        3) Takes the mean across samples within each condition.

        4) Averages the loss across conditions.
        """
        lambda_scaler = self.methods.get('sc_triplet', 0.0)
        if lambda_scaler == 0:
            return self.Y_hat.new_tensor(0.0)
        
        losses = []
        for (cat, pid), precalculated in self.combo_cache.items():
            if precalculated is None:
                continue

            pred_pert = precalculated['pred_pert']
            actual_pert_centroid = precalculated['actual_pert_centroid']
            pos = torch.norm(pred_pert - actual_pert_centroid, p=2, dim = 1)
            
            neg = precalculated["neg_dist_sc"]
            margin = precalculated['min_percentile_thresh'] * self.triplet_margin_frac
            loss_distances = pos - neg + margin
            loss_distances = loss_distances[loss_distances > 0]
            if loss_distances.numel() > 0:
                losses.append(loss_distances.mean())
        return self._finalize(losses, lambda_scaler)
            
    def bulk_actual(self):
        """
        Regularizes each perturbation to encourage separation from control within the category. 

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
        """
        lambda_scaler = self.methods.get('bulk_actual', 0.0)
        if lambda_scaler == 0:
            return self.Y_hat.new_tensor(0.0)
        losses = []
        for (cat, pid), precalculated in self.combo_cache.items():
            if precalculated is None:
                continue

            pred_dist = precalculated["neg_dist_bulk"]
            target_dist = precalculated["actual_dist_bulk"]

            if (not self.underestimate_only) or (target_dist > pred_dist):
                losses.append(torch.square(target_dist - pred_dist))
        return self._finalize(losses, lambda_scaler)
    
    def bulk_predicted(self):
        """
        Regularizes each perturbation to encourage separation from control within the category. 

        1) For each category and predicted perturbation condition, calculates the centroid of the 
        perturbation (predicted and actual) and the respective control perturbation (predicted and actual respectively).
        2) Calculates the Euclidean distance between perturbed centroid and the control centroid for 
        predicted and actual data. 
        3) Calculates the loss as the square error between the predicted distance and actual distance.
        4) Takes the mean across conditions. 

        Considerations:
        - prediction contrast anchors to predicted controls rather than actual controls
        - only controls are contrasted, not pairwise
        - centroids are used, rather than individual points
        """
        lambda_scaler = self.methods.get('bulk_predicted', 0.0)
        if lambda_scaler == 0:
            return self.Y_hat.new_tensor(0.0)
        
        losses = []
        for (cat, pid), precalculated in self.combo_cache.items():
            if precalculated is None:
                continue

            pred_pert_centroid = precalculated["pred_pert_centroid"]
            pred_ctrl_centroid = precalculated["pred_ctrl_centroid"]
            pred_dist = torch.norm(pred_pert_centroid - pred_ctrl_centroid, p=2)
    
            target_dist = precalculated["actual_dist_bulk"]

            if (not self.underestimate_only) or (target_dist > pred_dist):
                losses.append(torch.square(target_dist - pred_dist))
        return self._finalize(losses, lambda_scaler)


    def bulk_triplet(self):
        """
        Regularizes each perturbation to encourage separation from control within the category. 
        By using a triplet loss, ensures relative ordering of prediction w.r.t. actual control and perturbation.

        1) For each category and predicted perturbation condition, calculates the centroid of the 
        perturbation (predicted and actual) and the respective control perturbation (actual only).
        2) Calculates the Euclidean distance between predicted perturbed centroid and the control centroid for 
        predicted and actual data 
        positive = d(pred_pert, actual_ctrl), negative = d(pred_pert, actual_pert) 
        3) Calculates the loss as positive - negative + margin. Marign is a fraction of d(actual_pert, actual_ctrl)
        Ensures that positive + m < negative.
        4) Takes the mean across conditions. 

        Considerations:
        - only controls are contrasted, not pairwise
        - centroids are used, rather than individual points
        """
        lambda_scaler = self.methods.get('bulk_triplet', 0.0)
        if lambda_scaler == 0:
            return self.Y_hat.new_tensor(0.0)
        losses = []
        for (cat, pid), precalculated in self.combo_cache.items():
            if precalculated is None:
                continue

            pred_pert_centroid = precalculated['pred_pert_centroid']
            actual_pert_centroid = precalculated['actual_pert_centroid']
            pos = torch.norm(pred_pert_centroid - actual_pert_centroid, p=2)
            
            neg = precalculated['neg_dist_bulk']
            actual_dist = precalculated['actual_dist_bulk']

            margin = self.triplet_margin_frac * actual_dist
            loss = torch.relu(pos - neg + margin) # clamps at 0
            losses.append(loss)
        return self._finalize(losses, self.methods.get('bulk_triplet', 0.0))
    
