# TODO: replace CatDiscriminator with this, that uses the native CE label_smoothing parameter
    
    
class CatDiscriminator(nn.Module):
    """"Discriminator for categorical covariates in adversarial training of scLEMBAS.
    Adapted from scVI's `Classifier`.
    """
    
    DEFAULT_HYPER_PARAMS = {**FCLayers.DEFAULT_HYPER_PARAMS, 
                            **{'n_hidden_nodes': [16, 16, 16]}, 
                           'bionet_activation': False, 
                           'smooth_labels': False, 
                           'epsilon_smooth': 0.1}
    
    def __init__(
        self,
        n_features_in: int,
        n_labels: int,
        n_hidden_nodes: List[int] = [16, 16, 16],
        batch_momentum: float = 0.01,
        layer_norm: bool = False,
        spectral_norm: bool = False,
        dropout_rate: int | float = 0.1,
        activation_fn: nn.Module | None = nn.LeakyReLU,
        bionet_activation: bool = False, 
        rnn_params: dict = {'activation_function': 'MML', 'leak': 0.01}, 
        smooth_labels: bool = False, 
        epsilon_smooth: float = 0.1,
        # initialize: bool = True,
        dtype: torch.dtype=torch.float32,
        device: str = 'cpu', 
        seed: int = 888, 
    ):
        """Initialize discriminator

        Parameters
        ----------
        n_features_in : int
            number of inpute features to discriminator (should be number of latent features for scLEMBAS)
        n_labels : int
            number of categories for the given categorical covariate
        n_hidden_nodes : List[int], optional
            number of hidden nodes per hidden layer, by default [64]
            each element in the list corresponds to one hidden layer (i.e., no. of hidden layers = length of list), by default [16, 16, 16]
        batch_momentum : float, optional
            `momentum` parameter for `BatchNorm` layer, by default .01
            If None, a `BatchNorm` is not added
        layer_norm : bool, optional
            whether to have `LayerNorm` layers or not, by default False
        spectral_norm : bool, optional
            whether to apply spectral normalization (`torch.nn.utils.spectral_norm`) to the linear layers of the model
        dropout_rate : int | float, optional
            dropout rate to apply to each of the hidden layers, by default 0.1
            If None, dropout is not added
        activation_fn : nn.Module | None, optional
            non-linear Pytorch activation function, by default nn.ReLU. No activation if set to None
        bionet_activation: bool, optional
            whether to classify the direct generator output (False) or 
            the output once it's been run through the RNN (True) (i.e., run through the bionet activation function).
            The motivation here is from obervations that any lingering information (particularly non-linear) in the global bias 
            will be amplified by the non-linear function of the RNN
        rnn_params: dict, optional
          bionetwork 'activation_function' and 'leak'; see `model.bionetwork.forward` for details; necessary to know the exact
          bionet activation function being used
          only required if bionet_activation is True
        smooth_labels: bool, optional
            whether to smooth the labels (with epsilon parameter) (True) or not (False), by default False
        epsilon_smooth: float, optional
            if smoothing labels, epsilon parameter to apply smoothing with
        initialize: bool, optional
            whether to initialize the linear layer weights using `torch.nn.init`, by default True
        dtype : torch.dtype, optional
            datatype to store values in torch, by default torch.float32
        device : str
            whether to use gpu ("cuda") or cpu ("cpu"), by default "cpu"
        seed : int
            random seed for torch and numpy operations, by default 888
        """
        super().__init__()
        self.n_labels = n_labels
        self.device = device
        self.dtype = dtype
        self.seed = seed
        self.bionet_activation = bionet_activation
        self.smooth_labels = smooth_labels
        assert 0 < epsilon_smooth < 1, "epsilon_smooth must be between (0,1)"
        self.epsilon_smooth = epsilon_smooth
        

        if self.n_labels > 2: # multi-class
            if self.smooth_labels:
                self.loss_fn = nn.CrossEntropyLoss(label_smoothing = self.epsilon_smooth) #self.smooth_multi_loss
                # self.eval_loss_fn = self._kl_with_hard_labels # deprecated
            else:
                self.loss_fn = nn.CrossEntropyLoss() # applies softmax to logits prior to CE
            self.eval_loss_fn = nn.CrossEntropyLoss()
            out_features = self.n_labels
        elif self.n_labels == 2: # binary
            if self.smooth_labels:
                self.loss_fn = self.smooth_binary_loss
                self.eval_loss_fn = nn.BCEWithLogitsLoss()
            else:
                self.loss_fn = nn.BCEWithLogitsLoss() # applies sigmoid to logits prior to CE
                self.eval_loss_fn = self.loss_fn
            out_features = 1
        else:
            raise ValueError('There are no distinct classes.')

        cat_layers = []
        if self.bionet_activation: 
            cat_layers.append(activation_function_torch_map[rnn_params['activation_function']](leak = rnn_params['leak']))
        if self.seed:
            set_seeds(self.seed)
        cat_layers.append(FCLayers(layers = [n_features_in] + n_hidden_nodes, 
                                   batch_momentum = batch_momentum, # since bias is just a vector 
                                   layer_norm = layer_norm, 
                                   spectral_norm = spectral_norm,
                                   dropout_rate = dropout_rate, 
                                   activation_fn = activation_fn, 
                                #    initialize = initialize,
                                   dtype = self.dtype, device = self.device))
        cat_layers.append(FCLayers(layers = [n_hidden_nodes[-1], out_features], 
                                   dtype = self.dtype, device = self.device,
                                   batch_momentum = None, layer_norm = False, 
                                   spectral_norm = spectral_norm, dropout_rate = None, activation_fn = None))
                                #   initialize = initialize))

        self.classifier = nn.Sequential(*cat_layers)

    def forward(self, x):
        """Returns logits for labels"""
        return self.classifier(x) 
    
    def smooth_binary_loss(self, logits, labels):
        smoothed_labels = labels * (1 - self.epsilon_smooth) + self.epsilon_smooth
        return F.binary_cross_entropy_with_logits(logits, smoothed_labels, reduction='mean')

    # deprecated: used smoothing with KL div rather than native parameter in CE    
    # def smooth_multi_loss(self, logits, labels):
    #     one_hot = F.one_hot(labels, num_classes=self.n_labels)
    #     smoothed_labels = one_hot * (1 - self.epsilon_smooth) + self.epsilon_smooth / self.n_labels
    #     log_probs = F.log_softmax(logits, dim=1)
    #     return F.kl_div(log_probs, smoothed_labels, reduction='batchmean')
    
    # def _kl_with_hard_labels(self, logits, labels):
    #     one_hot = F.one_hot(labels, num_classes=self.n_labels).float()
    #     log_probs = F.log_softmax(logits, dim=1)
    #     return F.kl_div(log_probs, one_hot, reduction='batchmean')
    
    def L2_reg(self, lambda_L2: Annotated[float, Ge(0)] = 0):
        """Get the L2 regularization term for the linear layers' parameters.
        
        Parameters
        ----------
        lambda_2 : Annotated[float, Ge(0)]
            the regularization parameter, by default 0 (no penalty) 
        
        Returns
        -------
        regularization_loss : torch.Tensor
            the regularization term
        """
        regularization_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        if lambda_L2 != 0:
            for layer in self.classifier.modules():
                if isinstance(layer, nn.Linear):
                    regularization_loss += torch.sum(torch.square(layer.weight))
                    if layer.bias is not None:
                        regularization_loss += torch.sum(torch.square(layer.bias))
            regularization_loss = lambda_L2 * regularization_loss
        
        return regularization_loss

    def get_probability(self, y):
        """Calculate the probability from output logits."""
        if self.n_labels > 2:
            return F.softmax(y.detach(), dim=-1)
        else:
            return F.sigmoid(y.detach(), dim = -1) # probability of the "positive" (labeled "1") layer
        
    def random_loss(self, class_probs, train_mode = True):
        """
        Calculates the expected loss of a discriminator with random predictions.

        To estimate the baseline KL divergence for a random discriminator under label smoothing, 
        we compute the average KL divergence between each smoothed one-hot label 
        (i.e., the softened target for each class) and a fixed "random" prediction distribution (class-weighted). 
        This simulates the expected loss a discriminator would incur if it always output the same uninformative 
        prediction, allowing us to benchmark how well it is performing compared to chance. 
        By averaging across all possible true classes, we account for the full range of label configurations 
        the discriminator might encounter.


        Parameters
        ----------
        class_probs 
            an array of the probabilities of each class for weighting unbalanced classes in random loss calculation
        """
        if len(class_probs) != self.n_labels:
            raise ValueError('The class_probs must be the same length as self.n_labels')
            
        eps = self.epsilon_smooth if (train_mode and self.smooth_labels) else 0.0
        if self.n_labels == 2:
            # Binary classification baseline
            p1 = class_probs[1]
            p0 = class_probs[0]

            # Model outputs random prediction equal to p1
            y_pred = p1

            # Target labels after smoothing
            t1 = 1.0 - eps
            t0 = eps

            # BCE(t, y_pred) = -[t * log(y_pred) + (1 - t) * log(1 - y_pred)]
            bce_1 = - (t1 * np.log(y_pred + 1e-10) + (1 - t1) * np.log(1 - y_pred + 1e-10))
            bce_0 = - (t0 * np.log(y_pred + 1e-10) + (1 - t0) * np.log(1 - y_pred + 1e-10))

            return p1 * bce_1 + p0 * bce_0
        
        
        else:
            total_ce = 0.0
            for i in range(self.n_labels):
                # Smoothed target for true class i
                target = np.full(self.n_labels, eps / self.n_labels, dtype=np.float32)
                target[i] = 1.0 - eps + eps / self.n_labels

                # CE = -sum(p_true * log(p_pred))
                ce_i = -np.sum(target * np.log(class_probs + 1e-10))
                total_ce += class_probs[i] * ce_i

            return total_ce
#             if self.smooth_labels: # KL divergence (used in training, kept for eval but with eps = 0)
#                 total_kl = 0

#                 Q = torch.tensor(class_probs, dtype=torch.float32) # random prediction
#                 log_Q = torch.log(Q + 1e-10)

#                 for i in range(self.n_labels):
#                     # Smoothed target for class i
#                     one_hot = np.zeros(self.n_labels)
#                     one_hot[i] = 1
#                     smoothed = one_hot * (1 - eps) + eps / self.n_labels

#                     P = torch.tensor(smoothed, dtype=torch.float32) # label-smoothing adjusted true target

#                     kl = F.kl_div(log_Q, P, reduction='sum')
#                     total_kl += class_probs[i] * kl.item() # gives a weighted average
#                 return total_kl
#             else: # cross entropy
#                 return -np.sum(class_probs * np.log(class_probs)) 


        
