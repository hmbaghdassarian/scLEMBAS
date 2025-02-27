Here, we ask how each component of the scLEMBAS model structure effects the prediction:
- i) Get the prediction using the full model, and separately removing certain components from the forward pass (bias components or adjacency matrix), to see how each component separately contributes to the prediction. We also calculate the EMD loss of these predictions.
- ii) Visualize the prediction from the complete forward pass
- iii) Visualie the predictions with various components removed
- iv) Visualize the bias vector from the predictions (not the TF output prediction)

In instances where we use the latent space (for visualization or calculations), we project the predictions into the latent space of the acual data with the existing PC model calculated on the actual data in Notebook 03. We then recalculate the neighbors graph and re-run clustering on the combined projected data, using the resolution identified in notebook 03 that optimized NMI between condition and cluster label.  

For a given test condition, we are always predicting the same cell type in the opposite stimulation. However, given the flexibility of the counterfactual, we can make the predictions of the OOD cells from a number of gene expression inputs. Specifically, we can predict from the following gene expression inputs:
- in-distribution: all train cells
- opposite: for each test condition, we predict from the same cell type but opposite stimulation condition (these are all in-distribution as well)

Additional options not explored:
- OOD: test cells only
- all: all cells (in-distribution + OOD)
- stimulated: in-distribution stimulated cells
- unstimulated: in-distribution control cells
- cell type: prediction from a specific in-distribution cell type
- condition: prediction from a specific in-distribution cell type and stimulation