"""
Helper functions for data visualization. 
"""
import math
from typing import Literal

import anndata
import pandas as pd
import numpy as np
import plotnine as p9


palette_map = {'Set1': ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3','#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999'], 
               'Set2': ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3'],
              'Set3': ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3','#fdb462', '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd', '#ccebc5', '#ffed6f']}

def shade_plot(X: np.array, Y: np.array, sigma: np.array, x_label: str, y_label: str, 
              width: int = 5, height: int = 3):
    """_summary_

    Parameters
    ----------
    X : np.array
        x axis values
    Y : np.array
        y axis values
    sigma : np.array
        standard deviation of y axis values
    x_label : str
        x axis label 
    y_label : str
        y axis label

    Returns
    -------
    plot : plotnine.ggplot.ggplot
        _description_
    """
    data = pd.DataFrame(data = {
        x_label: X, y_label: Y, 'sigma': sigma
    })
    data['sigma_min'] = data[y_label] - data.sigma
    data['sigma_max'] = data[y_label] + data.sigma

    plot = (
        p9.ggplot(data, p9.aes(x=x_label)) +
        p9.geom_line(p9.aes(y=y_label), color = '#1E90FF') +
        p9.geom_ribbon(p9.aes(y = y_label, ymin = 'sigma_min', ymax = 'sigma_max'), alpha = 0.2) +
        p9.xlim([0, data.shape[0]]) +
        # p9.ylim(10**min_log, round(data[y_label].max(), 1)) +
        # p9.scale_y_log10() + 
        p9.theme_bw() + 
        p9.theme(figure_size=(width, height))
    )
    return plot

def plot_embedding(adata: anndata.AnnData, group_label: str, embedding: str = 'pca', 
                   palette: Literal['Set1', 'Set2', 'Set3'] = 'Set1', width: int = 5, height: int = 3):
    """Scatter plot of reduced space. 

    Parameters
    ----------
    adata : anndata.AnnData
        the anndata object
    group_label : str
        the categorical variable in `adata.obs` to color the points by
    embedding : str, optional
        which embedding to use, by default 'pca'
    palette : Literal['Set1', 'Set2', 'Set3'], optional
        which embedding to use, by default 'Set1'
    width : int, optional
        figure width, by default 5
    height : int, optional
        figure height, by default 3
    """
    embedding_col_map = {'pca': 'PC_', 'umap': 'UMAP_'}
    md = adata.obs.copy()
    X = pd.DataFrame(adata.obsm['X_' + embedding][:, :2], index = md.index, 
                         columns = [embedding_col_map[embedding] + str(i+1) for i in range(2)])
    col_labels = [' '.join(i.split('_')) for i in X.columns]
    X = pd.concat([X, md[group_label]], axis = 1)

    p = (
        p9.ggplot(X, p9.aes(x=X.columns[0], y = X.columns[1], color = group_label)) +
        p9.geom_point() +
        p9.xlab(col_labels[0]) + p9.ylab(col_labels[1]) + 
        p9.theme_bw() + p9.theme(figure_size=(width, height)) + 
        p9.scale_color_manual(values=palette_map[palette])
    )
    return p