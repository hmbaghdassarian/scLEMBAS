"""
Appropriately format the signaling network topology to be used as input to the model.
"""

import itertools
from typing import Literal, Union, List 
from tqdm import tqdm

import omnipath as op
import pandas as pd
import networkx as nx


ppi_link = 'https://zenodo.org/records/10823116/files/organism_omnipath_ppi_03_15_24.csv'

def load_network(network_type: str = 'omnipath', 
                 organism: Literal['human', 'rat', 'mouse'] = 'human',
                static: bool = True):
    """Loads a full PPI network from Omnipath.

    Parameters
    ----------
    network_type : str, optional
        _description_, by default 'omnipath'
    organism : Literal['human', 'rat', 'mouse'], optional
        which organism genes should be derived from, by default 'human'
    static : bool, optional
        whether to download a static (from 03/15/24) version of the Omnipath PPI DB or the most current, by default True
        for stable results and consistency with downstream analyses, recommended to use static = True
    
    Returns
    -------
    sn_ppis : pd.DataFrame
        an edge list representing the signaling network
    """

    
    if network_type == 'omnipath': 
        if not static:
            sn_ppis = op.interactions.OmniPath().get(genesymbols = True, organisms = organism)
        else:
            sn_ppis = pd.read_csv(ppi_link.replace('organism', organism), index_col = 0) # static PPI as of 03/15/24
    else:
        raise ValueError('Only omnipath networks can be loaded right now')
    
    return sn_ppis

def extract_network(sn_ppis: pd.DataFrame, 
                              curation_effort_thresh: int = 5, n_references_thresh: int = 3,
                              resources: Union[List, str] = ['HuRI','IntAct','KEGG-MEDICUS','NetPath','Reactome_SignaLink3','SPIKE','SignaLink3','SIGNOR', 
                                                            'Baccin2019', 'Ramilowski2015', 'Reactome_LRdb', 'UniProt_LRdb', 'CellChatDB', 'CellPhoneDB', 'connectomeDB2020', 'scConnect'],
                            drop_self: bool = True,
                             verbose: bool = True):
    """Various filters on the ppi network.

    Parameters
    ----------
    sn_ppis : pd.DataFrame
        an edge list representing the signaling network, output of `load_signaling_network`
    curation_effort_thresh : int, optional
        threshold of curation effort to retain interaction, by default 5
    n_references_thresh : int, optional
        threshold of number of references to retain interaction, by default 3
    resources : Union[List, str], optional
        resources from which to retain interactions, by default ['HuRI','IntAct','KEGG-MEDICUS','NetPath','Reactome_SignaLink3','SPIKE','SignaLink3','SIGNOR']
        if None or 'all', will not filter for resources
    verbose : bool, optional
        get rid of self-interacting nodes, by default True
    verbose : bool, optional
        print status of network extraction, by default True

    Returns
    -------
    sn_ppis : pd.DataFrame
        an edge list representing the filtered signaling network
    """

    n_int_a = sn_ppis.shape[0]
    
    sn_ppis = sn_ppis[(sn_ppis.n_references >= n_references_thresh) & (sn_ppis.curation_effort >= curation_effort_thresh)]
    if verbose:
        n_int_b = sn_ppis.shape[0]
        print('The thresholds filtered {}  of {} interactions'.format(n_int_a - n_int_b, n_int_a))
    
    if resources and resources != 'all':
        n_sources = sn_ppis.sources.apply(lambda x: len(set(x.split(';')).intersection(resources)))
        sn_ppis = sn_ppis[n_sources > 0]
        if verbose:
            n_int_c = sn_ppis.shape[0]
            print('The resources filtered {}  of {} interactions'.format(n_int_b - n_int_c, n_int_b))

    sn_ppis.drop_duplicates(subset = ['source', 'target'], keep = False, inplace = True)
    if drop_self:
        sn_ppis = sn_ppis[sn_ppis[['source', 'target']].apply(lambda x: x.nunique() == 2, axis = 1)]

    return sn_ppis

def fully_connected_network(sn_ppis: pd.DataFrame, ligand_labels: List[str], tf_labels: List[str], source_label: str, target_label: str, 
                           path_finder: Literal['all', 'shortest', 'connected'] = 'connected'):
    """Filter the input network for those interactions that provide full paths between ligands and transcription factors. 

    Parameters
    ----------
    sn_ppis : pd.DataFrame
        an edge list representing the signaling network
    ligand_labels : List[str]
        the list of ligands
    tf_labels : List[str]
        the list of transcription factors
    source_label : str
        the column label for source nodes in the graph
    target_label : str
        the column label for the target node in the graph
    path_finder : Literal['all', 'shortest', 'connected']], optional
        method by which to identify the interactions to retain, by default 'connected'. Options include: 
            - 'all': filter interactions by finding all paths between the sources and targets (can be very slow)
            - 'shortest': filter interactions by finding the shortest path between the sources and targets
            - 'connected': filter interactions by finding those that contain nodes which are connected to both the sources and the targets

    Returns
    -------
    sn_ppis : pd.DataFrame
        an edge list representing the filtered signaling network
    """
    
    # get interactions for all paths b/w source and target in input network
    G = nx.from_pandas_edgelist(sn_ppis, source_label, target_label,
                                create_using = nx.DiGraph() if sn_ppis[sn_ppis.is_directed].shape[0] == sn_ppis.shape[0] else None)
    
    all_nodes = sorted(G.nodes)
    ligand_labels = set(ligand_labels).intersection(all_nodes)
    tf_labels = set(tf_labels).intersection(all_nodes)

    if path_finder in ['all', 'shortest']:
        all_edges = set()
        if path_finder == 'all':
            for source, target in tqdm(list(itertools.product(ligand_labels, tf_labels))):
                for path in nx.all_simple_paths(G, source=source, target=target):
                    all_edges.update(zip(path, path[1:]))
        elif path_finder == 'shortest':
            for source, target in tqdm(list(itertools.product(ligand_labels, tf_labels))):
                if nx.shortest_paths.has_path(G, source=source, target=target):
                    path = nx.shortest_path(G, source=source, target=target)
                    all_edges.update(zip(path, path[1:]))
    
        # filter input network 
        all_connected = pd.DataFrame(columns = [source_label, target_label], index = range(len(all_edges)))
        for idx, interaction in enumerate(all_edges):
            all_connected.loc[idx, :] = [interaction[0], interaction[1]]
        sn_ppis = sn_ppis.merge(all_connected, on = [source_label, target_label])
    elif path_finder == 'connected':
        tf_connected = set()
        for source in all_nodes: 
            for target in tf_labels:
                if source == target or nx.shortest_paths.has_path(G, source, target):
                    tf_connected.add(source)
                    break
        
        both_connected = set()
        for source in tf_connected: 
            for target in ligand_labels:
                if source == target or nx.shortest_paths.has_path(G, source, target):
                    both_connected.add(source)
        sn_ppis = sn_ppis[sn_ppis[[source_label, target_label]].apply(lambda row: row.isin(both_connected).all(), axis=1)]
        
    return sn_ppis

def format_network(sn_ppis: pd.DataFrame, 
                   weight_label: str = 'mode_of_action', 
                   stimulation_label: str = 'consensus_stimulation', 
                   inhibition_label: str = 'consensus_inhibition') -> pd.DataFrame:
    """Formats the standard sn_ppiswork file format to that needed by `SignalingModel.parse_sn_ppiswork`

    Parameters
    ----------
    sn_ppis : pd.DataFrame
        signaling sn_ppiswork adjacency list with the following columns:
            - `weight_label`: whether the interaction is stimulating (1) or inhibiting (-1) or unknown (0.1). Exclude non-interacting (0) nodes. 
            - `stimulation_label`: binary whether an interaction is stimulating (1) or [not stimultaing or unknown] (0)
            - `inhibition_label`: binary whether an interaction is inhibiting (1) or [not inhibiting or unknown] (0)
    weight_label : str, optional
        converts `stimulation_label` and `inhibition_label` to a single column of stimulating (1), inhibiting (-1), or
        unknown (0.1), by default 'mode_of_action'
    stimulation_label : str, optional
        column name of stimulating interactions, see `sn_ppis`, by default 'stimulation'
    inhibition_label : str, optional
        column name of inhibitory interactions, see `sn_ppis`, by default 'inhibition'

    Returns
    -------
    formatted_sn_ppis : pd.DataFrame
        the same dataframe with the additional `weight_label` column
    """
    if sn_ppis[(sn_ppis[stimulation_label] == 1) & (sn_ppis[inhibition_label] == 1)].shape[0] > 0:
        raise ValueError('An interaction can either be stimulating (1,0), inhibition (0,1) or unknown (0,0)')
    
    formatted_sn_ppis = sn_ppis.copy()
    formatted_sn_ppis[weight_label] = 0
    formatted_sn_ppis.loc[formatted_sn_ppis[stimulation_label] == 1, weight_label] = 1
    formatted_sn_ppis.loc[formatted_sn_ppis[inhibition_label] == 1, weight_label] = -1
    
    #ensuring that lack of known MOA does not imply lack of representation in scipy.sparse.find(A)
    formatted_sn_ppis[weight_label] = formatted_sn_ppis[weight_label].replace(0, 0.1)
    formatted_sn_ppis[weight_label] = formatted_sn_ppis[weight_label].replace(np.nan, 0.1)

    return formatted_sn_ppis