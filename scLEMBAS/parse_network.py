"""
Appropriately format the signaling network topology to be used as input to the model.
"""

import itertools
from typing import Literal, Union, List, Any 
from tqdm import tqdm
import warnings

import omnipath as op
import pandas as pd
import numpy as np
import networkx as nx

ppi_link = 'https://zenodo.org/records/11477837/files/organism_omnipath_ppi_05_24_24.csv'

def load_network(network_type: str = 'omnipath', 
                 organism: Literal['human', 'rat', 'mouse'] = 'human',
                static: bool = True, 
                fill_na: bool = True):
    """Loads a full PPI network from Omnipath.

    Parameters
    ----------
    network_type : str, optional
        resource from which to get network, by default 'omnipath'
    organism : Literal['human', 'rat', 'mouse'], optional
        which organism genes should be derived from, by default 'human'
    static : bool, optional
        whether to download a static (from 03/15/24) version of the Omnipath PPI DB or the most current, by default True
        for stable results and consistency with downstream analyses, recommended to use static = True
    fill_na : bool, optional
        whether to fill the "n_references" and "curation_effort" columns that have NA values with a minimum value. Useful for 
        future thresholding, by default True. 
    
    Returns
    -------
    sn_ppis : pd.DataFrame
        an edge list representing the signaling network (likely a superset of the signaling network, since considering all PPIs)
    """

    
    if network_type == 'omnipath': 
        if not static: # from https://workflows.omnipathdb.org/networks-py.html
            sn_ppis = op.interactions.PostTranslational.get(genesymbols = True, directed = True, organism = organism)
        else: # same command as not static above, but generated on 05/24/24 (to ensure reproducibility)
            sn_ppis = pd.read_csv(ppi_link.replace('organism', organism), index_col = 0) 
    else:
        raise ValueError('Only omnipath networks can be loaded right now')

    # fill threshold values with less than the minimum
    if fill_na:
        sn_ppis['n_references'] = sn_ppis.n_references.fillna(sn_ppis.n_references.min() - 1)
        sn_ppis['curation_effort'] = sn_ppis.curation_effort.fillna(sn_ppis.curation_effort.min() - 1)
    return sn_ppis

def flatten_list(list1: List[List[Any]]) -> list:
    """Create a single list from a  list of lists.

    Parameters
    ----------
    list1 : List[List[Any]]
        a list of lists

    Returns
    -------
    list
        a single list
    """
    # https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists
    return [item for sublist in list1 for item in sublist]

def correct_moa(sn_ppis: pd.DataFrame,
                stimulation_label: str = 'consensus_stimulation',
                inhibition_label: str = 'consensus_inhibition'):
    """In the case where mode of action is True for both stimulating and inhibiting, make this an unknown MOA.

    Parameters
    ----------
    sn_ppis : pd.DataFrame
        an edge list representing the signaling network, output of `scLEMBAS.parse_network.load_network`
    stimulation_label : str, optional
        column name of stimulating interactions, see `sn_ppis`, by default 'consensus_stimulation'
    inhibition_label : str, optional
        column name of inhibitory interactions, see `sn_ppis`, by default 'consensus_inhibition'

    Returns
    -------
    sn_ppis : pd.DataFrame
        a copy of the input interaction network with duplicates aggregated
    """
    sn_ppis = sn_ppis.copy()
    sn_ppis.loc[sn_ppis[(sn_ppis[stimulation_label] == 1) & (sn_ppis[inhibition_label] == 1)].index, 
    [stimulation_label, inhibition_label]] = [False, False]
    
    return sn_ppis

def drop_duplicate_interactions(sn_ppis: pd.DataFrame, 
                             source_label: str = 'source_genesymbol',
                             target_label: str = 'target_genesymbol',
                             stimulation_label: str = 'consensus_stimulation',
                             inhibition_label: str = 'consensus_inhibition'):
    """Systematically aggregate any duplicate interactions between the source and target node. 

    Parameters
    ----------
    sn_ppis : pd.DataFrame
        an edge list representing the signaling network, output of `scLEMBAS.parse_network.load_network`
    source_label : str
        the column label for source nodes in the graph, by default 'source_genesymbol'
    target_label : str
        the column label for the target node in the graph, by default 'target_genesymbol'
    stimulation_label : str, optional
        column name of stimulating interactions, see `sn_ppis`, by default 'consensus_stimulation'
    inhibition_label : str, optional
        column name of inhibitory interactions, see `sn_ppis`, by default 'consensus_inhibition'

    Returns
    -------
    sn_ppis : pd.DataFrame
        a copy of the input interaction network with duplicates aggregated
    """

    sn_ppis = sn_ppis.copy()
    duplicated_interactions = sn_ppis[sn_ppis.duplicated(subset = [source_label, target_label], keep='first')].drop_duplicates(subset = [source_label, target_label])
    unique_vals = pd.DataFrame(columns = sn_ppis.columns)
    drop_indeces = []
    for idx in duplicated_interactions.index:
        source = duplicated_interactions.loc[idx, source_label]
        target = duplicated_interactions.loc[idx, target_label]

        dup_int = sn_ppis[(sn_ppis[source_label] == source) & (sn_ppis[target_label] == target)].copy()
        drop_indeces += dup_int.index.tolist()
        dup_int.reset_index(inplace = True, drop = True)

        if (dup_int[[source_label, target_label, stimulation_label, inhibition_label, 'n_references', 'curation_effort', 'references' ]].nunique() == 1).all():
            dup_int = pd.DataFrame(dup_int.iloc[0, :]).T
        else:
            # filter by n_references
            dup_int = dup_int[dup_int.n_references == dup_int.n_references.max()]
            # filter by curation effort
            if dup_int.shape[0] > 1:
                dup_int = dup_int[dup_int.curation_effort == dup_int.curation_effort.max()]
            # merge remaining
            if dup_int.shape[0] > 1:
                if dup_int[stimulation_label].nunique() > 1:
                    dup_int[stimulation_label] = False
                if dup_int[inhibition_label].nunique() > 1:
                    dup_int[stimulation_label] = False
                references = sorted(set(flatten_list([ref.split(';') if not (type(ref) == float and np.isnan(ref)) else [ref] for ref in dup_int.references.tolist()])))
                references = [ref for ref in references if not (type(ref) == float and np.isnan(ref))]
                n_references = len(references)
                dup_int = dup_int.iloc[0, :]
                dup_int['n_references'] = n_references
                dup_int['references'] = ';'.join(references) if len(references) > 0 else np.nan
                dup_int = pd.DataFrame(dup_int).T
        if dup_int.shape[0] != 1:
            raise ValueError('There are still duplicated interactions')

        unique_vals = pd.concat([unique_vals, dup_int], axis = 0)

    sn_ppis.drop(index = drop_indeces, inplace = True)
    sn_ppis = pd.concat([sn_ppis, unique_vals], axis = 0)
    sn_ppis.reset_index(inplace = True, drop = True)
    if sn_ppis.duplicated(subset = [source_label, target_label]).any():
        raise ValueError('Interaction DB still has duplicates')

    return sn_ppis
    
def correct_network(sn_ppis: pd.DataFrame,
                     source_label: str = 'source_genesymbol',
                     target_label: str = 'target_genesymbol',
                     stimulation_label: str = 'consensus_stimulation',
                     inhibition_label: str = 'consensus_inhibition'):
    """
    Corrects mode of action: In the case where MOA is positive for both stimulating and inhibiting, make this an
    unknown MOA.
    Drops duplicates: systematically aggregate any duplicate interactions between the source and target node. 

    Parameters
    ----------
    sn_ppis : pd.DataFrame
        an edge list representing the signaling network, output of `scLEMBAS.parse_network.load_network`
    source_label : str
        the column label for source nodes in the graph, by default 'source_genesymbol'
    target_label : str
        the column label for the target node in the graph, by default 'target_genesymbol'
    stimulation_label : str, optional
        column name of stimulating interactions, see `sn_ppis`, by default 'consensus_stimulation'
    inhibition_label : str, optional
        column name of inhibitory interactions, see `sn_ppis`, by default 'consensus_inhibition'

    Returns
    -------
    sn_ppis : pd.DataFrame
        a copy of the input interaction network with duplicates aggregated
    """
    sn_ppis = correct_moa(sn_ppis, stimulation_label, inhibition_label)
    sn_ppis = drop_duplicate_interactions(sn_ppis, source_label, target_label, stimulation_label, inhibition_label)

    return sn_ppis

def add_omnipath_interaction(sn_ppis: pd.DataFrame, 
                             interactions_to_add: str | List[str], 
                             moa: List[Literal[1, 0, -1, None]] = None,
                             delim: str = '-',
                             source_label: str = 'source_genesymbol',
                             target_label: str = 'target_genesymbol',
                             stimulation_label: str = 'consensus_stimulation',
                             inhibition_label: str = 'consensus_inhibition'):
    """Adds custom interactions to Omnipath PPI network output from `load_network`.
    Note, Assumes the mode of action is unknown. 
    Note, adds the resource type as "Custom", so must include this in the `resources` argument of `extract_network` to retain the interaction.

    Parameters
    ----------
    sn_ppis : pd.DataFrame
        output of `scLEMBAS.parse_network.load_network`
    interactions_to_add : str | List[str]
        the source and target to add in the format "<source_id><delim><target_id>"
    moa: List[Literal[1, 0, -1, None]]
        each element corresponds to the mechanism of action for the interaction in `interactions_to_add`, 
        by default None (will set all values to 0)
            - 1: stimulating interaction
            - 0 or None: unknown mechanism of action
            - -1: inhibiting interaction
    delim: str, optional;
        the character separating the source and target for each element in the `interactions_to_add`
    source_label : str
        the column label for source nodes in the graph, by default 'source_genesymbol'
    target_label : str
        the column label for the target node in the graph, by default 'target_genesymbol'
    stimulation_label : str, optional
        column name of stimulating interactions, see `sn_ppis`, by default 'consensus_stimulation'
    inhibition_label : str, optional
        column name of inhibitory interactions, see `sn_ppis`, by default 'consensus_inhibition'
    """
    if type(interactions_to_add) == str:
        interactions_to_add = [interactions_to_add]
    if moa is None:
        moa = [0]*len(interactions_to_add)

    for idx, interaction in enumerate(interactions_to_add):
        source, target = interaction.split(delim)
        moa_ = moa[idx]
        
        # add the interaction
        init = [np.nan]*sn_ppis.shape[1]
        init[sn_ppis.columns.tolist().index(source_label)] = source
        init[sn_ppis.columns.tolist().index(target_label)] = target
        if moa is None or moa == 0:
            init[sn_ppis.columns.tolist().index(stimulation_label)] = False
            init[sn_ppis.columns.tolist().index(inhibition_label)] = False
        elif moa == 1:
            init[sn_ppis.columns.tolist().index(stimulation_label)] = True
            init[sn_ppis.columns.tolist().index(inhibition_label)] = False
        elif moa == -1:
            init[sn_ppis.columns.tolist().index(stimulation_label)] = False
            init[sn_ppis.columns.tolist().index(inhibition_label)] = True
    
        # ensure that these PPIs will not be excluded during thresholding
        init[sn_ppis.columns.tolist().index('sources')] = 'Custom'
        init[sn_ppis.columns.tolist().index('curation_effort')] = np.inf
        init[sn_ppis.columns.tolist().index('n_references')] = np.inf
        init[sn_ppis.columns.tolist().index('is_directed')] = True
    
        sn_ppis.loc[sn_ppis.shape[0], :] = init

    return sn_ppis

def extract_network(sn_ppis: pd.DataFrame, 
                              curation_effort_thresh: int = 5, n_references_thresh: int = 3,
                              resources: Union[List, str] = 'all',
                            drop_self: bool = True,
                             source_label: str = 'source_genesymbol',
                             target_label: str = 'target_genesymbol',
                             verbose: bool = True):
    """Various filters on the ppi network.

    Parameters
    ----------
    sn_ppis : pd.DataFrame
        an edge list representing the signaling network
    curation_effort_thresh : int, optional
        threshold of curation effort to retain interaction, by default 5
    n_references_thresh : int, optional
        threshold of number of references to retain interaction, by default 3
    resources : Union[List, str], optional
        resources from which to retain interactions, by default 'all' which won't filter out resources
    drop_self : bool, optional
        get rid of self-interacting nodes, by default True
    source_label : str
        the column label for source nodes in the graph, by default 'source_genesymbol'
    target_label : str
        the column label for the target node in the graph, by default 'target_genesymbol'
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

    if drop_self:
        sn_ppis = sn_ppis[sn_ppis[[source_label, target_label]].apply(lambda x: x.nunique() == 2, axis = 1)]

    return sn_ppis

def drop_nodes(nodes: List[str], 
               sn_ppis: pd.DataFrame,
               source_label: str,
               target_label: str):
    """Drops nodes and all their interactions from the network. 

    Parameters
    ----------
    nodes : List[str]
        IDs of the nodes to drop
    sn_ppis : pd.DataFrame
        an edge list representing the signaling network
    source_label : str
        the column label for source nodes in the graph
    target_label : str
        the column label for the target node in the graph
    """
    drop_idx = sn_ppis[(sn_ppis[source_label].isin(nodes)) | (sn_ppis[target_label].isin(nodes))].index
    sn_ppis = sn_ppis.drop(index = drop_idx).reset_index(drop = True)
    
    return sn_ppis

def map_connections(sn_ppis: pd.DataFrame, 
                      ligand_labels: List[str], 
                      tf_labels: List[str], 
                      source_label: str, 
                      target_label: str
                     ):
    """Maps each ligand to each TF it has a path to.

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

    Returns
    -------
    ligand_connections : Dict[str, List[str]]
        a dictionary with keys as ligands and values as the list of TFs that the ligand has a connected path to
    """
    ligand_connections = {label: [] for label in ligand_labels}
    
    G = nx.from_pandas_edgelist(sn_ppis, source_label, target_label,
                            create_using = nx.DiGraph() if sn_ppis[sn_ppis.is_directed].shape[0] == sn_ppis.shape[0] else None)
    all_nodes = sorted(G.nodes)
    ligand_labels = set(ligand_labels).intersection(all_nodes)
    tf_labels = set(tf_labels).intersection(all_nodes)

    
    for source, target in list(itertools.product(ligand_labels, tf_labels)):
         if nx.shortest_paths.has_path(G, source=source, target=target):
                ligand_connections[source] += [target]
 
    return ligand_connections

def create_connected_network(sn_ppis: pd.DataFrame, ligand_labels: List[str], tf_labels: List[str], source_label: str, target_label: str, 
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
    ligand_connections : Dict[str, List[str]]
        a dictionary with keys as ligands and values as the list of TFs that the ligand has a connected path to
    """
    
    # get interactions for all paths b/w source and target in input network
    G = nx.from_pandas_edgelist(sn_ppis, source_label, target_label,
                                create_using = nx.DiGraph() if sn_ppis[sn_ppis.is_directed].shape[0] == sn_ppis.shape[0] else None)

    all_nodes = sorted(G.nodes)
    ligand_labels_unfiltered = ligand_labels.copy()
    ligand_labels = set(ligand_labels).intersection(all_nodes)
    tf_labels = set(tf_labels).intersection(all_nodes)

    if path_finder in ['all', 'shortest']:
        ligand_connections = {label: [] for label in ligand_labels_unfiltered}
        all_edges = set()
        if path_finder == 'all':
            for source, target in tqdm(list(itertools.product(ligand_labels, tf_labels))):
                all_path_counter = 0
                for path in nx.all_simple_paths(G, source=source, target=target):
                    all_edges.update(zip(path, path[1:]))
                    all_path_counter += 1
                if all_path_counter > 0:
                    ligand_connections[source] += [target]
        elif path_finder == 'shortest':
            for source, target in tqdm(list(itertools.product(ligand_labels, tf_labels))):
                if nx.shortest_paths.has_path(G, source=source, target=target):
                    ligand_connections[source] += [target]
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
    #     sn_ppis = sn_ppis[sn_ppis[[source_label, target_label]].apply(lambda row: row.isin(both_connected).all(), axis=1)]
        sn_ppis = sn_ppis[sn_ppis[source_label].isin(both_connected) & sn_ppis[target_label].isin(both_connected)]
        ligand_connections = map_connections(sn_ppis, ligand_labels_unfiltered, tf_labels, source_label, target_label)

    return sn_ppis, ligand_connections

def stringent_connected_network(all_ppis: pd.DataFrame,
                                input_labels: List[str], 
                                output_labels: List[str], 
                                threshold_on: Literal['n_references', 'curation_effort']  = 'n_references',
                                min_curation_effort: int = None, 
                               min_references_thresh: int = None, 
                               resources: str = 'all', 
                               drop_self: bool = True,
                                source_label: str = 'source_genesymbol',
                                target_label: str = 'target_genesymbol', 
                               path_finder: Literal['all', 'shortest', 'connected'] = 'connected'):

    """
    Creates the most "stringent" extracted network that still contains all input nodes connected to atleast one output node, 
    if possible. Stringency is defined by thresholding on the number of references or curation effort. 

    This can replace `extract_network` and `create_connected_network` .

    Parameters
    ----------
    all_ppis : pd.DataFrame
        the input signaling network 
    input_labels : List[str]
        the list of input nodes to the signaling network (e.g., ligands)
    output_labels : List[str]
        the list of output nodes to the signaling network (e.g., transcription factors)
    threshold_on : Literal['n_references', 'curation_effort']
        whether to threshold on the references or the curation threshold, by default 'n_references'
        The two values are highly correlated and should give similar results. 
    min_curation_effort : int, optional
        the starting threshold for the curation effort, by default not thresholded
    min_references_thresh : int, optional
        the starting threshold for the number of references, by default not thresholded
    resources : Union[List, str], optional
        resources from which to retain interactions, by default 'all' which won't filter out resources
    drop_self : bool, optional
        whether to drop self-interacting nodes, by default True
    source_label : str
        the column label for source nodes in the graph, by default 'source_genesymbol'
    target_label : str
        the column label for the target node in the graph, by default 'target_genesymbol'
    path_finder : Literal['all', 'shortest', 'connected']], optional
        method by which to identify the interactions to retain, by default 'connected'. Options include: 
            - 'all': filter interactions by finding all paths between the sources and targets (can be very slow)
            - 'shortest': filter interactions by finding the shortest path between the sources and targets
            - 'connected': filter interactions by finding those that contain nodes which are connected to both the sources and the targets

    Returns
    -------
    sn_ppis : pd.DataFrame
        an edge list representing the filtered signaling network at its most stringent threshold
    n_references_thresh : int
        the final threshold for the number of references
    curation_effort_thresh : int
        the final threshold for the curation effort
    """

    ppi_list = []
    if all_ppis.n_references.isna().any():
        all_ppis['n_references'] = all_ppis.n_references.fillna(all_ppis.n_references.min() - 1)
    if all_ppis.curation_effort.isna().any():
        all_ppis['curation_effort'] = all_ppis.curation_effort.fillna(all_ppis.curation_effort.min() - 1)
    
    # setting to min is the same as no thresholds, so start at + 1
    if not min_curation_effort: 
        curation_effort_thresh = all_ppis.curation_effort.min() # this starts at no thresholding
    else:
        curation_effort_thresh = min_curation_effort
    
    if not min_references_thresh: 
        n_references_thresh = all_ppis.n_references.min() # this starts at no thresholding
    else:
        n_references_thresh = min_references_thresh
    
    
    all_nodes_ = sorted(set(all_ppis[source_label].tolist() + all_ppis[target_label].tolist()))
    if len(set(input_labels).difference(all_nodes_)) != 0:
        warnings.warn('Not all input_labels are in the starting network')
    
    counter = 0
    ligand_connections_list = []
    ligand_counts = np.array([1])
    # filter for most stringent n_references threshold
    while np.all(ligand_counts != 0): # while extracted network still has full paths for every ligand to atleast one TF
        print('Iteration: {}'.format(counter))
        sn_ppis = extract_network(all_ppis.copy(), curation_effort_thresh = curation_effort_thresh, 
                                  n_references_thresh = n_references_thresh,
                                  resources = resources, 
                                  drop_self = drop_self, 
                                  source_label = source_label, 
                                  target_label = target_label,
                                  verbose = False)

        sn_ppis, ligand_connections = create_connected_network(sn_ppis = sn_ppis, 
                                                               ligand_labels = input_labels, 
                                                               tf_labels = output_labels, 
                                                               source_label = source_label, 
                                                               target_label = target_label,
                                                               path_finder = path_finder)
        print('Number of interactions: {}'.format(sn_ppis.shape[0]))
        ligand_counts = np.array([len(v) for v in ligand_connections.values()])

        all_nodes_ = sorted(set(sn_ppis[source_label].tolist() + sn_ppis[target_label].tolist()))
        if counter == 0 and np.any(ligand_counts == 0):
            warnings.warn('There are disconnected input_labels in the starting signaling network with default thresholds')

        if threshold_on == 'n_references':
            n_references_thresh += 1
        elif threshold_on == 'curation_effort':
            curation_effort_thresh += 1

        ppi_list.append(sn_ppis)
        ligand_connections_list.append(ligand_connections)

        counter += 1

    if threshold_on == 'n_references':
        n_references_thresh -= 2
    elif threshold_on == 'curation_effort':
        curation_effort_thresh -= 2

    if len(ppi_list) > 1:
        sn_ppis = ppi_list[-2]
        ligand_connections = ligand_connections_list[-2]
    else: # situation in which the first iteration already filtered out many nodes
        sn_ppis = ppi_list[0]
        ligand_connections = ligand_connections_list[0]

    return sn_ppis, ligand_connections, n_references_thresh, curation_effort_thresh

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