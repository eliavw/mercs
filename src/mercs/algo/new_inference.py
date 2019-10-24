import networkx as nx
import numpy as np
from dask import delayed
from functools import partial

from ..composition import o


# Main algorithm
def inference_algorithm(g, m_list, i_list, data, nominal_ids):

    data_node = lambda k,n: k=='D'
    model_node = lambda k,n: k=='M'
    imputation_node = lambda k,n: k=='I'

    nodes = list(nx.topological_sort(g))
    nb_rows, _ = data.shape

    g_desc_ids = list(g.desc_ids)
    data = delayed(data[:, g_desc_ids])

    for n in nodes:
        if data_node(*n):
            if g.in_degree(n) == 0:
                dask_input_data_node(g, n, g_desc_ids, data)
            elif g.in_degree(n) == 1:
                dask_single_data_node(g, n, m_list)
            elif g.in_degree(n) > 1:
                if n[1] in nominal_ids:
                    dask_nominal_data_node(g, n, m_list)
                else:
                    dask_numeric_data_node(g, n, m_list)
        elif model_node(*n):
            dask_model_node(g, n, m_list)
        elif imputation_node(*n):
            dask_imputation_node(g, n, i_list, nb_rows)
        else:
            raise ValueError("Did not recognize node kind of {}".format(n))

    return


# Specific Nodes
def dask_input_data_node(g, node, g_desc_ids, data):
    g.node[node]["dask"] = delayed(_select_numeric(g_desc_ids.index(node[1])))(data)
    return


def dask_model_node(g, node, m_list):
    # Collect input data
    parent_functions = _get_parents_of_model_node(g, node)
    collector = delayed(np.stack)(parent_functions, axis=1)

    # Convert function
    g.node[node]["dask"] = delayed(m_list[node[1]].predict)(collector)

    if hasattr(m_list[node[1]], "predict_proba"):
        g.node[node]["dask_proba"] = delayed(node["predict_proba"])(collector)

    return


def dask_imputation_node(g, node, i_list, nb_rows):

    f1 = _dummy_array
    f2 = i_list[node[1]].transform
    f3 = np.ravel
    f = o(f3, o(f2, f1))

    g.node[node]["dask"] = delayed(f)(nb_rows)
    return


def dask_single_data_node(g, node, m_list):
    # Single output to recover from model, I do not have to merge or anything.
    idx, parent_functions = _get_parents_of_numeric_data_node(g, m_list, node)[0]
    g.node[node]["dask"] = delayed(_select_numeric(idx))(parent_functions)
    return


def dask_nominal_data_node(g, node, m_list):
    idx_cls_fnc = _get_parents_of_nominal_data_node(g, m_list, node)
    classes = np.unique(np.hstack([c for _, c, _ in idx_cls_fnc]))

    # Reduce
    parent_functions = []
    for idx, c, fnc in idx_cls_fnc:
        f1 = delayed(_select_nominal(idx))(fnc)
        if len(c) < len(classes):
            f2 = delayed(_pad_proba(c, classes))(f1)
            parent_functions.append(f2)
        else:
            parent_functions.append(f1)

    f3 = delayed(partial(np.sum, axis=0))(parent_functions)
    g.node[node]["dask_proba"] = f3
    g.node[node]["classes"] = classes
    
    # Vote
    def vote(X):
        return classes.take(np.argmax(X, axis=1), axis=0)
    
    g.node[node]["dask"] = delayed(vote)(f3)
    return


def dask_numeric_data_node(g, node, m_list):

    idx_fnc = _get_parents_of_numeric_data_node(g, m_list, node)

    parent_functions = [delayed(_select_numeric(idx))(fnc) for idx, fnc in idx_fnc]
    g.node[node]["dask"] = delayed(partial(np.mean, axis=0))(parent_functions)
    return


# Helpers - Graph
def _get_parents_of_model_node(g, node):
    parent_functions = {a: g.nodes[(m, a)]["dask"] for m, a in g.predecessors(node)}
    parent_functions = [v for k, v in sorted(parent_functions.items())]
    return parent_functions


def _get_parents_of_numeric_data_node(g, m_list, node):
    rel_idx = lambda p_idx, n_idx: m_list[p_idx].targ_ids.index(n_idx) 
    

    parents = ((m, p_idx) for m, p_idx in g.predecessors(node))

    idx_fnc = [
        (rel_idx(p_idx, node[1]) if m=='M' else 0, g.node[(m, p_idx)]["dask"]) for m, p_idx in parents
    ]

    return idx_fnc


def _get_parents_of_nominal_data_node(g, m_list, node):
    rel_idx = lambda p_idx, n_idx: m_list[p_idx].targ_ids.index(n_idx)
    classes = lambda p_idx, r_idx: m_list[p_idx].classes_[r_idx]

    parents = ((m, p_idx) for m, p_idx in g.predecessors(node))

    idx_fnc = (
        (rel_idx(p_idx, node[1]) if m=='M' else 0, p_idx, g.node[(m, p_idx)]["dask"])
        for m, p_idx in parents
    )
    idx_cls_fnc = [(r_idx, classes(p_idx, r_idx), f) for r_idx, p_idx, f in idx_cls_fnc]

    return idx_cls_fnc


# Helpers - Data Handling
def _dummy_array(nb_rows):
    a = np.empty((nb_rows, 1))
    a.fill(np.nan)
    return a


def _pad_proba(classes, all_classes):
    idx = _map_classes(classes, all_classes)

    def pad(X):
        R = np.zeros((X.shape[0], len(all_classes)))
        R[:, idx] = X
        return R

    return pad


def _map_classes(classes, all_classes):
    sorted_idx = np.argsort(all_classes)
    matches = np.searchsorted(all_classes[sorted_idx], classes)
    return sorted_idx[matches]


def _select_numeric(idx):
    def select(X):
        if X.ndim == 2:
            return X.take(idx, axis=1)
        else:
            return X

    return select


def _select_nominal(idx):
    def select(X):
        if isinstance(X, list):
            return X[idx]
        elif isinstance(X, np.ndarray):
            return X

    return select
