from functools import partial

import networkx as nx
import numpy as np
from dask import delayed

from ..composition import o
from ..utils.inference_tools import (
    _dummy_array,
    _map_classes,
    _pad_proba,
    _select_nominal,
    _select_numeric,
)


# Main algorithm
def inference_algorithm(g, m_list, i_list, data, nominal_ids):

    data_node = lambda k, n: k == "D"
    model_node = lambda k, n: k == "M"
    imputation_node = lambda k, n: k == "I"

    nodes = list(nx.topological_sort(g))
    nb_rows, _ = data.shape

    g_desc_ids = list(g.desc_ids)
    data = data[:, g_desc_ids]

    for n in nodes:
        if data_node(*n):
            in_degree = g.in_degree(n)
            if in_degree == 0:
                dask_input_data_node(g, n, g_desc_ids, data)
            elif in_degree == 1:
                dask_single_data_node(g, n, m_list)
            elif in_degree > 1:
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

    f = _select_numeric(g_desc_ids.index(node[1]))

    g.node[node]["dask"] = delayed(f)(data)

    # New
    g.node[node]["inputs"] = data
    g.node[node]["compute"] = f

    return


def dask_imputation_node(g, node, i_list, nb_rows):

    # Build function
    f1 = _dummy_array
    f2 = i_list[node[1]].transform
    f3 = np.ravel
    f = o(f3, o(f2, f1))

    g.node[node]["dask"] = delayed(f)(nb_rows)

    # New
    g.node[node]["inputs"] = nb_rows
    g.node[node]["compute"] = f
    return


def dask_single_data_node(g, node, m_list):

    # Build function
    idx, parent_function = _dask_get_parents_of_numeric_data_node(g, m_list, node)[
        0
    ]  # Single input
    f = _select_numeric(idx)
    g.node[node]["dask"] = delayed(f)(parent_function)

    # New
    parents = _numeric_parents(g, m_list, node)

    def f(parents):
        collector = _numeric_inputs(parents)
        return collector.pop()

    g.node[node]["inputs"] = parents
    g.node[node]["compute"] = f

    return


def dask_numeric_data_node(g, node, m_list):

    idx_fnc = _dask_get_parents_of_numeric_data_node(g, m_list, node)

    parent_functions = [delayed(_select_numeric(idx))(fnc) for idx, fnc in idx_fnc]

    f1 = partial(np.mean, axis=0)

    g.node[node]["dask"] = delayed(f1)(parent_functions)

    # New
    parents = _numeric_parents(g, m_list, node)

    def f(parents):
        collector = _numeric_inputs(parents)
        return f1(collector)

    g.node[node]["inputs"] = parents
    g.node[node]["compute"] = f
    return


def dask_nominal_data_node(g, node, m_list):
    idx_cls_fnc = _dask_get_parents_of_nominal_data_node(g, m_list, node)
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

    # New
    """
    parents = (
        _select_nominal(idx)(fnc)
        if len(c) == len(classes)
        else _pad_proba(c, classes)(_select_nominal(idx)(fnc))
        for idx, c, fnc in _nominal_parents(g, m_list, node)
    )
    
    def F(parents):
        collector = np.sum(list(parents), axis=0)
        return collector

    def F2(parents):
        try:
            collector = np.sum(list(parents), axis=0)
            return vote(collector)
        except:
            collector = np.sum(list(parents), axis=0)
            return collector
    """

    parents = _nominal_parents(g, m_list, node)
    def F(parents):
        collector = _nominal_inputs(g, parents, classes)
        return np.sum(collector, axis=0)

    def F2(parents):
        return vote(F(parents))

    g.node[node]["inputs"] = parents
    g.node[node]["compute_proba"] = F
    g.node[node]["compute"] = F2

    return


def dask_model_node(g, node, m_list):
    # Collect input data
    parent_functions = _dask_get_parents_of_model_node(g, node)

    collector = delayed(np.stack, pure=True)(parent_functions, axis=1)

    # Convert function
    g.node[node]["dask"] = delayed(m_list[node[1]].predict)(collector)

    if hasattr(m_list[node[1]], "predict_proba"):
        g.node[node]["dask_proba"] = delayed(m_list[node[1]].predict_proba)(collector)

    # New
    parents = _model_parents(g, node)
    
    def f(parents):
        X = _model_inputs(g, parents)
        return m_list[node[1]].predict(X)

    g.node[node]["inputs"] = parents
    g.node[node]["compute"] = f

    if hasattr(m_list[node[1]], "predict_proba"):
        def f2(parents):
            X = _model_inputs(g, parents)
            return m_list[node[1]].predict_proba(X)

        g.node[node]["compute_proba"] = f2

    return


# Helpers - Function
def _nominal_inputs(g, parents, classes):
    collector = [
            _select_nominal(idx)(compute(g, n, proba=True))
            if len(c) == len(classes)
            else _pad_proba(c, classes)(_select_nominal(idx)(compute(g, n, proba=True)))
            for idx, c, n in parents
        ]
    return collector

def _numeric_inputs(g, parents):
    collector = [_select_numeric(idx)(compute(g, n)) for idx, n in parents]
    return collector

def _model_inputs(g, parents):
    collector = [compute(g, n) for n in parents]
    collector = np.stack(collector, axis=1)
    return collector

def compute(g, node, proba=False):

    result = "result"
    compute = "compute"

    if proba:
        result += "_proba"
        compute += "_proba"

    r = g.node[node].get(result, None)
    if r is None:
        i = g.node[node].get("inputs")
        f = g.node[node].get(compute)
        g.node[node][result] = f(i)
        return g.node[node][result]
    else:
        return r


def _numeric_parents(g, m_list, node):
    rel_idx = lambda p_idx, n_idx: m_list[p_idx].targ_ids.index(n_idx)

    # parents = (
    #    (rel_idx(p_idx, node[1]) if m == "M" else 0, compute(g, (m, p_idx)))
    #    for m, p_idx in g.predecessors(node)
    #

    parents = [
        (rel_idx(p_idx, node[1]) if m == "M" else 0, (m, p_idx))
        for m, p_idx in g.predecessors(node)
    ]

    return parents


def _nominal_parents(g, m_list, node):
    rel_idx = lambda p_idx, n_idx: m_list[p_idx].targ_ids.index(n_idx)
    classes = lambda p_idx, r_idx: m_list[p_idx].classes_[r_idx]

    # parents = (
    #    (
    #        rel_idx(p_idx, node[1]) if m == "M" else 0,
    #        classes(p_idx, rel_idx(p_idx, node[1]) if m == "M" else 0),
    #        compute(g, (m, p_idx), proba=True),
    #    )
    #    for m, p_idx in g.predecessors(node)
    # )

    parents = [
        (
            rel_idx(p_idx, node[1]) if m == "M" else 0,
            classes(p_idx, rel_idx(p_idx, node[1]) if m == "M" else 0),
            (m, p_idx),
        )
        for m, p_idx in g.predecessors(node)
    ]

    return parents


def _model_parents(g, node):
    idxs = {p_idx: (m, p_idx) for m, p_idx in g.predecessors(node)}

    # parents = (compute(g, n) for k, n in sorted(idxs.items()))

    parents = [n for k, n in sorted(idxs.items())]

    return parents


# Helpers - Dask
def _dask_get_parents_of_model_node(g, node):
    parent_functions = {a: g.nodes[(m, a)]["dask"] for m, a in g.predecessors(node)}
    parent_functions = [v for k, v in sorted(parent_functions.items())]
    return parent_functions


def _dask_get_parents_of_numeric_data_node(g, m_list, node):
    rel_idx = lambda p_idx, n_idx: m_list[p_idx].targ_ids.index(n_idx)

    idx_fnc = [
        (rel_idx(p_idx, node[1]) if m == "M" else 0, g.node[(m, p_idx)]["dask"])
        for m, p_idx in g.predecessors(node)
    ]

    return idx_fnc


def _dask_get_parents_of_nominal_data_node(g, m_list, node):
    rel_idx = lambda p_idx, n_idx: m_list[p_idx].targ_ids.index(n_idx)
    classes = lambda p_idx, r_idx: m_list[p_idx].classes_[r_idx]

    idx_fnc = (
        (
            rel_idx(p_idx, node[1]) if m == "M" else 0,
            p_idx,
            g.node[(m, p_idx)]["dask_proba"],
        )
        for m, p_idx in g.predecessors(node)
    )
    idx_cls_fnc = [(r_idx, classes(p_idx, r_idx), f) for r_idx, p_idx, f in idx_fnc]

    return idx_cls_fnc
