from functools import partial

import networkx as nx
import numpy as np

from ..composition import o
from ..utils.inference_tools import (
    _dummy_array,
    _map_classes,
    _pad_proba,
    _select_nominal,
    _select_numeric,
)


# Main algorithm
def inference_algorithm(g, m_list, i_list, c_list, data, nominal_ids):
    """Add inference information to graph g
    
    Arguments:
        g {[type]} -- [description]
        m_list {[type]} -- [description]
        i_list {[type]} -- [description]
        c_list {[type]} -- [description]
        data {[type]} -- [description]
        nominal_ids {[type]} -- [description]
    
    Raises:
        ValueError: [description]
    
    Returns:
        [type] -- [description]
    """

    _data_node = lambda k, n: k == "D"
    _model_node = lambda k, n: k == "M"
    _imputation_node = lambda k, n: k == "I"
    _composite_node = lambda k, n: k == "C"

    nb_rows, _ = data.shape

    g_desc_ids = list(g.desc_ids)

    if data is not None:
        g.data = data[:, g_desc_ids]
    else:
        g.data = None

    for n in g.nodes():
        if _data_node(*n):
            in_degree = g.in_degree(n)
            if in_degree == 0:
                input_data_node(g, n, g_desc_ids)
            elif in_degree > 0:
                if n[1] in nominal_ids:
                    nominal_data_node(g, n, m_list, c_list)
                else:
                    numeric_data_node(g, n, m_list, c_list)
        elif _model_node(*n):
            model_node(g, n, m_list)
        elif _imputation_node(*n):
            imputation_node(g, n, i_list, nb_rows)
        elif _composite_node(*n):
            composite_node(g, n, c_list)
        else:
            raise ValueError("Did not recognize node kind of {}".format(n))

    return


# Specific Nodes
def input_data_node(g, node, g_desc_ids):
    def f(rel_idx):
        f1 = _select_numeric(rel_idx)
        return f1(g.data)

    # New
    g.nodes[node]["inputs"] = g_desc_ids.index(node[1])
    g.nodes[node]["compute"] = f

    return


def imputation_node(g, node, i_list, nb_rows):

    # Build function
    def f(n):
        return i_list[node[1]].transform(_dummy_array(n)).ravel()

    # New
    g.nodes[node]["inputs"] = nb_rows
    g.nodes[node]["compute"] = f
    return

"""
def single_data_node(g, node, m_list, c_list):
    # New
    parents = _numeric_parents(g, m_list, c_list, node)

    def f(parents):
        collector = _numeric_inputs(g, parents)
        return collector.pop()

    g.nodes[node]["inputs"] = parents
    g.nodes[node]["compute"] = f

    return
"""

def numeric_data_node(g, node, m_list, c_list):
    parents = _numeric_parents(g, m_list, c_list, node)

    def f(parents):
        collector = _numeric_inputs(g, parents)
        return np.mean(collector, axis=0)

    g.nodes[node]["inputs"] = parents
    g.nodes[node]["compute"] = f
    return


def nominal_data_node(g, node, m_list, c_list):
    # New
    parents = _nominal_parents(g, m_list, c_list, node)

    classes = np.unique(np.hstack([c for _, c, _ in parents]))

    def vote(X):
        return classes.take(np.argmax(X, axis=1), axis=0)

    def F(parents):
        collector = _nominal_inputs(g, parents, classes)
        return np.sum(collector, axis=0)

    def F2(parents):
        return vote(F(parents))

    g.nodes[node]["classes"] = classes
    g.nodes[node]["inputs"] = parents
    g.nodes[node]["compute_proba"] = F
    g.nodes[node]["compute"] = F2

    return


def model_node(g, node, m_list):

    # New
    parents = _model_parents(g, node)

    def f(parents):
        X = _model_inputs(g, parents)
        return m_list[node[1]].predict(X)

    g.nodes[node]["inputs"] = parents
    g.nodes[node]["compute"] = f

    if hasattr(m_list[node[1]], "predict_proba"):

        def f2(parents):
            X = _model_inputs(g, parents)
            return m_list[node[1]].predict_proba(X)

        g.nodes[node]["compute_proba"] = f2

    return


def composite_node(g, node, c_list):
    return model_node(g, node, c_list)


# Helpers - Function
def compute(g, node, proba=False):

    result = "result"
    compute = "compute"

    if proba:
        result += "_proba"
        compute += "_proba"

    r = g.nodes[node].get(result, None)
    if r is None:
        i = g.nodes[node].get("inputs")
        f = g.nodes[node].get(compute)
        g.nodes[node][result] = f(i)
        return g.nodes[node][result]
    else:
        return r


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


def _numeric_parents(g, m_list, c_list, node):

    parents = [
        (rel_idx(p_idx, node[1], k, m_list, c_list), (k, p_idx))
        for k, p_idx in g.predecessors(node)
    ]

    return parents


def _nominal_parents(g, m_list, c_list, node):
    parents = [
        (
            rel_idx(p_idx, node[1], k, m_list, c_list),
            classes(
                p_idx, rel_idx(p_idx, node[1], k, m_list, c_list), k, m_list, c_list
            ),
            (k, p_idx),
        )
        for k, p_idx in g.predecessors(node)
    ]

    return parents


def _model_parents(g, node):
    idxs = {p_idx: (m, p_idx) for m, p_idx in g.predecessors(node)}

    parents = [n for k, n in sorted(idxs.items())]

    return parents


def rel_idx(p_idx, n_idx, k, m_list, c_list):
    if k == "M":
        return m_list[p_idx].targ_ids.index(n_idx)
    elif k == "C":
        return c_list[p_idx].targ_ids.index(n_idx)
    else:
        return 0


def classes(p_idx, r_idx, k, m_list, c_list):
    if k == "M":
        return m_list[p_idx].classes_[r_idx]
    elif k == "C":
        return c_list[p_idx].classes_[r_idx]

