import numpy as np

from ..utils.inference_tools import (
    dummy_array,
    pad_proba,
    select_nominal,
    select_numeric,
)


# Main algorithm
def inference_algorithm(g, m_list, i_list, c_list, data, nominal_ids):
    """Add inference information to graph g
    The information is added to the graph passed as parameter, no need object is returned
    
    Arguments:
        g {[type]} -- graph
        m_list {[type]} -- models
        i_list {[type]} -- imputation nodes
        c_list {[type]} -- composite nodes
        data {[type]} -- test data to predict
        nominal_ids {[type]} -- identifiers of the nominal attributes
    
    Raises:
        ValueError: raised when a node type cannot be recognized

    """

    # Helper functions to check node type
    def _data_node(k): return k == "D"
    def _model_node(k): return k == "M"
    def _imputation_node(k): return k == "I"
    def _composite_node(k): return k == "C"

    nb_rows, _ = data.shape

    g_descriptive_ids = list(g.desc_ids)

    if data is not None:
        g.data = data[:, g_descriptive_ids]
    else:
        g.data = None

    for node in g.nodes():
        if _data_node(node[0]):
            in_degree = g.in_degree(node)
            if in_degree == 0:
                input_data_node(g, node, g_descriptive_ids)
            elif in_degree > 0:
                if node[1] in nominal_ids:
                    nominal_data_node(g, node, m_list, c_list)
                else:
                    numeric_data_node(g, node, m_list, c_list)
        elif _model_node(node[0]):
            model_node(g, node, m_list)
        elif _imputation_node(node[0]):
            imputation_node(g, node, i_list, nb_rows)
        elif _composite_node(node[0]):
            composite_node(g, node, c_list)
        else:
            raise ValueError("Did not recognize node kind of {}".format(node))


# Specific Nodes
def input_data_node(g, node, g_desc_ids):
    def f(rel_idx):
        f1 = select_numeric(rel_idx)
        return f1(g.data)

    # New
    g.nodes[node]["inputs"] = g_desc_ids.index(node[1])
    g.nodes[node]["compute"] = f


def imputation_node(g, node, i_list, nb_rows):
    # Build function
    def f(n):
        return i_list[node[1]].transform(dummy_array(n)).ravel()

    # New
    g.nodes[node]["inputs"] = nb_rows
    g.nodes[node]["compute"] = f


def numeric_data_node(g, node, m_list, c_list):
    node_parents = _numeric_parents(g, m_list, c_list, node)

    def f(parents):
        collector = _numeric_inputs(g, parents)
        return np.mean(collector, axis=0)

    g.nodes[node]["inputs"] = node_parents
    g.nodes[node]["compute"] = f


def nominal_data_node(g, node, m_list, c_list):
    node_parents = _nominal_parents(g, m_list, c_list, node)

    classes = np.unique(np.hstack([c for _, c, _ in node_parents]))

    def vote(X):
        return classes.take(np.argmax(X, axis=1), axis=0)

    def F(parents):
        collector = _nominal_inputs(g, parents, classes)
        return np.sum(collector, axis=0)

    def F2(parents):
        return vote(F(parents))

    g.nodes[node]["classes"] = classes
    g.nodes[node]["inputs"] = node_parents
    g.nodes[node]["compute_proba"] = F
    g.nodes[node]["compute"] = F2


def model_node(g, node, m_list):
    model_parents = _model_parents(g, node)

    def f(parents):
        X = _model_inputs(g, parents)
        return m_list[node[1]].predict(X)

    g.nodes[node]["inputs"] = model_parents
    g.nodes[node]["compute"] = f

    if hasattr(m_list[node[1]], "predict_proba"):
        def f2(parents):
            X = _model_inputs(g, parents)
            return m_list[node[1]].predict_proba(X)

        g.nodes[node]["compute_proba"] = f2


def composite_node(g, node, c_list):
    return model_node(g, node, c_list)


# Helper functions
def compute(g, node, proba=False):
    result_str = "result"
    compute_str = "compute"

    if proba:
        result_str += "_proba"
        compute_str += "_proba"

    r = g.nodes[node].get(result_str, None)
    if r is None:
        i = g.nodes[node].get("inputs")
        f = g.nodes[node].get(compute_str)
        g.nodes[node][result_str] = f(i)
        return g.nodes[node][result_str]
    else:
        return r


def _nominal_inputs(g, parents, classes):
    collector = [
        select_nominal(idx)(compute(g, n, proba=True))
        if len(c) == len(classes)
        else pad_proba(c, classes)(select_nominal(idx)(compute(g, n, proba=True)))
        for idx, c, n in parents
    ]
    return collector


def _numeric_inputs(g, parents):
    collector = [select_numeric(idx)(compute(g, n)) for idx, n in parents]
    return collector


def _model_inputs(g, parents):
    collector = [compute(g, n) for n in parents]
    collector = np.stack(collector, axis=1)
    return collector


def _numeric_parents(g, m_list, c_list, node):
    parents = [
        (_rel_idx(p_idx, node[1], k, m_list, c_list), (k, p_idx))
        for k, p_idx in g.predecessors(node)
    ]

    return parents


def _nominal_parents(g, m_list, c_list, node):
    parents = [
        (
            _rel_idx(p_idx, node[1], k, m_list, c_list),
            _classes(
                p_idx, _rel_idx(p_idx, node[1], k, m_list, c_list), k, m_list, c_list
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


def _rel_idx(p_idx, n_idx, k, m_list, c_list):
    if k == "M":
        return m_list[p_idx].targ_ids.index(n_idx)
    elif k == "C":
        return c_list[p_idx].targ_ids.index(n_idx)
    else:
        return 0


def _classes(p_idx, r_idx, k, m_list, c_list):
    # FIXME: breaks for mixed trees. list index out of range
    if k == "M":
        return m_list[p_idx].classes_[r_idx]
    elif k == "C":
        return c_list[p_idx].classes_[r_idx]
