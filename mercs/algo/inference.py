import numpy as np

from mercs.utils.inference_tools import (
    dummy_array,
    pad_proba,
    select_nominal,
    select_numeric,
)

INPUTS = "inputs"
COMPUTE = "compute"


# Main algorithm
def inference_algorithm(g, m_list, i_list, c_list, data, nominal_ids):
    """Add inference information to graph g
    The information is added to the graph passed as parameter, no new object is returned

    reminder: node[1] = node attribute id

    Arguments:
        g {[type]} -- graph
        m_list {[type]} -- models
        i_list {[type]} -- imputation nodes
        c_list {[type]} -- composition nodes
        data {[type]} -- test data to predict
        nominal_ids {[type]} -- identifiers of the nominal attributes
    
    Raises:
        ValueError: raised when a node type cannot be recognized

    """

    # Helper functions to check node type
    def _data_node(kind): return kind == "D"
    def _model_node(kind): return kind == "M"
    def _imputation_node(kind): return kind == "I"
    def _composite_node(kind): return kind == "C"

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


# Specific nodes:
# 1. input data
# 2. imputation
# 3. numeric data
# 4. nominal data
# 5. model
def input_data_node(g, node, g_desc_ids):
    def f(rel_idx):
        f1 = select_numeric(rel_idx)
        return f1(g.data)

    g.nodes[node][INPUTS] = g_desc_ids.index(node[1])
    g.nodes[node][COMPUTE] = f


def imputation_node(g, node, i_list, nb_rows):
    # Build function
    def f(n):
        return i_list[node[1]].transform(dummy_array(n)).ravel()

    g.nodes[node][INPUTS] = nb_rows
    g.nodes[node][COMPUTE] = f


def numeric_data_node(g, node, m_list, c_list):
    node_parents = _get_parents(g, m_list, c_list, node)

    def f(parents):
        collector = _numeric_inputs(g, parents)
        return np.mean(collector, axis=0)

    g.nodes[node][INPUTS] = node_parents
    g.nodes[node][COMPUTE] = f


def nominal_data_node(g, node, m_list, c_list):
    node_parents = _get_parents(g, m_list, c_list, node, nominal=True)
    classes = np.unique(np.hstack([c for _, c, _, _ in node_parents]))

    def vote(X):
        max_x = np.argmax(X, axis=1)
        return classes.take(max_x, axis=0)

    def F(parents):
        collector = _nominal_inputs(g, parents, classes)
        return np.sum(collector, axis=0)

    def F2(parents):
        return vote(F(parents))

    g.nodes[node]["classes"] = classes
    g.nodes[node][INPUTS] = node_parents
    g.nodes[node]["compute_proba"] = F
    g.nodes[node][COMPUTE] = F2


def model_node(g, node, m_list):
    model_parents = _model_parents(g, node)

    def f(parents):
        X = _model_inputs(g, parents)
        return m_list[node[1]].predict(X)

    g.nodes[node][INPUTS] = model_parents
    g.nodes[node][COMPUTE] = f

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
    compute_str = COMPUTE

    if proba:
        result_str += "_proba"
        compute_str += "_proba"

    r = g.nodes[node].get(result_str, None)
    if r is None:
        i = g.nodes[node].get(INPUTS)
        f = g.nodes[node].get(compute_str)
        g.nodes[node][result_str] = f(i)
        return g.nodes[node][result_str]
    else:
        return r


def _nominal_inputs(g, parents, classes):
    # Returns the 'nominal' inputs of the parent nodes
    collector = []
    for rel_idx, parent_classes, n, model_type in parents:
        if len(parent_classes) == len(classes):
            select_nom = select_nominal(rel_idx)
            X = compute(g, n, proba=True)
            selected = select_nom(X, model_type)
            collector.append(selected)
        else:
            prob = pad_proba(parent_classes, classes)
            select_nom = select_nominal(rel_idx)
            X = compute(g, n, proba=True)
            selected = prob(select_nom(X, model_type))
            collector.append(selected)

    return collector


def _numeric_inputs(g, parents):
    # Returns the 'numeric' inputs of the parent nodes
    collector = []
    for rel_idx, n, model_type in parents:
        select_num = select_numeric(rel_idx)
        X = compute(g, n)
        collector.append(select_num(X))
    return collector


def _model_inputs(g, parents):
    # Returns the 'model' inputs of the parent nodes
    collector = [compute(g, n) for n in parents]
    collector = np.stack(collector, axis=1)
    return collector


def _get_parents(g, m_list, c_list, node, nominal=False):
    # Returns the 'nominal' parents of a node
    parents = []
    for kind, predecessor_idx in g.predecessors(node):
        rel_idx = _rel_idx(predecessor_idx, node[1], kind, m_list, c_list)
        model_type = m_list[predecessor_idx].out_kind
        if nominal:
            classes = _classes(
                predecessor_idx,
                rel_idx,
                kind,
                m_list,
                c_list
            )
            parents.append((rel_idx, classes, (kind, predecessor_idx), model_type))
        else:
            parents.append((rel_idx, (kind, predecessor_idx), model_type))

    return parents


def _model_parents(g, node):
    # Returns the 'model' parents of a node
    idxs = {predecessor_idx: (m, predecessor_idx) for m, predecessor_idx in g.predecessors(node)}

    parents = [n for kind, n in sorted(idxs.items())]

    return parents


def _rel_idx(predecessor_idx, node_idx, kind, m_list, c_list):
    # Calculates the relative id of a node with respect to its predecessor, based on node kind
    if kind == "M":
        return m_list[predecessor_idx].targ_ids.index(node_idx)
    elif kind == "C":
        return c_list[predecessor_idx].targ_ids.index(node_idx)
    else:
        return 0


def _classes(predecessor_idx, rel_idx, kind, m_list, c_list):
    # Returns the classes of a model, based on node kind
    if kind == "M":
        return m_list[predecessor_idx].classes_[rel_idx]
    elif kind == "C":
        return c_list[predecessor_idx].classes_[rel_idx]
