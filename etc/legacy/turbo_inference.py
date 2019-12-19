import numpy as np
from dask import delayed
from functools import partial
from graph_tool.topology import topological_sort

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

    # Lambdas
    data_node = lambda k, n: k == "D"
    model_node = lambda k, n: k == "M"
    imputation_node = lambda k, n: k == "I"

    # Vertex Props
    dask = g.new_vertex_property("object")
    dask_proba = g.new_vertex_property("object")
    g_classes = g.new_vertex_property("vector<long>")

    nb_rows, _ = data.shape

    nodes = ((v_idx, g.vp.names[v_idx]) for v_idx in topological_sort(g))

    g_desc_ids = list(g.desc_ids)
    data = delayed(data[:, g_desc_ids])

    for v_idx, n in nodes:
        vertex = g.vertex(v_idx)

        if data_node(*n):
            if vertex.in_degree() == 0:
                dask_input_data_node(dask, n, v_idx, g_desc_ids, data)
            elif vertex.in_degree() == 1:
                dask_single_data_node(dask, g, n, v_idx, m_list)
            elif vertex.in_degree() > 1:
                if n[1] in nominal_ids:
                    dask_nominal_data_node(g, n, m_list)
                else:
                    dask_numeric_data_node(dask, g, n, v_idx, m_list)
        elif model_node(*n):
            dask_model_node(dask, dask_proba, g, n, v_idx, m_list)
        elif imputation_node(*n):
            dask_imputation_node(dask, n, v_idx, i_list, nb_rows)
        else:
            raise ValueError("Did not recognize node kind of {}".format(n))

    return dask


# Nodes - Imputation
def dask_imputation_node(dask, node, v_idx, i_list, nb_rows):

    f1 = _dummy_array
    f2 = i_list[node[1]].transform
    f3 = np.ravel
    f = o(f3, o(f2, f1))

    dask[v_idx] = delayed(f)(nb_rows)
    return


# Nodes - Data
def dask_input_data_node(dask, node, v_idx, g_desc_ids, data):
    dask[v_idx] = delayed(_select_numeric(g_desc_ids.index(node[1])))(data)
    return


def dask_single_data_node(dask, g, node, v_idx, m_list):
    # Single output to recover from model, I do not have to merge or anything.
    idx, parent_functions = _get_parents_of_numeric_data_node(
        dask, g, m_list, v_idx, node
    )[0]

    dask[v_idx] = delayed(_select_numeric(idx))(parent_functions)
    return


def dask_numeric_data_node(dask, g, node, v_idx, m_list):

    idx_fnc = _get_parents_of_numeric_data_node(dask, g, m_list, v_idx, node)

    parent_functions = [delayed(_select_numeric(idx))(fnc) for idx, fnc in idx_fnc]
    dask[v_idx] = delayed(partial(np.mean, axis=0))(parent_functions)
    return


def dask_nominal_data_node(dask, dask_proba, g_classes, g, node, v_idx, m_list):
    idx_cls_fnc = _get_parents_of_nominal_data_node(dask, g, m_list, v_idx, node)
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
    dask_proba[v_idx] = f3
    g_classes[v_idx] = classes

    # Vote
    def vote(X):
        return classes.take(np.argmax(X, axis=1), axis=0)

    dask[v_idx] = delayed(vote)(f3)
    return


# Nodes - Model
def dask_model_node(dask, dask_proba, g, node, v_idx, m_list):
    # Collect input data
    parent_functions = _get_parents_of_model_node(g, dask, v_idx)
    collector = delayed(np.stack)(parent_functions, axis=1)

    # Convert function
    dask[v_idx] = delayed(m_list[node[1]].predict)(collector)

    if hasattr(m_list[node[1]], "predict_proba"):
        dask_proba[v_idx] = delayed(m_list[node[1]].predict_proba)(collector)

    return


# Finding Parents
def _get_parents_of_numeric_data_node(dask, g, m_list, v_idx, node):
    rel_idx = lambda p_idx, n_idx: m_list[p_idx].targ_ids.index(n_idx)

    parents = ((g.vp.names[idx], idx) for idx in g.get_in_neighbours(v_idx))

    idx_fnc = [
        (rel_idx(p_idx, node[1]) if k == "M" else 0, dask[v_idx])
        for (k, p_idx), v_idx in parents
    ]

    return idx_fnc


def _get_parents_of_nominal_data_node(dask, g, m_list, v_idx, node):
    rel_idx = lambda p_idx, n_idx: m_list[p_idx].targ_ids.index(n_idx)
    classes = lambda p_idx, r_idx: m_list[p_idx].classes_[r_idx]

    parents = ((g.vp.names[idx], idx) for idx in g.get_in_neighbours(v_idx))

    idx_fnc = (
        (rel_idx(p_idx, node[1]) if k == "M" else 0, p_idx, dask[v_idx])
        for (k, p_idx), v_idx in parents
    )

    idx_cls_fnc = [(r_idx, classes(p_idx, r_idx), f) for r_idx, p_idx, f in idx_cls_fnc]

    return idx_cls_fnc


def _get_parents_of_model_node(g, dask, v_idx):
    parent_functions = {
        g.vp.names[idx][1]: dask[idx] for idx in g.get_in_neighbours(v_idx)
    }
    parent_functions = [v for k, v in sorted(parent_functions.items())]
    return parent_functions
