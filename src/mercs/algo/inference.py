import networkx as nx
import numpy as np

from functools import partial, reduce
from dask import delayed

from ..graph.network import get_ids, node_label
from ..composition import o, x
from ..utils import debug_print

VERBOSITY = 0


def base_inference_algorithm(g, X=None):

    # Convert the graph to its functions
    sorted_nodes = list(nx.topological_sort(g))

    msg = """
    sorted_nodes:    {}
    """.format(
        sorted_nodes
    )
    debug_print(msg, level=1, V=VERBOSITY)
    functions = {}
    q_desc_ids = list(get_ids(g, kind="desc"))

    for node_name in sorted_nodes:
        node = g.nodes(data=True)[node_name]

        if node.get("kind", None) == "data":
            if len(nx.ancestors(g, node_name)) == 0:
                functions[node_name] = _select_numeric(q_desc_ids.index(node["idx"]))
            else:
                # Select the relevant output
                previous_node = [t[0] for t in g.in_edges(node_name)][0]
                previous_t_idx = g.nodes[previous_node]["tgt"]
                relevant_idx = previous_t_idx.index(node["idx"])

                functions[node_name] = o(
                    _select_numeric(relevant_idx), functions[previous_node]
                )

        elif node.get("kind", None) == "imputation":
            functions[node_name] = node["function"]

        elif node.get("kind", None) == "model":
            previous_nodes = [t[0] for t in g.in_edges(node_name)]
            inputs = {g.nodes[n]["tgt"][0]: functions[n] for n in previous_nodes}
            inputs = [
                inputs[k] for k in sorted(inputs)
            ]  # We need to sort to get the inputs in the correct order.

            inputs = o(np.transpose, x(*inputs, return_type=np.array))

            f = node["function"]
            functions[node_name] = o(f, inputs)

        elif node.get("kind", None) == "prob":
            # Select the relevant output
            prob_idx = node["idx"]
            prob_classes = node["classes"]

            previous_nodes = [t[0] for t in g.in_edges(node_name)]
            previous_classes = [g.edges[t]["classes"] for t in g.in_edges(node_name)]
            previous_t_idx = [g.nodes[n]["tgt"] for n in previous_nodes]

            inputs = [
                (functions[n], t, c)
                for n, t, c in zip(previous_nodes, previous_t_idx, previous_classes)
            ]

            for idx, (f1, t, c) in enumerate(inputs):
                f2 = o(_select_nominal(t.index(prob_idx)), f1)

                if len(c) < len(prob_classes):
                    f2 = o(_pad_proba(c, prob_classes), f2)

                inputs[idx] = f2

            f = partial(np.sum, axis=0)
            functions[node_name] = o(f, x(*inputs, return_type=np.array))

        elif node.get("kind", None) == "vote":
            # Convert probabilistic votes to single prediction
            previous_node = [t[0] for t in g.in_edges(node_name)][0]
            functions[node_name] = o(node["function"], functions[previous_node])

        elif node.get("kind", None) == "merge":
            merge_idx = node["idx"]
            previous_nodes = [t[0] for t in g.in_edges(node_name)]
            previous_t_idx = [g.nodes[n]["tgt"] for n in previous_nodes]

            inputs = [(functions[n], t) for n, t in zip(previous_nodes, previous_t_idx)]

            inputs = [
                o(_select_numeric(t_idx.index(merge_idx)), f) for f, t_idx in inputs
            ]
            inputs = o(np.transpose, x(*inputs, return_type=np.array))

            f = partial(np.mean, axis=1)
            functions[node_name] = o(f, inputs)

    return functions


def dask_inference_algorithm(g, X=None, sorted_nodes=None):
    if sorted_nodes is None:
        sorted_nodes = list(nx.topological_sort(g))
    
    functions = {}

    q_desc_ids = list(get_ids(g, kind="desc"))

    if X is None:
        data = None
    else:
        data = delayed(X[:, q_desc_ids])

    for node_name in sorted_nodes:
        kind = g.nodes[node_name]['kind']
        node = g.nodes[node_name]

        if kind in {'imputation'}:
            actions[kind](node, data)
        elif kind in {'data'}:
            actions[kind](g, node, node_name, data, q_desc_ids)
            functions[node_name] = node["dask"]
        elif kind in {'prob'}:
            actions[kind](g, node, node_name)
            functions[node_name] = node["dask"]
        else:
            actions[kind](g, node, node_name)

    return functions


def dask_imputation_node(node, data):
    node["dask"] = delayed(node["function"])(data)
    return


def dask_data_node(g, node, node_name, data, q_desc_ids):
    n_parents = len(g.in_edges(node_name))

    if n_parents == 0:
        idx = node["idx"]
        node["dask"] = delayed(_select_numeric(q_desc_ids.index(idx)))(data)
    else:
        # Select the relevant output
        parent_relative_idx, parent_function = _get_parents_of_data_node(g, node, node_name)
        node["dask"] = delayed(_select_numeric(parent_relative_idx))(parent_function)
    return


def dask_model_node(g, node, node_name):
    # Collect input data
    parent_functions = _get_parents_of_model_node(g, node, node_name)
    collector = delayed(np.stack)(parent_functions, axis=1)

    # Convert function
    node["dask"] = delayed(node["predict"])(collector)

    if "predict_proba" in node:
        node["dask_proba"] = delayed(node["predict_proba"])(collector)

    return


def dask_prob_node(g, node, node_name):
    # Parent nodes
    parent_nodes = [s for s, t in g.in_edges(node_name)]

    parent_functions = [g.nodes[n]["dask_proba"] for n in parent_nodes]
    parent_targets = [g.nodes[n]["tgt"] for n in parent_nodes]
    parent_classes = [g.edges[e]["classes"] for e in g.in_edges(node_name)]

    inputs = zip(parent_functions, parent_targets, parent_classes)

    # Incorporate extra step(s)
    for idx, (f1, t, c) in enumerate(inputs):
        f2 = delayed(_select_nominal(t.index(node["idx"])))(f1)

        if len(c) < len(node["classes"]):
            f3 = delayed(_pad_proba(c, node["classes"]))(f2)
        else:
            f3 = f2

        # Overwrite parent functions
        parent_functions[idx] = f3

    # Collect everything in one single array
    node["dask"] = delayed(partial(np.sum, axis=0))(parent_functions)
    return


def dask_vote_node(g, node, node_name):
    parent_node = [s for s, t in g.in_edges(node_name)].pop()
    parent_function = g.nodes[parent_node]["dask"]
    classes = node["classes"]

    # Build function
    def vote(X):
        return classes.take(np.argmax(X, axis=1), axis=0)

    node["dask"] = delayed(vote)(parent_function)
    return


def dask_merge_node(g, node, node_name):
    # Parent nodes
    parent_nodes = [s for s, t in g.in_edges(node_name)]

    parent_functions = [g.nodes[n]["dask"] for n in parent_nodes]
    parent_targets = [g.nodes[n]["tgt"] for n in parent_nodes]

    inputs = zip(parent_functions, parent_targets)

    # Incorporate extra step(s)
    for idx, (f1, t) in enumerate(inputs):
        f2 = delayed(_select_numeric(t.index(node["idx"])))(f1)
        parent_functions[idx] = f2

    node["dask"] = delayed(partial(np.mean, axis=0))(parent_functions)
    return


actions = dict(
    data=dask_data_node,
    prob=dask_prob_node,
    model=dask_model_node,
    vote=dask_vote_node,
    merge=dask_merge_node,
    imputation=dask_imputation_node,
)


# Helpers
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


def _get_parents_of_data_node(g, node, node_name):
    # It can only be one parent
    parent_node = g.nodes[[s for s, t in g.in_edges(node_name)].pop()]

    parent_relative_idx = parent_node["tgt"].index(node["idx"])
    parent_function = parent_node["dask"]

    return parent_relative_idx, parent_function


def _get_parents_of_model_node(g, node, node_name):
    parent_nodes = [s for s, t in g.in_edges(node_name)]
    parent_indices = [g.nodes[n]["idx"] for n in parent_nodes]

    parent_functions = {idx: g.nodes[n]["dask"] for idx, n in zip(parent_indices, parent_nodes)}
    parent_functions = [parent_functions[k] for k in sorted(parent_functions)]

    return parent_functions


def _get_parents_of_prob_node(g, node, node_name):
    parent_nodes = [s for s, t in g.in_edges(node_name)]
    parent_functions = [g.nodes[n]["dask"] for n in parent_nodes]

    return parent_functions
