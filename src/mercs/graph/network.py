import networkx as nx
import numpy as np

from itertools import product
from .graphviz import fix_layout


NODE_PREFIXES = dict(
    data="d",
    vote="v",
    prob="p",
    merge="M",
    imputation="I",
    imputer="I",
    function="f",
    composition="c",
    model="f")

SHAPES = dict(
    data='"circle',
    vote='"triangle"',
    prob='"circle',
    merge='"triangle"',
    imputation='"invtriangle"',
    imputer='"invtriangle"',
    function='"square"',
    composition='"square"',
    model='"square"')


# Building a graph
def model_to_graph(model, types=None, idx=0, composition=False):
    """
    Convert a model to a network.

    Given a model with inputs and outputs, we convert it to a graphical
    model that represents it.

    Parameters
    ----------
    types
    model:      model
                Model which is to be converted to a schematic representation
    idx:        int, default=0
                index (or equivalent:id) of the model

    Returns
    -------

    TODO
    ----
    Numpy throws a runtimewarning somewhere in this method, I need to fix that.

    """

    G = nx.DiGraph()

    # Create nodes

    if composition:
        func_nodes = [_create_comp_node(model, idx=idx)]
    else:
        func_nodes = [_create_func_node(model, idx=idx)]

    nominal = func_nodes[0][1]["out"] == "nominal"  # Nominal model yes/no

    data_nodes_src = [_create_data_node(i, types) for i in model.desc_ids]
    data_nodes_tgt = [_create_data_node(i, types) for i in model.targ_ids]

    G.add_nodes_from(data_nodes_src)
    G.add_nodes_from(data_nodes_tgt)
    G.add_nodes_from(func_nodes)

    if nominal:
        classes = get_classes(model)
        prob_nodes_tgt = [
            _create_prob_node(i, types, classes) for i in model.targ_ids
        ]
        vote_nodes_tgt = [_create_vote_node(i, types, classes) for i in model.targ_ids]
        G.add_nodes_from(prob_nodes_tgt)
        G.add_nodes_from(vote_nodes_tgt)

    # Create edges
    f = [t[0] for t in func_nodes]
    s = [t[0] for t in data_nodes_src]
    t = [t[0] for t in data_nodes_tgt]

    src_to_fnc_edges = [
        (*e, {"idx": d, "fi": fi})
        for e, d, fi in zip(product(s, f), model.desc_ids, model.feature_importances_)
    ]
    G.add_edges_from(src_to_fnc_edges)

    if nominal:
        p = [t[0] for t in prob_nodes_tgt]
        v = [t[0] for t in vote_nodes_tgt]

        fnc_to_prb_edges = [
            (*e, {"idx": d, "classes": G.node[e[1]]["classes"]})
            for e, d in zip(product(f, p), model.targ_ids)
        ]
        prb_to_vot_edges = [
            (*e, {"idx": d}) for e, d in zip(product(p, v), model.targ_ids)
        ]
        vot_to_tgt_edges = [
            (*e, {"idx": d}) for e, d in zip(product(v, t), model.targ_ids)
        ]

        G.add_edges_from(fnc_to_prb_edges)
        G.add_edges_from(prb_to_vot_edges)
        G.add_edges_from(vot_to_tgt_edges)
    else:
        mod_to_tgt_edges = [
            (*e, {"idx": d}) for e, d in zip(product(f, t), model.targ_ids)
        ]
        G.add_edges_from(mod_to_tgt_edges)

    # Graph attributes
    # TODO: Assign this in a single place. In the core file, we assign this to the composites.
    G.graph["desc_ids"] = set(model.desc_ids)
    G.graph["targ_ids"] = set(model.targ_ids)
    G.graph["id"] = idx

    # FI
    G = add_fi_to_graph(G)

    # Final touch
    fix_layout(G)
    return G


def _create_comp_node(model, idx=0):
    comp_node = (
        node_label(idx, kind="composition"),
        dict(
            kind="model",
            idx=idx,
            mod=model,
            src=model.desc_ids,
            tgt=model.targ_ids,
            out=model.out_kind
        ),
    )

    if comp_node[1]["out"] in {"numeric"}:
        comp_node[1]["predict"] = model.predict
        
        # No need to do inference, this goes right away
        comp_node[1]["dask"] = comp_node[1]["predict"]
    elif comp_node[1]["out"] in {"nominal", "mix"}:
        comp_node[1]["predict"] = model.predict
        comp_node[1]["predict_proba"] = model.predict_proba

        # No need to do inference, this goes right away
        comp_node[1]["dask"] = comp_node[1]["predict"]
        comp_node[1]["dask_proba"] = comp_node[1]["predict_proba"]
    else:
        pass
    return comp_node


def _create_func_node(model, idx=0):

    func_node = (
        node_label(idx, kind="function"),
        dict(
            kind="model",
            idx=idx,
            mod=model,
            src=model.desc_ids,
            tgt=model.targ_ids,
            out=model.out_kind
        ),
    )

    if func_node[1]["out"] in {"numeric"}:
        func_node[1]["predict"] = model.predict
    elif func_node[1]["out"] in {"nominal", "mix"}:
        func_node[1]["predict"] = model.predict
        func_node[1]["predict_proba"] = model.predict_proba
    else:
        pass
    return func_node


def _create_data_node(idx, types):
    data_node = (
        node_label(idx, kind="data"),
        dict(kind="data", idx=idx, tgt=[idx], type=types[idx]),
    )
    return data_node


def _create_prob_node(idx, types, classes):
    prob_node = (
        node_label(idx, kind="prob"),
        dict(kind="prob", idx=idx, tgt=[idx], type=types[idx], classes=classes[idx]),
    )
    return prob_node


def _create_vote_node(idx, types, classes):
    vote_node = (
        node_label(idx, kind="vote"),
        dict(kind="vote", idx=idx, tgt=[idx], type=types[idx], classes=classes[idx]),
    )


    return vote_node


def compose_all(g_list):
    R = g_list[0].__class__()

    # Add nodes
    classes = {}
    for g in g_list:
        # Add nodes (one at a time)
        for n, d in g.nodes(data=True):
            R.add_node(n, **d)

            if d["kind"] in {"vote", "prob"}:
                n_classes = d.get("classes", np.array([]))
                r_classes = classes.get(n, np.array([]))
                classes[n] = np.concatenate([r_classes, n_classes])

        # Add edges (in bulk)
        R.add_edges_from(g.edges(data=True))

    # Joining classes
    all_prob_nodes = get_nodes(R, kind="prob")
    all_vote_nodes = get_nodes(R, kind="vote")
    for n in all_prob_nodes.union(all_vote_nodes):
        R.node[n]["classes"] = np.unique(classes[n])

    return R


# Add FI
def add_fi_to_graph(G):
    """
    Add feature importances to input data nodes.

    Parameters
    ----------
    G

    Returns
    -------

    """

    src_nodes = (
        n
        for n, in_degree in G.in_degree()
        if G.nodes[n]["kind"] == "data"
        if in_degree == 0
    )

    src_edges = ((n, G.out_edges(n)) for n in src_nodes)

    for n, edges in src_edges:
        fi = np.mean([G.edges()[e]["fi"] for e in edges])
        G.nodes[n]["fi"] = fi

    return G


# Add
def add_merge_nodes(g):

    relevant_nodes = [n for n in get_nodes(g, kind="data") if len(g.in_edges(n)) > 1]

    for node in relevant_nodes:
        # Collect original node
        original_node_attributes = g.nodes(data=True)[node]
        original_node = (node, original_node_attributes)

        # Convert original
        merge_node_label = convert_data_node(g, node, kind="merge")

        # Insert the original back in, after the merge node
        insert_new_node(g, original_node, merge_node_label, location="back")
    return


def add_imputation_nodes(g, q_desc):

    relevant_nodes = [n for n in get_desc(g, ids=False)
                      if g.nodes[n]["idx"] not in q_desc]

    for node in relevant_nodes:
        convert_data_node(g, node, kind="imputation")
    return


# Node operations
def convert_data_node(G, data_node_label, kind="merge"):

    # Init
    mapping = {}
    prefix = NODE_PREFIXES[kind]
    shape = SHAPES[kind]

    # Actions
    mapping[data_node_label] = "{}({})".format(prefix, data_node_label)
    new_node_label = mapping[data_node_label]

    nx.relabel_nodes(G, mapping, copy=False)

    G.nodes[new_node_label]["shape"] = shape
    G.nodes[new_node_label]["kind"] = kind

    return mapping[data_node_label]


def insert_new_node(G, new_node, existing_node_label, location="behind"):
    G.add_node(new_node[0], **new_node[1])

    if location in {"behind", "after", "back"}:
        G.add_edge(existing_node_label, new_node[0], idx=new_node[1]["idx"])
    elif location in {"before", "front"}:
        G.add_edge(new_node[0], existing_node_label, idx=new_node[1]["idx"])
    else:
        msg = """
        Did not recognize location:     {}
        to add new node relative to existing node.
        """.format(
            location
        )
        raise ValueError(msg)

    return


def compose(G, H):
    """
    Return a new graph of G composed with H.

    Composition is the simple union of the node sets and edge sets.
    The node sets of G and H need not be disjoint.

    Parameters
    ----------
    G, H : graph
       A NetworkX graph

    Returns
    -------
    C: A new graph  with the same type as G

    Notes
    -----
    This is a custom version of nx.compose
    """

    if not G.is_multigraph() == H.is_multigraph():
        raise nx.NetworkXError("G and H must both be graphs or multigraphs.")

    R = G.__class__()

    # add graph attributes, H attributes take precedent over G attributes
    R.graph.update(G.graph)
    R.graph.update(H.graph)

    R.add_nodes_from(G.nodes(data=True))
    R.add_nodes_from(H.nodes(data=True))

    if G.is_multigraph():
        R.add_edges_from(G.edges(keys=True, data=True))
    else:
        R.add_edges_from(G.edges(data=True))
    if H.is_multigraph():
        R.add_edges_from(H.edges(keys=True, data=True))
    else:
        R.add_edges_from(H.edges(data=True))

    # Custom modification
    all_prob_nodes = get_nodes(R, kind="prob")
    all_vote_nodes = get_nodes(R, kind="vote")
    for node in all_prob_nodes.union(all_vote_nodes):
        a = get_attribute(G, node, "classes", np.array([]))
        b = get_attribute(H, node, "classes", np.array([]))

        R.node[node]["classes"] = np.unique(np.concatenate([a, b]))
    return R


# Utils
def get_attribute(G, node, attribute, default=None):
    """
    From a node that may not exist, collect an attribute that may not be there.

    Parameters
    ----------
    G
    node
    attribute
    default

    Returns
    -------

    """

    return G.nodes.get(node, {}).get(attribute, default)


def get_classes(model):
    if len(model.targ_ids) > 1:
        return {
            tgt_idx: model.classes_[idx] for idx, tgt_idx in enumerate(model.targ_ids)
        }
    else:
        # Single target has no indexes, so just pass model.classes_ directly
        return {tgt_idx: model.classes_ for idx, tgt_idx in enumerate(model.targ_ids)}


def node_label(idx, kind="function"):
    """
    Generate a unique name for a node.

    Args:
        idx:        int
                    Node id
        kind:       str, {'func', 'model', 'function'} or {'data}
                    Every node represents either a function or data.

    Returns:

    """
    prefix = NODE_PREFIXES[kind]
    return "{}-{:02d}".format(prefix, idx)


def get_ids(g, kind="desc"):

    if kind in {"s", "src", "source", "d", "desc", "descriptive"}:
        r = get_desc(g, ids=True)
    elif kind in {"t", "tgt", "targ", "target"}:
        r = get_targ(g, ids=True)
    else:
        msg = """
        Did not recognize kind:   {}
        """.format(
            kind
        )
        raise ValueError(msg)

    return r


def get_desc(g, ids=False):
    data_nodes = get_nodes(g, kind="data")

    r = {n for n in data_nodes if len(g.in_edges(n)) == 0}

    if ids:
        r = {g.nodes[n]["idx"] for n in r}

    return r


def get_targ(g, ids=False):
    data_nodes = get_nodes(g, kind="data")

    r = {n for n in data_nodes if len(g.out_edges(n)) == 0}

    if ids:
        r = {g.nodes[n]["idx"] for n in r}

    return r


def get_nodes(g, kind="data"):
    prefix = NODE_PREFIXES[kind]
    r = {n for n in g.nodes if n.startswith(prefix)}
    return r
