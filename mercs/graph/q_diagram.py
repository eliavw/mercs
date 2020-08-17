import networkx as nx
import numpy as np

from mercs.utils import code_to_query

NODE_PREFIXES = dict(data="D", model="M", imputation="I", composition="C")
NODE_KINDS = {v: k for k, v in NODE_PREFIXES.items()}


def build_diagram(m_list, m_sel, q_code, g=None, prune=False, composition=False):
    if not isinstance(m_sel[0], (list, np.ndarray)):
        m_sel = [m_sel]

    if g is None:
        g = nx.DiGraph()

    f_tgt = set([])
    a_src, a_tgt, _ = code_to_query(q_code, return_sets=True)

    g.desc_ids = a_src.copy()
    g.targ_ids = a_tgt.copy()

    for m_layer in m_sel:
        models = [(m_idx, m_list[m_idx]) for m_idx in m_layer]

        a_src, f_tgt, g = build_diagram_single_layer(models, a_src, f_tgt, g=g, composition=composition)

    if prune:
        _prune(g)

    return g


def build_diagram_single_layer(models, a_src, f_tgt, g=None, composition=False):
    if g is None:
        g = nx.DiGraph()

    node_kind = "composition" if composition else "model"

    def valid_source_id(idx): return idx in a_src
    def valid_target_id(idx): return idx not in f_tgt

    e_src = []
    e_tgt = []

    for m_idx, model in models:
        for a_idx in model.desc_ids:
            if valid_source_id(a_idx):
                e_src.append((v_name(a_idx, kind="data"), v_name(m_idx, kind=node_kind)))
                f_tgt.add(a_idx)
            else:
                e_src.append((v_name(a_idx, kind="imputation"), v_name(m_idx, kind=node_kind)))

    for m_idx, model in models:
        for a_idx in model.targ_ids:
            if valid_target_id(a_idx):
                e_tgt.append((v_name(m_idx, kind=node_kind), (v_name(a_idx, kind="data"))))
                a_src.add(a_idx)

    g.add_edges_from(e_src + e_tgt)
    return a_src, f_tgt, g


# Helpers
def v_name(idx, kind="model"):
    return NODE_PREFIXES[kind], idx


def _prune(g):
    tgt_nodes = {("D", n) for n in g.targ_ids}
    ancestors = [nx.ancestors(g, source=n) for n in tgt_nodes]
    retain = set.union(*ancestors, tgt_nodes)
    remove = [n for n in g.nodes if n not in retain]
    g.remove_nodes_from(remove)
