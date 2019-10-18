from functools import reduce

import networkx as nx
import numpy as np

from ..utils import code_to_query


NODE_PREFIXES = dict(data="D", model="M", imputation="I")


def build_diagram(m_list, m_sel, q_code, g=None, prune=False):
    if g is None:
        g = nx.DiGraph()

    f_tgt = set([])
    a_src, a_tgt, _ = code_to_query(q_code, return_sets=True)

    g.desc_ids = a_src
    g.targ_ids = a_tgt

    for m_layer in m_sel:
        models = [(m_idx, m_list[m_idx]) for m_idx in m_layer]

        a_src, f_tgt, g = build_diagram_SL(models, a_src, f_tgt, g=g)

    if prune:
        _prune(g)

    return g


def build_diagram_SL(models, a_src, f_tgt, g=None):

    if g is None:
        g = nx.DiGraph()

    valid_src = lambda a: a in a_src
    valid_tgt = lambda a: a not in f_tgt

    e_src = set([])
    e_tgt = set([])

    for m_idx, m in models:
        for a in m.desc_ids:
            if valid_src(a):
                e_src.add((_name(a, kind="data"), _name(m_idx, kind="model")))
                f_tgt.add(a)
            else:
                e_src.add((_name(a, kind="imputation"), _name(a, kind="data")))

    for m_idx, m in models:
        for a in m.targ_ids:
            if valid_tgt(a):
                e_tgt.add((_name(m_idx, kind="model"), (_name(a, kind="data"))))
                a_src.add(a)

    g.add_edges_from(e_src)
    g.add_edges_from(e_tgt)
    return a_src, f_tgt, g


# Helpers
def _name(idx, kind="model"):
    return (NODE_PREFIXES[kind], idx)


def _prune(g):

    tgt_nodes = {("D", n) for n in g.targ_ids}
    ancestors = [nx.ancestors(g, source=n) for n in tgt_nodes]
    retain = set.union(*ancestors, tgt_nodes)
    remove = [n for n in g.nodes if n not in retain]
    g.remove_nodes_from(remove)

    return


