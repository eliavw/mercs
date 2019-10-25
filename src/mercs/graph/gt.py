from ..utils import code_to_query
from graph_tool.all import GraphView, Graph
import numpy as np

NODE_PREFIXES = dict(data="D", model="M", imputation="I")


def build_graph(m_codes, m_list):
    n_models, n_attributes = m_codes.shape

    g = Graph()

    v_map = {}
    names = g.new_vertex_property("object")

    v_atts = g.add_vertex(n_attributes)
    v_mods = g.add_vertex(n_models)
    v_imps = g.add_vertex(n_attributes)

    for v_idx, v in enumerate(v_atts):
        v_n = v_name(v_idx, kind="data")
        v_map[v_n] = int(v)
        names[v] = v_n

    for v_idx, v in enumerate(v_mods):
        v_n = v_name(v_idx, kind="model")
        v_map[v_n] = int(v)
        names[v] = v_n

        in_edges = ((d, v) for d in m_list[v_idx].desc_ids)
        out_edges = ((v, t) for t in m_list[v_idx].targ_ids)

        g.add_edge_list(in_edges)
        g.add_edge_list(out_edges)

    for v_idx, v in enumerate(v_imps):
        v_n = v_name(v_idx, kind="imputation")
        v_map[v_n] = int(v)
        names[v] = v_n

    g.vp.names = names
    g.v_map = v_map
    return g


def build_diagram(g, m_list, m_sel, q_code, prune=False):
    g.clear_filters()

    if not isinstance(m_sel[0], (list, np.ndarray)):
        m_sel = [m_sel]

    # Init (graph properties)
    g_a_src = g.new_vertex_property("bool", False)
    g_f_tgt = g.new_vertex_property("bool", False)

    v_filter = g.new_vertex_property("bool", False)
    e_filter = g.new_edge_property("bool", False)

    # Availability of attributes (= available sources and forbidden targets)
    f_tgt = set([])
    a_src, a_tgt, _ = code_to_query(q_code, return_sets=True)

    a_src = [g.v_map[v_name(a, kind="data")] for a in a_src]
    f_tgt = [g.v_map[v_name(a, kind="data")] for a in f_tgt]

    for a in a_src:
        g_a_src[a] = True

    for a in f_tgt:
        g_f_tgt[a] = True

    imputation_edges = []
    for m_layer in m_sel:
        vertices = [g.v_map[v_name(m_idx, kind="model")] for m_idx in m_layer]
        vertices = [(v_idx, g.vertex(v_idx)) for v_idx in vertices]

        imputation_edges_single_layer = build_diagram_single_layer(
            vertices, g_a_src, g_f_tgt, v_filter, e_filter, g
        )
        imputation_edges.extend(imputation_edges_single_layer)

    g.add_edge_list(imputation_edges, eprops=[e_filter])

    q_diagram = GraphView(g, efilt=e_filter, vfilt=v_filter)

    if prune:
        q_diagram = _prune(q_diagram, a_tgt, g.v_map)

    # Attributes based on query
    q_diagram.desc_ids = a_src
    q_diagram.targ_ids = a_tgt

    return q_diagram


def build_diagram_single_layer(vertices, g_a_src, g_f_tgt, v_filter, e_filter, g):

    imputation_edges = []

    for v_idx, vertex in vertices:
        v_filter[vertex] = True

        for e in vertex.in_edges():
            a = e.source()
            if g_a_src[a]:
                v_filter[a] = True
                g_f_tgt[a] = True
                e_filter[e] = True
            else:
                i_idx = g.v_map[v_name(int(a), kind="imputation")]
                v_filter[i_idx] = True
                imputation_edges.append([i_idx, v_idx, True])

    for v_idx, vertex in vertices:
        for e in vertex.out_edges():
            a = e.target()

            if not g_f_tgt[a]:
                e_filter[e] = True
                g_a_src[a] = True
                v_filter[a] = True


    return imputation_edges


# Utils
def v_name(idx, kind="model"):
    return (NODE_PREFIXES[kind], idx)


# Helpers
def _prune(g, q_tgt, v_map):
    result = g.new_vertex_property("bool", False)

    for a in q_tgt:
        vertex = g.vertex(v_map[("D", a)])
        result[vertex] = True
        _ancestor_filter(g, vertex, result)

    return GraphView(g, vfilt=result)


def _ancestor_filter(g, v, result=None):
    if result is None:
        result = g.new_vertex_property("bool", False)

    in_vertices = g.get_in_neighbors(v)
    in_degrees = g.get_in_degrees(in_vertices)

    for d, v in zip(in_degrees, in_vertices):
        result[v] = True
        if d:
            _ancestor_filter(g, v, result)
    return result

