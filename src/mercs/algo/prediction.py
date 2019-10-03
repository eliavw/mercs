import copy
import warnings
from functools import reduce

import networkx as nx
import numpy as np

from .new_prediction import _stochastic_pick
from ..graph import add_imputation_nodes, add_merge_nodes, compose, get_ids, get_nodes
from ..utils import code_to_query, debug_print, encode_attribute, query_to_code

DESC_ENCODING = encode_attribute(1, [1], [2])
TARG_ENCODING = encode_attribute(2, [1], [2])
MISS_ENCODING = encode_attribute(0, [1], [2])

VERBOSITY = 0


def mi_algorithm(g_list, q_code, random_state=997):
    q_desc, q_targ, q_miss = code_to_query(q_code)

    def criterion(g):
        """If g predicts a relevant target = OK"""
        tgt_ids = get_ids(g, kind="tgt")
        return len(set(q_targ).intersection(tgt_ids)) > 0

    g_relevant = [g for g in g_list if criterion(g)]
    g_relevant = [copy.copy(g) for g in g_relevant]

    result = reduce(compose, g_relevant)

    # Post-processing to get a good graph
    add_imputation_nodes(result, q_desc)
    add_merge_nodes(result)

    return result


def ma_algorithm(g_list, q_code, init_threshold=1.00, stepsize=0.05, random_state=997):
    q_desc, q_targ, q_miss = code_to_query(q_code)

    def criterion(g):
        src_ids = get_ids(g, kind="src")
        tgt_ids = get_ids(g, kind="tgt")

        quantifier = len(set(q_desc).intersection(src_ids)) / len(src_ids)
        rel_tgt = tgt_ids.intersection(q_targ)
        yes_no = len(rel_tgt) > 0
        relevance_criterion = int(yes_no) * quantifier

        msg = """
        rel_tgt:                     {}
        quantifier:                 {}
        relevance_criterion:        {}
        """.format(
            rel_tgt, quantifier, relevance_criterion
        )
        debug_print(msg, level=1, V=VERBOSITY)

        return relevance_criterion

    thresholds = np.clip(np.arange(init_threshold, -stepsize, -stepsize), 0, 1)

    for thr in thresholds:
        g_relevant = [g for g in g_list if criterion(g) > thr]
        if len(g_relevant) > 0:
            msg = """
            We have selected {0} model(s) at threshold: {1:.2f}
            """.format(
                len(g_relevant), thr
            )
            debug_print(msg, level=1, V=VERBOSITY)
            break

    g_relevant = [copy.copy(g) for g in g_relevant]
    result = reduce(compose, g_relevant)

    # Post-processing to get a good graph
    add_imputation_nodes(result, q_desc)
    add_merge_nodes(result)

    return result


def mrai_algorithm(
    g_list,
    q_code,
    init_threshold=1.0,
    stepsize=0.1,
    avoid_src=None,
    avoid_tgt=None,
    return_avl_g=False,
    greedy=True,
    stochastic=False,
    imputation_nodes=True,
    merge_nodes=True,
    random_state=997,
):

    # Preliminaries
    if avoid_src is None:
        avoid_src = set([])
    if avoid_tgt is None:
        avoid_tgt = set([])

    q_desc, q_targ, q_miss = code_to_query(q_code)
    nb_tgt = len(q_targ)

    thresholds = np.arange(init_threshold, -1 - stepsize, -stepsize)
    thresholds = np.clip(thresholds, -1, 1)

    # Methods
    def stopping_criterion(list_of_graphs):
        return len(list_of_graphs) > 0

    def criterion(g):
        src_ids = get_ids(g, kind="src")
        tgt_ids = get_ids(g, kind="tgt")

        rel_src = src_ids.intersection(q_desc)
        rel_tgt = tgt_ids.intersection(q_targ)

        bad_src = src_ids.intersection(set(q_targ).union(avoid_src))
        bad_tgt = tgt_ids.intersection(avoid_tgt)  # Things I should not predict

        data_nodes = get_nodes(g, kind="data")

        quantifier = 0
        for n in data_nodes:
            idx = g.node[n]["idx"]

            if idx in rel_src:
                quantifier += g.node[n]["fi"]
            if idx in bad_src:
                quantifier -= g.node[n]["fi"]

        f_01 = max(0.1, quantifier)
        f_02 = int(len(rel_tgt) > 0)
        f_03 = int(len(bad_tgt) > 0)

        if f_02:
            return f_01
        elif f_03:
            # print("Bad target: {}".format(bad_tgt))
            # No relevant targets, and a bad target. This model contributes nothing.
            return -1
        else:
            return -1

    # Actual algorithm
    if stochastic:
        criteria = [criterion(g) for g in g_list]
        picks = _pick(criteria, n=1, random_state=random_state)

        msg = """
                criteria:   {}
                picks:      {}
                """.format(
            criteria, picks
        )
        debug_print(msg, level=1, V=VERBOSITY)

        sel_g = [g_list[g_idx] for g_idx in picks]

        sel_g = [copy.deepcopy(g) for g in sel_g]
        res_g = reduce(compose, sel_g)

    elif nb_tgt == 1 or greedy:
        # Don't stop until you reach the goal
        criteria = [criterion(g) for g in g_list]

        for thr in thresholds:
            sel_g = [
                g_list[g_idx] for g_idx, c in enumerate(criteria) if c > thr
            ]  # Available graphs = Graphs that satisfy the criterion

            if stopping_criterion(sel_g):
                mod_ids = [get_nodes(g, kind="model") for g in sel_g]
                msg = """
                We have selected    {0} model(s) 
                at threshold:       {1:.2f}
                with model ids:     {2}
                """.format(
                    len(sel_g), thr, mod_ids
                )
                debug_print(msg, level=1, V=VERBOSITY)
                break

        sel_g = [copy.deepcopy(g) for g in sel_g]

        for g in sel_g:
            add_imputation_nodes(g, q_desc)

        res_g = reduce(compose, sel_g)

    elif nb_tgt > 1:
        msg = """
        Multi-target case:      
        with target attributes: {}
        """.format(
            q_targ
        )
        debug_print(msg, V=VERBOSITY)

        sel_g = []
        avl_g = g_list
        for i in range(len(q_targ)):
            tgt_q_targ = q_targ[i : i + 1]
            tgt_avoid_src = q_targ[0:i] + q_targ[i + 1 :]
            tgt_q_miss = q_miss + tgt_avoid_src

            tgt_q_code = query_to_code(q_desc, tgt_q_targ, q_miss=tgt_q_miss)

            msg = """
            tgt_q_code:     {}
            """.format(
                tgt_q_code
            )
            debug_print(msg, level=2, V=VERBOSITY)
            tgt_g, avl_g = mrai_algorithm(
                avl_g,
                tgt_q_code,
                avoid_src=tgt_avoid_src,
                return_avl_g=True,
                merge_nodes=False,
            )

            sel_g.append(tgt_g)

        res_g = reduce(compose, sel_g)

    else:
        msg = """
        nb_tgt:     {}
        We expect one or more targets.
        """.format(
            nb_tgt
        )
        raise ValueError(msg)

    if imputation_nodes:
        add_imputation_nodes(res_g, q_desc)
    if merge_nodes:
        add_merge_nodes(res_g)

    if return_avl_g:
        avl_g = [g for g in g_list if g not in sel_g]
        return res_g, avl_g
    else:
        return res_g


def it_algorithm(g_list, q_code, max_steps=4, random_state=997):
    """

    Notes
    -----
    `avl` is short for `available`

    avl_desc:   Available descriptive attributes = Attributes that are available AS descriptive ones.
                So, this means these attributes are KNOWN.
    avl_targ:   Available target attributes = Attributes that are available AS target ones.
                So, this means these attributes are UNKNOWN.

    Parameters
    ----------
    g_list
    q_code
    max_steps
    random_state

    Returns
    -------

    """

    def stopping_criterion(known_attributes, target_attributes):
        """All targets known = DONE"""
        return len(set(target_attributes).difference(known_attributes)) == 0

    # Init
    q_desc, q_targ, q_miss = code_to_query(q_code, return_list=True)
    avl_desc = set(q_desc)
    avl_targ = set(q_targ + q_miss)
    avl_atts = set(q_desc + q_targ + q_miss)

    avl_q = query_to_code(avl_desc, avl_targ, atts=avl_atts)  # 0: Known, 1: Unknown

    avl_g = g_list

    greedy = True
    sel_g = []
    for step in range(max_steps):
        last = step == (max_steps - 1)  # Check if this is our last chance

        if last:
            # All unknown targets are the target of the last step.
            avl_targ = set(q_targ).difference(avl_desc)
            avl_q = query_to_code(avl_desc, avl_targ, atts=avl_atts)
            greedy = False  # Get all remaining targets

        # Get next step
        nxt_g, avl_g = mrai_algorithm(
            avl_g,
            avl_q,
            return_avl_g=True,
            greedy=greedy,
            avoid_src=q_targ,
            avoid_tgt=avl_desc,
            imputation_nodes=True,
            merge_nodes=False,
            random_state=random_state,
        )
        # IT goes from front to back
        sel_g.append(nxt_g)

        # Update query
        nxt_targ = get_ids(nxt_g, kind="targ")
        avl_desc = avl_desc.union(nxt_targ)
        avl_targ = avl_targ.difference(nxt_targ)

        avl_q = query_to_code(avl_desc, avl_targ, atts=avl_atts)

        if stopping_criterion(avl_desc, q_targ):
            break

    # Composing
    res_g = nx.DiGraph()
    avl_desc = set(q_desc)
    for g in sel_g:
        msg = """
        AVL DESC: {}
        """.format(
            avl_desc
        )
        debug_print(msg, level=1, V=VERBOSITY)

        # add_imputation_nodes(g, avl_desc)
        res_g = compose(res_g, g)

        g_targ = get_ids(g, kind="targ")
        avl_desc = avl_desc.union(g_targ)

    add_merge_nodes(res_g)
    res_g = _prune(res_g, q_targ)

    return res_g


def rw_algorithm(g_list, q_code, max_steps=4, random_state=997):
    def stopping_criterion(targets, step_number):
        reason_01 = len(targets) == 0
        reason_02 = step_number == max_steps - 1
        return reason_01 or reason_02

    # Init - diagrams
    avl_g = g_list
    sel_g = []

    # Init - Attribute sets
    q_desc, q_targ, q_miss = code_to_query(q_code, return_list=True)

    sel_targ = set([])
    avl_desc = set(q_desc)
    avl_targ = set(q_targ)
    avl_atts = set(q_desc + q_targ + q_miss)
    avl_q = q_code

    for step in range(max_steps):
        # Get next step
        nxt_g, avl_g = mrai_algorithm(
            avl_g,
            avl_q,
            return_avl_g=True,
            avoid_src=q_targ,
            avoid_tgt=avl_desc,
            stochastic=True,
            merge_nodes=False,
            imputation_nodes=False,
            random_state=random_state,
        )
        # RW goes from back to front
        sel_g.insert(0, nxt_g)

        # Update query
        nxt_g_desc = get_ids(nxt_g, kind="desc")
        nxt_g_targ = get_ids(nxt_g, kind="targ")

        sel_targ = sel_targ.union(nxt_g_targ)

        avl_desc = avl_desc
        avl_targ = set(q_miss).difference(sel_targ).intersection(nxt_g_desc)
        avl_q = query_to_code(avl_desc, avl_targ, atts=avl_atts)

        if stopping_criterion(avl_targ, step):
            break

    # Composing
    res_g = nx.DiGraph()
    avl_desc = set(q_desc)
    for g in sel_g:
        msg = """
        AVL DESC: {}
        """.format(
            avl_desc
        )
        debug_print(msg, level=1, V=VERBOSITY)
        add_imputation_nodes(g, avl_desc)
        res_g = compose(res_g, g)

        g_targ = get_ids(g, kind="targ")
        avl_desc = avl_desc.union(g_targ)

    add_merge_nodes(res_g)
    res_g = _prune(res_g, q_targ)

    return res_g


# Helpers
def _prune(g, tgt_nodes=None):

    msg = """
    tgt_nodes:          {}
    tgt_nodes[0]:       {}
    type(tgt_nodes[0]): {}
    """.format(
        tgt_nodes, tgt_nodes[0], type(tgt_nodes[0])
    )
    debug_print(msg, level=1, V=VERBOSITY)

    if tgt_nodes is None:
        tgt_nodes = [
            n
            for n, out_degree in g.out_degree()
            if out_degree == 0
            if g.nodes[n]["kind"] == "data"
        ]
        msg = """
        tgt_nodes: {}
        """.format(
            tgt_nodes
        )
        debug_print(msg, level=1, V=VERBOSITY)
    elif isinstance(tgt_nodes[0], (int, np.int64)):
        tgt_nodes = [
            n
            for n in g.nodes
            if g.nodes[n]["kind"] == "data"
            if g.nodes[n]["idx"] in tgt_nodes
        ]
    else:
        assert isinstance(tgt_nodes[0], str)

    ancestors = [nx.ancestors(g, source=n) for n in tgt_nodes]
    retain_nodes = reduce(set.union, ancestors, set(tgt_nodes))

    nodes_to_remove = [n for n in g.nodes if n not in retain_nodes]
    for n in nodes_to_remove:
        g.remove_node(n)

    return g

def _pick(criteria, n=1, random_state=997):
    """
    Interpret an array of appropriateness scores as a distribution
    corresponding to the probability of a certain model being selected.


    Parameters
    ----------
    criteria:   list
                Array that quantifies how likely a pick should be.
    n:          int
                Number of picks

    Returns
    -------
    picks:      np.ndarray
                List of indices that were picked
    """

    np.random.seed(random_state)
    criteria += abs(np.min([0, np.min(criteria)]))  # Shift in case of negative values
    norm = np.linalg.norm(criteria, 1)

    if norm > 0:
        criteria = criteria / norm
    else:
        msg = """
        Not a single appropriate model was found, therefore
        making an arbitrary choice.
        """
        warnings.warn(msg)
        # If you cannot be right, be arbitrary
        criteria = [1 / len(criteria) for i in criteria]

    draw = np.random.multinomial(1, criteria, size=n)
    picks = np.where(draw == 1)[1]
    return picks