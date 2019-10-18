import warnings
from functools import reduce

import networkx as nx
import numpy as np
from networkx import NetworkXNoCycle
from networkx.algorithms import find_cycle

from ..graph import add_imputation_nodes, add_merge_nodes, compose_all, get_ids
from ..utils import (
    change_role,
    debug_print,
    get_att,
    ENCODING,
    DESC_ENCODING,
    TARG_ENCODING,
    MISS_ENCODING,
)

VERBOSITY = 0


def mi(g_list, q_code, fi, t_codes, random_state=997):
    return mrai(
        g_list,
        q_code,
        fi,
        t_codes,
        init_threshold=-0.1,
        stepsize=0.1,
        greedy=True,
        stochastic=False,
        any_target=False,
        imputation_nodes=True,
        merge_nodes=True,
        return_avl_g=False,
        random_state=random_state,
    )


def mi_old(g_list, q_code, fi, t_codes, random_state=997):

    # Init
    mod_ids = [g.graph["id"] for g in g_list]

    # Calculate criteria
    avl_att = _att_indicator(q_code, kind="desc")
    avl_mod = _mod_indicator(mod_ids, q_code, t_codes, any_target=True)
    criteria = avl_mod

    # Pick
    g_sel = _avl_pick(g_list, criteria)

    # Build new graph
    q_diagram = _build_diagram(
        g_sel, avl_att, imputation_nodes=True, merge_nodes=True, test=True
    )

    return q_diagram


def mrai(
    g_list,
    q_code,
    fi,
    t_codes,
    init_threshold=1.0,
    stepsize=0.1,
    greedy=True,
    stochastic=False,
    any_target=False,
    imputation_nodes=True,
    merge_nodes=True,
    return_avl_g=False,
    random_state=997,
):

    # Init
    thresholds = _init_thresholds(init_threshold, stepsize)

    mod_ids = [g.graph["id"] for g in g_list]

    # Calculate criteria
    avl_att = _att_indicator(q_code, kind="desc")
    avl_mod = _mod_indicator(mod_ids, q_code, t_codes, any_target=any_target)
    criteria = _mod_criteria(mod_ids, avl_att, fi)

    criteria = (
        criteria * avl_mod
    )  # All the criteria of unavailable models are set to zero

    criteria[np.where(avl_mod<=0)[0], :] = -1  # Unavailable models are set to -1, available ones keep their score.

    # Pick
    for c_idx in range(criteria.shape[1]):
        criterion = criteria[:, c_idx]

        if greedy:
            g_sel = _greedy_pick(g_list, criterion, thresholds)
        elif stochastic:
            g_sel = _stochastic_pick(g_list, criterion, n=1, random_state=random_state)
        else:
            msg = """
            You either need to pick in a greedy fashion (picking the most appropriate models first) or
            in a stochastic fashion (more likely to pick the most appropriate models). Both of those
            options evaluated to False so I have no idea what you are trying to do.
            """
            raise NotImplementedError(msg)

    # Build new graph
    q_diagram = _build_diagram(
        g_sel,
        avl_att,
        imputation_nodes=imputation_nodes,
        merge_nodes=merge_nodes,
        test=True,
    )

    if return_avl_g:
        sel_ids = [g.graph["id"] for g in g_sel]
        avl_g = [g for g in g_list if g.graph["id"] not in sel_ids]
        return q_diagram, avl_g
    else:
        return q_diagram


def it(
    g_list,
    q_code,
    fi,
    t_codes,
    max_steps=4,
    init_threshold=1.0,
    stepsize=0.1,
    random_state=997,
):
    # Init
    stochastic = False
    greedy = True
    any_target = True

    g_sel = []
    avl_att = _att_indicator(q_code, kind="desc")
    tgt_att = _att_indicator(q_code, kind="targ")

    step_g_list = g_list
    step_q_code = change_role(q_code, MISS_ENCODING, TARG_ENCODING)

    for step in range(max_steps):

        last = step + 1 == max_steps  # Check if this is our last chance
        if last:
            any_target = False  # We finish the job
            q_targ_todo = _targ_todo(step_q_code, tgt_att)
            step_q_code = change_role(step_q_code, TARG_ENCODING, MISS_ENCODING)
            step_q_code[q_targ_todo] = TARG_ENCODING

        step_q_diagram, step_g_list = mrai(
            step_g_list,
            step_q_code,
            fi,
            t_codes,
            init_threshold=init_threshold,
            stepsize=stepsize,
            greedy=greedy,
            stochastic=stochastic,
            any_target=any_target,
            imputation_nodes=True,
            merge_nodes=False,
            return_avl_g=True,
            random_state=random_state,
        )

        # Remember graph of this step
        g_sel.append(step_q_diagram)

        # Update query
        step_targ = list(step_q_diagram.graph["targ_ids"])
        step_q_code[step_targ] = DESC_ENCODING

        if _stopping_criterion_it(step_q_code, tgt_att):
            break

    # Build diagram
    q_diagram = _build_diagram(
        g_sel,
        avl_att,
        tgt_att,
        imputation_nodes=False,
        merge_nodes=True,
        prune=True,
        test=False,
    )
    return q_diagram


def rw(
    g_list,
    q_code,
    fi,
    t_codes,
    max_steps=4,
    nb_walks=1,
    init_threshold=1.0,
    stepsize=0.1,
    random_state=997,
):

    q_diagrams = [
        walk(
            g_list,
            q_code,
            fi,
            t_codes,
            max_steps=max_steps,
            init_threshold=init_threshold,
            stepsize=stepsize,
            random_state=random_state + i,
        )
        for i in range(nb_walks)
    ]

    return q_diagrams


def walk(
    g_list,
    q_code,
    fi,
    t_codes,
    max_steps=4,
    init_threshold=1.0,
    stepsize=0.1,
    random_state=997,
):

    # Init
    avl_att = _att_indicator(q_code, kind="desc")
    tgt_att = _att_indicator(q_code, kind="targ")
    stochastic = True
    greedy = False
    any_target = True

    g_sel = []
    g_tgt = set([])
    q_desc = set(get_att(q_code, kind="desc"))
    q_miss = set(get_att(q_code, kind="miss"))

    step_g_list = g_list
    step_q_code = q_code.copy()

    # Generate chain
    for step in reversed(range(max_steps)):
        step_q_diagram, step_g_list = mrai(
            step_g_list,
            step_q_code,
            fi,
            t_codes,
            init_threshold=init_threshold,
            stepsize=stepsize,
            greedy=greedy,
            stochastic=stochastic,
            any_target=any_target,
            imputation_nodes=False,
            merge_nodes=True,
            return_avl_g=True,
            random_state=random_state,
        )

        # Update query
        step_targ = step_q_diagram.graph["targ_ids"]
        step_desc = step_q_diagram.graph["desc_ids"]

        # Remember graph of this step (Append in front!)
        g_sel.insert(0, step_q_diagram)
        g_tgt = g_tgt.union(step_targ)

        # Extract info
        step_q_code[:] = MISS_ENCODING
        step_q_code[list(q_desc)] = DESC_ENCODING
        step_q_code[list(q_miss.intersection(step_desc))] = TARG_ENCODING
        step_q_code[
            list(g_tgt)
        ] = (
            MISS_ENCODING
        )  # Consider future targets forbidden (this might not be necessary)

        if _stopping_criterion_rw(step_q_code, step):
            break

    # Build diagram
    avl_desc = set(q_desc)
    for g in g_sel:
        add_imputation_nodes(g, avl_desc)
        avl_desc = avl_desc.union(g.graph["targ_ids"])

    q_diagram = _build_diagram(
        g_sel,
        avl_att,
        tgt_att,
        imputation_nodes=False,
        merge_nodes=True,
        prune=True,
        test=False,
    )
    return q_diagram


def rev(
    g_list,
    q_code,
    fi,
    t_codes,
    max_steps=4,
    init_threshold=1.0,
    stepsize=0.1,
    random_state=997,
):

    # Init
    avl_att = _att_indicator(q_code, kind="desc")
    tgt_att = _att_indicator(q_code, kind="targ")
    stochastic = True
    greedy = False
    any_target = True

    g_sel = []
    g_tgt = set([])
    q_desc = set(get_att(q_code, kind="desc"))
    q_miss = set(get_att(q_code, kind="miss"))

    step_g_list = g_list
    step_q_code = q_code

    # Generate chain
    for step in reversed(range(max_steps)):
        step_q_diagram, step_g_list = mrai(
            step_g_list,
            step_q_code,
            fi,
            t_codes,
            init_threshold=init_threshold,
            stepsize=stepsize,
            greedy=greedy,
            stochastic=stochastic,
            any_target=any_target,
            imputation_nodes=False,
            merge_nodes=True,
            return_avl_g=True,
            random_state=random_state,
        )

        # Update query
        step_targ = step_q_diagram.graph["targ_ids"]
        step_desc = step_q_diagram.graph["desc_ids"]

        # Remember graph of this step (Append in front!)
        g_sel.insert(0, step_q_diagram)
        g_tgt = g_tgt.union(step_targ)

        # Extract info
        step_q_code[list(q_miss.intersection(step_desc))] = TARG_ENCODING
        step_q_code[
            list(g_tgt)
        ] = (
            MISS_ENCODING
        )  # Consider future targets forbidden (this might not be necessary)

        if _stopping_criterion_rw(step_q_code, step):
            break

    # Build diagram
    avl_desc = set(q_desc)
    for g in g_sel:
        print(g.nodes())
        print(g.graph["desc_ids"], g.graph["targ_ids"])
        add_imputation_nodes(g, avl_desc)
        avl_desc = avl_desc.union(g.graph["targ_ids"])

    q_diagram = _build_diagram(
        g_sel,
        avl_att,
        tgt_att,
        imputation_nodes=False,
        merge_nodes=True,
        prune=True,
        test=False,
    )
    return q_diagram


# Stopping criteria
def _stopping_criterion_it(step_q_code, tgt_att):
    q_targ_todo = _targ_todo(step_q_code, tgt_att)
    return not np.any(q_targ_todo)


def _stopping_criterion_rw(step_q_code, step):
    reason_01 = len(get_att(step_q_code, kind="targ")) == 0
    reason_02 = step == 0
    return reason_01 or reason_02


def _stopping_criterion_greedy_pick(list_of_graphs):
    return len(list_of_graphs) > 0


def _targ_todo(step_q_code, tgt_att):
    """
    Everything you want to know and do not know yet.
    """
    return (tgt_att & (step_q_code != DESC_ENCODING)).astype(bool)


# Criteria-calculations
def _mod_criteria(mod_ids, avl_att, fi):
    return np.dot(fi[mod_ids, :], avl_att).reshape(-1, 1)


def _mod_indicator(mod_ids, q_code, t_codes, any_target=True):
    """Indicator vector of available models.

    Available models are models which are eligible for selection by 
    the algorithm. Currently, this just means that the model has to be 
    relevant, it has to predict at least one target that is required by the query.

    Hence, changes in behaviour (e.g., predicting a missing attribute is also OK) have to
    be realized at the level of the query, not here.

    Parameters
    ----------
    mod_ids: 
    q_code: 
    t_codes: 

    Returns
    -------

    """

    avl_tgt = np.eye(q_code.shape[0], dtype=int)[q_code == TARG_ENCODING].T
    avl_mod = np.dot(t_codes[mod_ids, :], avl_tgt)

    if any_target:
        # If any target works, I can just sum them up.
        # Predicting multiple interesting targets will help to some extent! (sum)
        avl_mod = np.sum(avl_mod, axis=1, keepdims=True)
        
        ones_idx = np.where(avl_mod > 0.)
        avl_mod[:] = 0
        avl_mod[ones_idx] = 1

        #avl_mod = np.clip(np.max(avl_mod, axis=1, keepdims=True), a_min=0, a_max=1)

    return avl_mod


def _att_indicator(q_code, kind="desc"):
    return (q_code == ENCODING[kind]).astype(int)


# Pick models
def _stochastic_pick(g_list, criteria, n=1, random_state=997):
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

    return [g_list[g_idx].copy() for g_idx in picks]


def _greedy_pick(g_list, criteria, thresholds):
    # I do not think I need to copy!
    for thr in thresholds:
        g_sel = [g_list[idx].copy() for idx, c in enumerate(criteria) if c >= thr]
        if _stopping_criterion_greedy_pick(g_sel):
            break
    return g_sel


def _avl_pick(g_list, criteria):
    g_sel = [g_list[idx].copy() for idx, c in enumerate(criteria) if c > 0.8]
    return g_sel


# Initializations
def _init_thresholds(init_threshold, stepsize):
    """Initialize thresholds array based on its two defining parameters.

    Parameters
    ----------
    init_threshold: 
    stepsize: 

    Returns
    -------

    """

    thresholds = np.arange(init_threshold, -1 - stepsize, -stepsize)
    thresholds = np.clip(thresholds, -1, 1)
    return thresholds


# Graph


def _build_diagram(
    g_sel,
    avl_att,
    tgt_att=None,
    imputation_nodes=True,
    merge_nodes=False,
    prune=False,
    test=False,
):
    if imputation_nodes:
        q_desc = np.where(avl_att)[0]
        for g in g_sel:
            add_imputation_nodes(g, q_desc)

    q_diagram = compose_all(g_sel)

    if merge_nodes:
        add_merge_nodes(q_diagram)

    if prune:
        assert tgt_att is not None, "If you prune, you need to provide tgt_att"
        q_targ = np.where(tgt_att)[0]
        _prune(q_diagram, q_targ)

    if test:
        try:
            cycles = find_cycle(q_diagram)
            msg = """
            Found a cycle!
            Cycle was: {}
            """.format(
                cycles
            )
            raise ValueError(msg)
        except NetworkXNoCycle:
            pass

    q_diagram.graph["desc_ids"] = get_ids(q_diagram, kind="desc")
    q_diagram.graph["targ_ids"] = get_ids(q_diagram, kind="targ")

    return q_diagram


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
    return
