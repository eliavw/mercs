import numpy as np

from ..utils import code_to_query, get_att_2d


# Strategies
def mi(m_codes, m_fimps, m_score, q_code, m_avl=None, random_state=997):
    return mrai(
        m_codes,
        m_fimps,
        m_score,
        q_code,
        thresholds=False,
        a_src=None,
        a_tgt=None,
        m_avl=m_avl,
        any_target=False,
        stochastic=False,
        random_state=random_state,
    )


def mrai(
    m_codes,
    m_fimps,
    m_score,
    q_code,
    a_src=None,
    a_tgt=None,
    m_avl=None,
    init_threshold=1.0,
    stepsize=0.1,
    any_target=False,
    stochastic=False,
    thresholds=None,
    random_state=997,
):
    # Init
    if m_avl is None:
        m_avl = np.arange(m_codes.shape[0], dtype=np.int32)

    if thresholds is None:
        thresholds = _init_thresholds(init_threshold, stepsize)

    if a_src is None or a_tgt is None:
        a_src, a_tgt, _ = code_to_query(q_code)

    # Criterion
    c_flt = criterion(m_score, m_filter=m_avl, a_filter=a_tgt, aggregation=True)
    m_avl = m_avl[c_flt.flat > 0]

    c_tgt = criterion(m_score, m_filter=m_avl, a_filter=a_tgt, aggregation=None)
    c_src = criterion(m_fimps, m_filter=m_avl, a_filter=a_src, aggregation=True)

    c_all = c_src.reshape(-1, 1) * c_tgt

    # Pick
    m_sel_idx = pick(
        c_all,
        thresholds=thresholds,
        any_target=any_target,
        stochastic=stochastic,
        random_state=random_state,
    )
    m_sel = m_avl[m_sel_idx]

    return m_sel


def it(
    m_codes,
    m_fimps,
    m_score,
    q_code,
    m_avl=None,
    max_steps=4,
    init_threshold=1.0,
    stepsize=0.1,
    random_state=997,
):
    m_sel = []
    thresholds = _init_thresholds(init_threshold, stepsize)
    any_target = True
    stochastic = False

    q_desc, q_targ, q_miss = code_to_query(q_code)
    a_src = q_desc
    a_tgt = np.hstack([q_targ, q_miss])

    if m_avl is None:
        m_avl = np.arange(m_codes.shape[0], dtype=np.int32)

    for step in range(max_steps):

        # Check if this is our last chance
        last = step + 1 == max_steps
        if last:
            any_target = False  # Finish the job
            a_tgt = np.setdiff1d(
                q_targ, a_src
            )  # Focus exclusively on non-predicted q_targ attributes.

        step_m_sel = mrai(
            m_codes,
            m_fimps,
            m_score,
            None,
            a_src=a_src,
            a_tgt=a_tgt,
            m_avl=m_avl,
            stochastic=stochastic,
            any_target=any_target,
            thresholds=thresholds,
            random_state=random_state,
        )

        a_prd = get_att_2d(m_codes[step_m_sel, :], kind="targ")

        a_src = np.union1d(a_src, a_prd)
        a_tgt = np.setdiff1d(a_tgt, a_prd)

        m_avl = np.setdiff1d(m_avl, step_m_sel)
        m_sel.append(step_m_sel)

        if _stopping_criterion_it(q_targ, a_src):
            break

        if len(step_m_sel) == 0:
            raise ValueError(
                "No progress was made. This indicates an impossible query."
            )

    return m_sel


# MRAI-IT-RW Criterion
def criterion(m_matrix, m_filter=None, a_filter=None, aggregation=None):
    """
    Typical usecase 
    
    m_matrix = m_fimps
    
    m_filter = available models
    a_filter = available attributes.
    
    """
    nb_rows = len(m_filter)
    nb_cols = len(a_filter)

    m_idx = a_filter + m_filter.reshape(-1, 1) * m_matrix.shape[1]
    c_matrix = m_matrix.take(m_idx.flat).reshape(nb_rows, nb_cols)

    if aggregation is None:
        return c_matrix
    else:
        return np.sum(c_matrix, axis=1).reshape(-1, 1).astype(np.float32)


# Picks
def pick(
    criteria, thresholds=False, any_target=False, stochastic=False, random_state=997
):

    if thresholds is False:
        return np.where(criteria >= 0)[0]
    else:
        m_sel = []

        picking_function = _stochastic_pick if stochastic else _greedy_pick

        if any_target:
            criteria = np.max(criteria, axis=1).reshape(-1, 1)

        for c_idx in range(criteria.shape[1]):
            m_sel.extend(
                picking_function(
                    criteria[:, c_idx], thresholds=thresholds, random_state=random_state
                )
            )

        return np.unique(m_sel)


def _greedy_pick(c_all, thresholds=None, **kwargs):

    for thr in thresholds:
        m_sel = np.where(c_all >= thr)[0]
        if _stopping_criterion_greedy_pick(m_sel):
            break
    return m_sel


def _stochastic_pick(c_all, random_state=997, **kwargs):
    np.random.seed(random_state)
    norm = np.linalg.norm(c_all, 1)

    if norm > 0:
        distribution = c_all / np.sum(c_all)
    else:
        distribution = np.full(len(c_all), 1 / len(c_all))

    draw = np.random.multinomial(1, distribution, size=1)
    return np.where(draw == 1)[1]


# Stopping Criteria
def _stopping_criterion_it(q_targ, a_src):
    return np.setdiff1d(q_targ, a_src).shape[0] == 0


def _stopping_criterion_greedy_pick(m_sel):
    return len(m_sel) > 0


# Helpers
def _init_thresholds(init_threshold, stepsize, tolerance=0.01):
    thresholds = np.arange(init_threshold, -stepsize, -stepsize, dtype=np.float32)
    
    # Otherwise rounding errors in feature importances fuck your shit up.
    thresholds[0] = thresholds[0]-tolerance
    return thresholds
