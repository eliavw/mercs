import numpy as np
import warnings

from sklearn.preprocessing import minmax_scale, normalize

from ..utils import code_to_query, get_att_2d, TARG_ENCODING

EPSILON = 0.00001


# Strategies
def mi(
    m_codes,
    m_fimps,
    m_score,
    q_code,
    a_src=None,
    a_tgt=None,
    m_avl=None,
    random_state=997,
):
    # Init
    m_sel = []
    m_avl = _init_m_avl(m_codes, m_avl=m_avl)
    a_src, a_tgt = _init_a_src_a_tgt(q_code=q_code, a_src=a_src, a_tgt=a_tgt)

    c_all = criterion(m_codes, row_filter=m_avl, col_filter=a_tgt, aggregation=False)
    m_sel_idx = np.unique(np.where(c_all == TARG_ENCODING)[0]).astype(int)
    m_sel = m_avl[m_sel_idx]

    if len(m_sel) == 0:
        return None
    else:
        return m_sel


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
    picking_function="greedy",
    thresholds=None,
    random_state=997,
):
    # Init
    m_sel = []
    m_avl = _init_m_avl(m_codes, m_avl=m_avl)
    thresholds = _init_thresholds(init_threshold, stepsize)
    a_src, a_tgt = _init_a_src_a_tgt(q_code=q_code, a_src=a_src, a_tgt=a_tgt)

    # Filtering
    m_flt = mi(
        m_codes,
        m_fimps,
        m_score,
        None,
        a_src=a_src,
        a_tgt=a_tgt,
        m_avl=m_avl,
        random_state=random_state,
    )

    if m_flt is None:
        warnings.warn("You reached a dead end.")
        return None
    else:
        # Criterion
        c_tgt = criterion(
            m_score, row_filter=m_flt, col_filter=a_tgt, aggregation=False
        )
        c_src = criterion(m_fimps, row_filter=m_flt, col_filter=a_src, aggregation=True)
        c_all = c_src * c_tgt + EPSILON

        if any_target:
            c_all = np.max(c_all, axis=1).reshape(-1, 1)

        # Normalize
        c_nrm = normalize(c_all, norm="l1", axis=0)
        c_nrm += 1 - np.max(c_nrm)

        # Pick
        m_sel_idx = pick(
            c_all,
            thresholds=thresholds,
            picking_function=picking_function,
            random_state=random_state,
        )
        m_sel = m_flt[m_sel_idx]

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
    picking_function="greedy",
    random_state=997,
):
    m_sel = []
    m_avl = _init_m_avl(m_codes, m_avl=m_avl)
    thresholds = _init_thresholds(init_threshold, stepsize)

    any_target = True
    q_desc, q_targ, q_miss = code_to_query(q_code)
    a_src = q_desc
    a_tgt = np.hstack([q_targ, q_miss])

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
            any_target=any_target,
            thresholds=thresholds,
            picking_function=picking_function,
            random_state=random_state,
        )

        if step_m_sel is None:
            raise ValueError(
                "No progress was made. This indicates an impossible query."
            )

        a_prd = get_att_2d(m_codes[step_m_sel, :], kind="targ")

        a_src = np.union1d(a_src, a_prd)
        a_tgt = np.setdiff1d(a_tgt, a_prd)

        m_avl = np.setdiff1d(m_avl, step_m_sel)
        m_sel.append(step_m_sel)

        if _stopping_criterion_it(q_targ, a_src):
            break

    return m_sel


def rw(
    m_codes,
    m_fimps,
    m_score,
    q_code,
    m_avl=None,
    nb_walks=5,
    max_steps=4,
    init_threshold=1.0,
    stepsize=0.1,
    straight=True,
    picking_function="stochastic",
    random_state=997,
):

    random_walks = [
        walk(
            m_codes,
            m_fimps,
            m_score,
            q_code,
            m_avl=m_avl,
            max_steps=max_steps,
            init_threshold=init_threshold,
            stepsize=stepsize,
            straight=straight,
            picking_function=picking_function,
            random_state=random_state + i * 997,  # Otherwise you do identical walks!
        )
        for i in range(nb_walks)
    ]

    return tuple(random_walks)


def walk(
    m_codes,
    m_fimps,
    m_score,
    q_code,
    m_avl=None,
    max_steps=4,
    init_threshold=1.0,
    stepsize=0.1,
    straight=True,
    picking_function="stochastic",
    random_state=997,
):
    m_sel = []
    m_avl = _init_m_avl(m_codes, m_avl=m_avl)
    thresholds = _init_thresholds(init_threshold, stepsize)

    any_target = True
    q_desc, q_targ, q_miss = code_to_query(q_code)
    a_src = q_desc
    a_tgt = q_targ

    for step in reversed(range(max_steps)):

        step_m_sel = mrai(
            m_codes,
            m_fimps,
            m_score,
            None,
            a_src=a_src,
            a_tgt=a_tgt,
            m_avl=m_avl,
            any_target=any_target,
            thresholds=thresholds,
            picking_function=picking_function,
            random_state=random_state + step,
        )

        if step_m_sel is None:
            break

        # Step source attributes = potential targets
        s_src = get_att_2d(m_codes[step_m_sel, :], kind="desc")

        # Allow to predict anything that follows
        if not straight:
            s_tgt = get_att_2d(m_codes[step_m_sel, :], kind="targ")  # HACK
            s_src = np.union1d(a_tgt, s_src)  # HACK

        a_tgt = np.setdiff1d(s_src, a_src)

        m_avl = np.setdiff1d(m_avl, step_m_sel)
        m_sel.insert(0, step_m_sel)

        if _stopping_criterion_rw(a_tgt, step, m_avl):
            break

    return m_sel


# MRAI-IT-RW Criterion
def criterion(matrix, row_filter=None, col_filter=None, aggregation=False):
    """
    Typical usecase 
    
    matrix = m_fimps
    
    row_filter = available models
    col_filter = available attributes.
    
    """
    nb_rows = len(row_filter)
    nb_cols = len(col_filter)

    m_idx = col_filter + row_filter.reshape(-1, 1) * matrix.shape[1]
    c_matrix = matrix.take(m_idx.flat).reshape(nb_rows, nb_cols)

    if aggregation:
        return np.sum(c_matrix, axis=1).reshape(-1, 1)
    else:
        return c_matrix


# Picks
def _all_pick(criterion, **kwargs):
    return np.where(criterion >= 0.0)[0]


def _greedy_pick(criterion, thresholds=None, **kwargs):

    for thr in thresholds:
        m_sel = np.where(criterion >= thr)[0]

        if _stopping_criterion_greedy_pick(m_sel):
            break
    return m_sel


def _stochastic_pick(c_all, random_state=997, normalize=True, **kwargs):
    assert len(c_all) > 0

    if normalize:
        c_all = _criteria_to_distribution(c_all)

    np.random.seed(random_state)

    try:
        draw = np.random.multinomial(1, c_all, size=1)
    except ValueError:
        draw = np.random.multinomial(1, c_all * 0.95, size=1)

    m_sel = np.where(draw == 1)[1]
    return m_sel


def _random_pick(criterion, random_state=997, **kwargs):
    criterion = _criteria_to_uniform_distribution(criterion)
    return _stochastic_pick(criterion, normalize=False)


PICKS = dict(
    all=_all_pick, greedy=_greedy_pick, stochastic=_stochastic_pick, random=_random_pick
)


def pick(criteria, thresholds=None, picking_function="greedy", random_state=997):
    m_sel = []

    for c_idx in range(criteria.shape[1]):
        c_act = criteria[:, c_idx]
        if not np.any(c_act):
            c_act = _criteria_to_uniform_distribution(c_act)

        m_sel.extend(
            PICKS[picking_function](
                c_act, thresholds=thresholds, random_state=random_state
            )
        )

    return np.unique(m_sel)


# Stopping Criteria
def _stopping_criterion_rw(q_targ, step, m_avl):
    reason_01 = len(q_targ) == 0
    reason_02 = step == 0
    reason_03 = len(m_avl) == 0
    return reason_01 or reason_02 or reason_03


def _stopping_criterion_it(q_targ, a_src):
    return np.setdiff1d(q_targ, a_src).shape[0] == 0


def _stopping_criterion_greedy_pick(m_sel):
    return len(m_sel) > 0


# Helpers
def _criteria_to_uniform_distribution(criteria):
    return np.full(criteria.shape, 1 / criteria.shape[0])


def _criteria_to_distribution(criteria, epsilon=EPSILON):
    # Criteria = VECTOR in [0,1]
    norm = np.sum(criteria)

    if norm < epsilon:
        dist = _criteria_to_uniform_distribution(criteria)
    else:
        dist = criteria / norm
    return dist


# Inits
def _init_thresholds(init_threshold, stepsize, thresholds=None, tolerance=EPSILON):

    if thresholds is None:
        thresholds = np.arange(init_threshold, -1 - stepsize, -stepsize)

        # Otherwise rounding errors in feature importances fuck your shit up.
        thresholds[0] = thresholds[0] - tolerance

    return thresholds


def _init_m_avl(m_codes, m_avl=None):
    if m_avl is None:
        return np.arange(m_codes.shape[0], dtype=int)
    else:
        return m_avl


def _init_a_src_a_tgt(q_code=None, a_src=None, a_tgt=None):
    if q_code is not None:
        a_src, a_tgt, _ = code_to_query(q_code)
        return a_src, a_tgt
    else:
        return a_src, a_tgt

