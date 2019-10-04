import numpy as np
import warnings
from ..utils import DESC_ENCODING, TARG_ENCODING, MISS_ENCODING


def base_selection_algorithm(metadata, nb_targets=1, nb_iterations=1, random_state=997):
    m_codes = random_selection_algorithm(
        metadata,
        nb_targets=nb_targets,
        nb_iterations=nb_iterations,
        fraction_missing=0.0,
        random_state=random_state,
    )
    return m_codes


def random_selection_algorithm(
    metadata, nb_targets=1, nb_iterations=1, fraction_missing=0.2, random_state=997
):
    if isinstance(fraction_missing, list):
        codes = []
        for f in fraction_missing:
            codes.append(
                random_selection_algorithm(
                    metadata,
                    nb_targets=nb_targets,
                    nb_iterations=nb_iterations,
                    fraction_missing=f,
                    random_state=random_state,
                )
            )
        m_codes = np.vstack(codes)
        return m_codes
    else:
        # Init
        np.random.seed(random_state)
        nb_attributes = metadata["n_attributes"]
        nb_targets = _set_nb_targets(nb_targets, nb_attributes)

        codes = []
        for attribute_kind in {"nominal_attributes", "numeric_attributes"}:
            potential_targets = np.array(list(metadata[attribute_kind]))

            if potential_targets.shape[0] > 0:
                for iterations in range(nb_iterations):
                    m_codes = _single_iteration_random_selection(
                        nb_attributes, nb_targets, fraction_missing, potential_targets
                    )
                    codes.append(m_codes)

        m_codes = np.vstack(codes)

        m_codes = _ensure_desc_atts(m_codes)
        return m_codes


def _single_iteration_random_selection(
    nb_attributes, nb_targets, fraction_missing, potential_targets
):
    nb_models, deficit = _nb_models_and_deficit(nb_targets, potential_targets)

    # Init
    m_codes = _init(nb_models, nb_attributes)

    target_sets = _target_sets(potential_targets, nb_targets, nb_models, deficit)

    m_codes = _set_targets(m_codes, target_sets)
    m_codes = _set_missing(m_codes, fraction_missing)
    m_codes.astype(int)
    return m_codes


# Helpers
def _set_missing(m_codes, fraction=0.2):
    random = np.random.rand(*m_codes.shape)

    noise = np.where(m_codes == 0, random, m_codes)
    missing = np.where(noise < fraction, -1, noise)

    res = np.floor(missing)

    return res


def _ensure_desc_atts(m_codes):
    """
    If there are no input attributes in a code, we flip one missing attribute at random.
    """
    for row in m_codes:
        if 0 not in np.unique(row):
            idx_of_minus_ones = np.where(row == -1)[0]
            idx_to_change_to_zero = np.random.choice(idx_of_minus_ones)
            row[idx_to_change_to_zero] = 0

    return m_codes


def _set_nb_targets(nb_targets, nb_atts):
    if (nb_targets > 0) & (nb_targets < 1):
        nb_out_atts = int(np.ceil(nb_targets * nb_atts))
    elif (nb_targets >= 1) & (nb_targets < nb_atts):
        nb_out_atts = int(nb_targets)
    else:
        msg = """
        Impossible number of output attributes per model: {}
        This means the value of `nb_targets` was set incorrectly.
        Re-adjusted to default=1; one model per attribute.
        """.format(
            nb_targets
        )
        warnings.warn(msg)
        nb_out_atts = 1

    return nb_out_atts


def _nb_models_and_deficit(nb_targets, potential_targets):

    nb_potential_targets = potential_targets.shape[0]

    nb_models_with_regular_nb_targets = nb_potential_targets // nb_targets
    nb_leftover_targets = nb_potential_targets % nb_targets

    if nb_leftover_targets:
        nb_models = nb_models_with_regular_nb_targets + 1
        deficit = nb_targets - nb_leftover_targets
    else:
        nb_models = nb_models_with_regular_nb_targets
        deficit = 0

    return nb_models, deficit


def _init(nb_models, nb_attributes):
    return np.zeros((nb_models, nb_attributes), dtype=int)


def _target_sets(potential_targets, nb_targets, nb_models, deficit):
    np.random.shuffle(potential_targets)
    choices = np.r_[potential_targets, potential_targets[:deficit]]

    return np.random.choice(choices, replace=False, size=(nb_models, nb_targets))


def _set_targets(m_codes, target_sets):

    row_idx = np.arange(m_codes.shape[0]).reshape(-1, 1)
    col_idx = target_sets

    m_codes[row_idx, col_idx] = TARG_ENCODING
    return m_codes

