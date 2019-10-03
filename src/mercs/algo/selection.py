import numpy as np
import warnings


def base_selection_algorithm(metadata, nb_targets=1, nb_iterations=1, random_state=997):
    """
    The easiest selection algorithm.
    """
    np.random.seed(random_state)
    n_attributes = metadata["n_attributes"]

    nb_targets = _set_nb_out_params_(nb_targets, n_attributes)  # Consistency check

    att_idx = np.array(range(n_attributes))
    result = np.zeros((1, n_attributes))

    for it_idx in range(nb_iterations):
        codes = _create_init(n_attributes, nb_targets)

        # Shuffle the results
        np.random.shuffle(att_idx)
        codes = codes[:, att_idx]

        result = np.concatenate((result, codes))

    return result[1:, :]


def random_selection_algorithm(
    metadata,
    nb_targets=1,
    nb_iterations=1,
    fraction_missing=0.2,
    nb_desc=None,
    random_state=997,
):
    if isinstance(fraction_missing, list):
        res = []
        for fm in fraction_missing:
            res.append(random_selection_algorithm(
                metadata,
                nb_targets=nb_targets,
                nb_iterations=nb_iterations,
                fraction_missing=fm,
                nb_desc=nb_desc,
                random_state=random_state,
            ))
        result = np.r_[res]
        return np.concatenate(res, axis=0)
    else:
        np.random.seed(random_state)
        n_attributes = metadata["n_attributes"]
        nb_targets = _set_nb_out_params_(nb_targets, n_attributes)

        if nb_desc is not None:
            fraction_missing = (n_attributes - nb_desc) / n_attributes

        att_idx = np.array(range(n_attributes))
        result = np.zeros((1, n_attributes))

        for it_idx in range(nb_iterations):
            codes = _create_init(n_attributes, nb_targets)
            codes = _add_missing(codes, fraction=fraction_missing)
            codes = _ensure_desc_atts(codes)

            # Shuffle the results
            np.random.shuffle(att_idx)
            codes = codes[:, att_idx]

            result = np.concatenate((result, codes))

        return result[1:, :]


# Helpers
def _create_init(nb_atts, nb_tgt):
    res = np.zeros((nb_atts, nb_atts))
    for k in range(nb_tgt):
        res += np.eye(nb_atts, k=k)

    return res[0::nb_tgt, :]


def _add_missing(init, fraction=0.2):
    random = np.random.rand(*init.shape)

    noise = np.where(init == 0, random, init)
    missing = np.where(noise < fraction, -1, noise)

    res = np.floor(missing)

    res = _ensure_desc_atts(res)
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


def _set_nb_out_params_(nb_targets, nb_atts):
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
