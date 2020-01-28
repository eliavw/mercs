import numpy as np

def get_i_o(data, desc_ids, targ_ids, filter_nan=True):
    if filter_nan:
        i_o = data[:, desc_ids+targ_ids]

        mask = ~np.any(np.isnan(i_o), axis=1)
        i_o_filtered = i_o[mask]

        # This works because we ordered the columns very specifically.
        i = i_o_filtered[:, :len(desc_ids)]
        o = i_o_filtered[:, len(desc_ids):]

    else:
        i, o = data[:, desc_ids], data[:, targ_ids]
    return i, o