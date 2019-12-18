from sklearn.impute import SimpleImputer
import numpy as np


def nan_imputation(X, nominal_attributes):
    # Init
    n_rows, n_cols = X.shape
    i_list = []

    # Make imputers
    for c in range(n_cols):
        i_config = dict(missing_values=np.nan, strategy="constant", fill_value=np.nan)

        # Initialize imputer
        i = SimpleImputer(**i_config)
        i.fit(X[:, [c]])

        if c in nominal_attributes:
            i.out_kind = "nominal"
        else:
            i.out_kind = "numeric"

        i_list.append(i)
    return i_list


def skl_imputation(X, nominal_attributes):

    # Init
    n_rows, n_cols = X.shape
    i_list = []

    # Make imputers
    for c in range(n_cols):

        # Generate config
        i_config = dict(missing_values=np.nan)
        if c in nominal_attributes:
            # Nominal attribute
            out_kind = "nominal"
            i_config["strategy"] = "most_frequent"
        else:
            # Numeric attribute
            i_config["strategy"] = "mean"
            out_kind = "numeric"

        # Initialize imputer
        i = SimpleImputer(**i_config)
        i.fit(X[:, [c]])
        i.out_kind = out_kind
        i_list.append(i)

    return i_list
