import warnings
from inspect import signature
from timeit import default_timer

import numpy as np
from dask import delayed
from networkx import NetworkXUnfeasible, find_cycle, topological_sort
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from ..algo import inference, prediction, selection, new_prediction
from ..algo.induction import base_induction_algorithm
from ..composition import CompositeModel, o, x
from ..graph import model_to_graph, get_targ, compose_all
from ..utils import DESC_ENCODING, TARG_ENCODING, MISS_ENCODING
from ..visuals import save_diagram, show_diagram


class Mercs(object):
    # TODO: Determine these automatically based on method names.
    delimiter = "_"

    selection_algorithms = {
        "base": selection.base_selection_algorithm,
        "random": selection.random_selection_algorithm,
    }

    classifier_algorithms = {
        "DT": DecisionTreeClassifier,
        "RF": RandomForestClassifier,
        "XGB": XGBClassifier,
    }

    regressor_algorithms = {
        "DT": DecisionTreeRegressor,
        "RF": RandomForestRegressor,
        "XGB": XGBRegressor,
    }

    prediction_algorithms = {
        "mi": new_prediction.mi,
        "mrai": new_prediction.mrai,
        "it": new_prediction.it,
        "rw": new_prediction.rw,
    }

    inference_algorithms = {
        "base": inference.base_inference_algorithm,
        "dask": inference.dask_inference_algorithm,
    }

    # Used in parse kwargs to identify parameters. If this identification goes wrong, you are sending settings
    # somewhere you do not want them to be. So, this is a tricky part, and moreover hardcoded. In other words:
    # this is risky terrain, and should probably be done differently in the future.
    configuration_prefixes = {
        "selection": ["selection", "sel"],
        "prediction": ["prediction", "pred", "prd"],
        "inference": ["inference", "infr", "inf"],
        "classification": ["classification", "classifier", "clf"],
        "regression": ["regression", "regressor", "rgr"],
        "metadata": ["metadata", "meta", "mtd"],
    }

    def __init__(
        self,
        selection_algorithm="base",
        classifier_algorithm="DT",
        regressor_algorithm="DT",
        prediction_algorithm="mi",
        inference_algorithm="dask",
        random_state=997,
        **kwargs
    ):
        self.random_state = random_state
        self.selection_algorithm = self.selection_algorithms[selection_algorithm]

        # N.b.: First try to look up the key. If the key is not found, we assume the algorithm itself was passed.
        self.classifier_algorithm = self.classifier_algorithms.get(
            classifier_algorithm, classifier_algorithm
        )
        self.regressor_algorithm = self.regressor_algorithms.get(
            regressor_algorithm, regressor_algorithm
        )

        self.prediction_algorithm = self.prediction_algorithms[prediction_algorithm]
        self.inference_algorithm = self.inference_algorithms[inference_algorithm]
        self.induction_algorithm = (
            base_induction_algorithm
        )  # For now, we only have one.

        # Data-structures
        self.m_codes = np.array([])
        self.m_list = []
        self.g_list = []
        self.i_list = []

        self.FI = np.array([])
        self.targ_ids = np.array([])

        # Query-related things
        self.q_code = None
        self.q_desc_ids = None
        self.q_targ_ids = None
        self.q_diagram = None
        self.q_compose = None
        self.q_methods = []

        # Configurations
        self.sel_cfg = self._default_config(self.selection_algorithm)
        self.clf_cfg = self._default_config(self.classifier_algorithm)
        self.rgr_cfg = self._default_config(self.regressor_algorithm)
        self.prd_cfg = self._default_config(self.prediction_algorithm)
        self.inf_cfg = self._default_config(self.inference_algorithm)

        self.configuration = dict(
            selection=self.sel_cfg,
            classification=self.clf_cfg,
            regression=self.rgr_cfg,
            prediction=self.prd_cfg,
            inference=self.inf_cfg,
        )  # Collect all configs in one

        self._update_config(random_state=random_state, **kwargs)

        self.metadata = dict()
        self.model_data = dict()

        self._extra_checks_on_config()

        return

    def fit(self, X, **kwargs):
        """

        Parameters
        ----------
        X:      np.ndarray,
                training data.
        kwargs

        Returns
        -------

        """

        assert isinstance(X, np.ndarray)
        tic = default_timer()

        self.metadata = self._default_metadata(X)
        self._update_metadata(**kwargs)

        self._fit_imputer(X)

        # N.b.: `random state` parameter is in `self.sel_cfg`
        self.m_codes = self.selection_algorithm(self.metadata, **self.sel_cfg)

        self.m_list = self.induction_algorithm(
            X,
            self.m_codes,
            self.metadata,
            classifier=self.classifier_algorithm,
            regressor=self.regressor_algorithm,
            classifier_kwargs=self.clf_cfg,
            regressor_kwargs=self.rgr_cfg,
            random_state=self.configuration.get("random_state", 997),
        )

        self._update_g_list()
        self._update_fi()
        self._update_t_codes()

        toc = default_timer()
        self.model_data["ind_time"] = toc - tic

        return

    def predict(self, X, q_code=None, prediction_algorithm=None, reuse=True, **kwargs):
        if q_code is None:
            q_code = self._default_q_code()

        if not np.array_equal(self.q_code, q_code):
            reuse = False

        if prediction_algorithm is not None:
            reuse = False
            self._reconfig_prediction(
                prediction_algorithm=prediction_algorithm, **kwargs
            )

        tic = default_timer()
        if not reuse:
            # Adjust data
            self.q_code = q_code
            self.q_desc_ids = np.where(q_code == DESC_ENCODING)[0].tolist()
            self.q_targ_ids = np.where(q_code == TARG_ENCODING)[0].tolist()

            # Make custom diagram
            self.q_diagram = self.prediction_algorithm(
                self.g_list, q_code, self.fi, self.t_codes, **self.prd_cfg
            )

            if isinstance(self.q_diagram, list):
                self.q_diagrams = self.q_diagram
                self.q_models = [self._get_q_model(d, X) for d in self.q_diagram]
                self.q_diagram, self.q_model = self.merge_models(self.q_models)
            else:
                self.q_model = self._get_q_model(self.q_diagram, X)

        res = self.q_model.predict.compute()
        toc = default_timer()
        self.model_data["inf_time"] = toc - tic
        return res

    # Diagrams
    def show_q_diagram(self, kind="svg", fi=False, ortho=False, **kwargs):
        return show_diagram(self.q_diagram, kind=kind, fi=fi, ortho=ortho, **kwargs)

    def save_diagram(self, fname=None, kind="svg", fi=False, ortho=False):
        return save_diagram(self.q_diagram, fname, kind=kind, fi=fi, ortho=ortho)

    # Inference
    def merge_models(self, q_models):

        types = self._get_types(self.metadata)

        walks = [
            model_to_graph(m, types, idx=idx, composition=True)
            for idx, m in enumerate(q_models)
        ]
        q_diagram = compose_all(walks)
        filtered_nodes = self.filter_nodes(q_diagram)

        try:
            self.inference_algorithm(q_diagram, sorted_nodes=filtered_nodes)
        except NetworkXUnfeasible:
            cycle = find_cycle(q_diagram, orientation="original")
            msg = """
            Topological sort failed, investigate diagram to debug.
            
            I will never be able to squeeze a prediction out of a diagram with a loop.
            
            Cycle was:  {}
            """.format(
                cycle
            )
            raise RecursionError(msg)

        q_model = CompositeModel(q_diagram)
        return q_diagram, q_model

    def _get_q_model(self, q_diagram, X):

        self._add_imputer_function(q_diagram)

        try:
            self.inference_algorithm(q_diagram, X=X)
        except NetworkXUnfeasible:
            cycle = find_cycle(q_diagram, orientation="original")
            msg = """
            Topological sort failed, investigate diagram to debug.
            
            I will never be able to squeeze a prediction out of a diagram with a loop.
            
            Cycle was:  {}
            """.format(
                cycle
            )
            raise RecursionError(msg)

        q_model = CompositeModel(q_diagram)
        return q_model

    # Graphs
    def _update_g_list(self):
        types = self._get_types(self.metadata)
        self.g_list = [
            model_to_graph(m, types=types, idx=idx) for idx, m in enumerate(self.m_list)
        ]
        return

    def _update_fi(self):

        self.fi = np.zeros(self.m_codes.shape)

        for m_idx, mod in enumerate(self.m_list):

            self.fi[m_idx, list(mod.desc_ids)] = mod.feature_importances_

        return

    def _update_t_codes(self):
        self.t_codes = (self.m_codes == TARG_ENCODING).astype(int)
        return

    # Imputer
    def _fit_imputer(self, X):
        """
        Construct and fit an imputer
        """
        _, n_cols = X.shape

        i_list = []
        for c in range(n_cols):
            if c in self.metadata["nominal_attributes"]:
                current_strategy = "most_frequent"
            else:
                current_strategy = "mean"

            i = SimpleImputer(missing_values=np.nan, strategy=current_strategy)
            i.fit(X[:, [c]])
            i_list.append(i)

        self.i_list = i_list

        return

    def _add_imputer_function(self, g):

        for n in g.nodes:
            if g.nodes[n]["kind"] == "imputation":
                idx = g.nodes[n]["idx"]

                f_1 = self._dummy_array  # Artificial input
                f_2 = self.i_list[idx].transform  # Actual imputation
                f_3 = np.ravel  # Return a vector, not array

                g.nodes[n]["function"] = o(f_3, o(f_2, f_1))

        return

    # Add ids
    @staticmethod
    def _add_ids(g, desc_ids, targ_ids):
        g.graph["desc_ids"] = set(desc_ids)
        g.graph["targ_ids"] = set(targ_ids)
        return g

    # Metadata
    def _default_metadata(self, X):
        if X.ndim != 2:
            X = X.reshape(-1, 1)

        n_rows, n_cols = X.shape

        types = [X[0, 0].dtype for _ in range(n_cols)]
        nominal_attributes = set(
            [att for att, typ in enumerate(types) if self._is_nominal(typ)]
        )
        numeric_attributes = set(
            [att for att, typ in enumerate(types) if self._is_numeric(typ)]
        )

        metadata = dict(
            attributes=set(range(n_cols)),
            n_attributes=n_cols,
            types=types,
            nominal_attributes=nominal_attributes,
            numeric_attributes=numeric_attributes,
        )
        return metadata

    def _update_metadata(self, **kwargs):

        self._update_dictionary(self.metadata, kind="metadata", **kwargs)

        # Assure every attribute is `typed`: If not every attribute is here, set to numeric type (default)
        numeric = self.metadata["numeric_attributes"]
        nominal = self.metadata["nominal_attributes"]
        att_ids = self.metadata["attributes"]

        if len(nominal) + len(numeric) != len(att_ids):
            numeric = att_ids - nominal
            self._update_dictionary(
                self.metadata, kind="metadata", numeric_attributes=numeric
            )

        return

    # Configuration
    def _reconfig_prediction(self, prediction_algorithm="mi", **kwargs):
        self.prediction_algorithm = self.prediction_algorithms[prediction_algorithm]
        self.prd_cfg = self._default_config(self.prediction_algorithm)

        self.configuration["prediction"] = self.prd_cfg
        self._update_config(**kwargs)

        return

    @staticmethod
    def _default_config(method):
        config = {}
        sgn = signature(method)

        for key, parameter in sgn.parameters.items():
            if parameter.default is not parameter.empty:
                config[key] = parameter.default
        return config

    def _update_config(self, **kwargs):

        for kind in self.configuration:
            self._update_dictionary(self.configuration[kind], kind=kind, **kwargs)

        return

    def _extra_checks_on_config(self):

        self._check_xgb_single_target()
        return

    def _check_xgb_single_target(self):

        nb_targets = self.configuration["selection"]["nb_targets"]

        if nb_targets == 1:
            return None
        else:
            if (
                self.classifier_algorithm is self.classifier_algorithms["XGB"]
                or self.regressor_algorithm is self.regressor_algorithms["XGB"]
            ):
                xgb = True
            else:
                xgb = False

            if xgb:
                msg = """
                XGBoost cannot deal with multi-target outputs.

                Hence, the `nb_targets` parameter is automatically adapted to 1,
                so only single-target trees will be learned. 

                Please take this into account.
                """
                warnings.warn(msg)
                self.configuration["selection"]["nb_targets"] = 1

            return

    def _parse_kwargs(self, kind="selection", **kwargs):

        prefixes = [e + self.delimiter for e in self.configuration_prefixes[kind]]

        parameter_map = {
            x.split(prefix)[1]: x
            for x in kwargs
            for prefix in prefixes
            if x.startswith(prefix)
        }

        return parameter_map

    def _update_dictionary(self, dictionary, kind=None, **kwargs):
        # Immediate matches
        overlap = set(dictionary).intersection(set(kwargs))

        for k in overlap:
            dictionary[k] = kwargs[k]

        if kind is not None:
            # Parsed matches
            parameter_map = self._parse_kwargs(kind=kind, **kwargs)
            overlap = set(dictionary).intersection(set(parameter_map))

            for k in overlap:
                dictionary[k] = kwargs[parameter_map[k]]
        return

    # Helpers
    def _filter_X(self, X):
        # Filter relevant input attributes
        if X.shape[1] != len(self.q_compose.desc_ids):
            indices = self._overlapping_indices(
                self.q_desc_ids, self.q_compose.desc_ids
            )
        return X[:, indices]

    @staticmethod
    def _dummy_array(X):
        """
        Return an array of np.nan, with the same number of rows as the input array.

        Parameters
        ----------
        X:      np.ndarray(), n_rows, n_cols = X.shape,
                We use the shape of X to deduce shape of our output.

        Returns
        -------
        a:      np.ndarray(), shape= (n_rows, 1)
                n_rows is the same as the number of rows as X.

        """
        n_rows, _ = X.shape

        a = np.empty((n_rows, 1))
        a.fill(np.nan)

        return a

    def _default_q_code(self):

        q_code = np.zeros(self.metadata["n_attributes"])
        q_code[-1] = TARG_ENCODING

        return q_code

    @staticmethod
    def _is_nominal(t):
        condition_01 = t == np.dtype(int)
        return condition_01

    @staticmethod
    def _is_numeric(t):
        condition_01 = t == np.dtype(float)
        return condition_01

    @staticmethod
    def _get_types(metadata):
        nominal = {i: "nominal" for i in metadata["nominal_attributes"]}
        numeric = {i: "numeric" for i in metadata["numeric_attributes"]}
        return {**nominal, **numeric}

    @staticmethod
    def _overlapping_indices(a, b):
        """
        Given an array a and b, return the indices (in a) of elements that occur in both a and b.

        Parameters
        ----------
        a
        b

        Returns
        -------

        Examples
        --------
        a = [4,5,6]
        b = [4,6,7]

        overlapping_indices(a, b) = [0,2]

        """
        return np.nonzero(np.in1d(a, b))[0]

    @staticmethod
    def filter_nodes(g):
        # This is not as safe as it should be

        sorted_nodes = list(topological_sort(g))
        filtered_nodes = []
        for n in reversed(sorted_nodes):
            if g.nodes[n]["kind"] == "model":
                break
            filtered_nodes.append(n)
        filtered_nodes = list(reversed(filtered_nodes))
        return filtered_nodes
