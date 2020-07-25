import itertools
import warnings
from inspect import signature
from timeit import default_timer

import numpy as np
from networkx import NetworkXUnfeasible, find_cycle
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from mercs.algo import (
    imputation,
    induction,
    inference_legacy,
    inference,
    selection,
    prediction,
    evaluation,
)
from mercs.composition import CompositeModel, o
from mercs.graph import build_diagram
from mercs.utils import (
    TARG_ENCODING,
    code_to_query,
    query_to_code,
    DecoratedDecisionTreeClassifier,
    DecoratedDecisionTreeRegressor,
    DecoratedRandomForestClassifier
)
from mercs.visuals import show_diagram

try:
    from xgboost import XGBClassifier as XGBC
    from xgboost import XGBRegressor as XGBR
except:
    XGBC, XGBR = None, None

try:
    from lightgbm import LGBMClassifier as LGBMC
    from lightgbm import LGBMRegressor as LGBMR
except:
    LGBMC, LGBMR = None, None

try:
    from catboost import CatBoostClassifier as CBC
    from catboost import CatBoostRegressor as CBR
except:
    CBC, CBR = None, None

try:
    from wekalearn import RandomForestClassifier as WLC
    from wekalearn import RandomForestRegressor as WLR
except:
    WLC, WLR = None, None

try:
    from morfist import MixedRandomForest as MRF
except:
    MRF = None


class Mercs(object):
    delimiter = "_"

    selection_algorithms = dict(
        default=selection.base_selection_algorithm,
        base=selection.base_selection_algorithm,
        random=selection.random_selection_algorithm,
    )

    induction_algorithms = dict(
        base=induction.base_induction_algorithm,
        default=induction.base_induction_algorithm,
        expand=induction.expand_induction_algorithm,
    )

    classifier_algorithms = dict(
        DT=DecisionTreeClassifier,
        DDT=DecoratedDecisionTreeClassifier,
        RF=RandomForestClassifier,
        DRF=DecoratedRandomForestClassifier,
        XGB=XGBC,
        xgb=XGBC,
        weka=WLC,
        LGBM=LGBMC,
        lgbm=LGBMC,
        CB=CBC,
        extra=ExtraTreesClassifier,
    )

    regressor_algorithms = dict(
        DT=DecisionTreeRegressor,
        DDT=DecoratedDecisionTreeRegressor,
        RF=RandomForestRegressor,
        DRF=DecoratedDecisionTreeRegressor,
        XGB=XGBR,
        xgb=XGBR,
        weka=WLR,
        LGBM=LGBMR,
        lgbm=LGBMR,
        CB=CBR,
        extra=ExtraTreesRegressor,
    )

    prediction_algorithms = dict(
        mi=prediction.mi,
        mrai=prediction.mrai,
        it=prediction.it,
        random_walk=prediction.random_walk,
        default=prediction.it
    )

    inference_algorithms = dict(
        base=inference.inference_algorithm,
        dask=inference_legacy.dask_inference_algorithm,
        legacy=inference_legacy.base_inference_algorithm,
    )

    imputer_algorithms = dict(
        nan=imputation.nan_imputation,
        NAN=imputation.nan_imputation,
        NaN=imputation.nan_imputation,
        null=imputation.nan_imputation,
        NULL=imputation.nan_imputation,
        skl=imputation.skl_imputation,
        base=imputation.skl_imputation,
        default=imputation.skl_imputation,
    )

    evaluation_algorithms = dict(
        base=evaluation.base_evaluation,
        default=evaluation.base_evaluation,
        dummy=evaluation.dummy_evaluation,
    )

    mixed_algorithms = dict(
        morfist=MRF,
        default=MRF,
    )

    # Used in parse kwargs to identify parameters. If this identification goes wrong, you are sending settings
    # somewhere you do not want them to be. So, this is a tricky part, and moreover hardcoded. In other words:
    # this is risky terrain, and should probably be done differently in the future.
    configuration_prefixes = dict(
        imputation={"imputation", "imp"},
        induction={"induction", "ind"},
        selection={"selection", "sel"},
        prediction={"prediction", "pred", "prd"},
        inference={"inference", "infr", "inf"},
        classification={"classification", "classifier", "clf"},
        regression={"regression", "regressor", "rgr"},
        mixed={"mixed"},
        metadata={"metadata", "meta", "mtd"},
        evaluation={"evaluation", "evl"},
    )

    def __init__(
            self,
            selection_algorithm="base",
            induction_algorithm="base",
            classifier_algorithm="DT",
            regressor_algorithm="DT",
            prediction_algorithm="mi",
            inference_algorithm="base",
            imputer_algorithm="default",
            evaluation_algorithm="default",
            mixed_algorithm=None,
            random_state=42,
            **kwargs
    ):
        # Explicit parameters
        self.params = dict(
            selection_algorithm=selection_algorithm,
            induction_algorithm=induction_algorithm,
            classifier_algorithm=classifier_algorithm,
            regressor_algorithm=regressor_algorithm,
            prediction_algorithm=prediction_algorithm,
            inference_algorithm=inference_algorithm,
            imputer_algorithm=imputer_algorithm,
            evaluation_algorithm=evaluation_algorithm,
            mixed_algorithm=mixed_algorithm,
            random_state=random_state
        )

        # Add MixedRandomForest configuration if the algorithm has been selected
        # As opposed to the rest of algorithms, this one is optional, so it has to be initialised in a different way
        if mixed_algorithm in self.mixed_algorithms:
            self.mixed_algorithm = self.mixed_algorithms[mixed_algorithm]
        elif mixed_algorithm is not None:
            print("Unknown mixed algorithm")
            exit(-1)
        else:
            self.mixed_algorithm = None

        # For some reason, some parameters are expected to be passed as kwargs, so we aggregate them here with the
        # explicitly-passed parameters
        self.params = {**self.params, **kwargs}
        self.random_state = random_state
        # Parse some parameters from the list of possible values
        #   N.b.: For some parameters, first try to look up the key.
        #   If the key is not found, we assume the algorithm itself was passed.
        self.selection_algorithm = self.selection_algorithms[selection_algorithm]
        self.classifier_algorithm = self.classifier_algorithms.get(classifier_algorithm, classifier_algorithm)
        self.regressor_algorithm = self.regressor_algorithms.get(regressor_algorithm, regressor_algorithm)
        self.prediction_algorithm = self.prediction_algorithms[prediction_algorithm]
        self.inference_algorithm = self.inference_algorithms[inference_algorithm]
        self.induction_algorithm = self.induction_algorithms[induction_algorithm]  # For now, we only have one.
        self.imputer_algorithm = self.imputer_algorithms[imputer_algorithm]
        self.evaluation_algorithm = self.evaluation_algorithms[evaluation_algorithm]

        # Global variables initialization
        self.m_codes = np.array([])
        self.m_list = []
        self.c_list = []
        self.g_list = []
        self.i_list = []

        # Feature importances of each model
        self.m_feature_importances = np.array([])
        # Model score
        self.m_score = np.array([])

        self.FI = np.array([])
        self.targ_ids = np.array([])

        # Query-related things
        self.q_code = None
        self.q_desc_ids = None
        self.q_targ_ids = None
        self.q_diagram = None
        self.q_compose = None
        self.q_methods = []
        self.q_model = None

        self.c_sel = []
        self.c_diagram = []

        self.metadata = dict()
        self.model_data = dict()

        # Configurations for each algorithm
        self.imputer_config = self._default_config(self.imputer_algorithm)
        self.induction_config = self._default_config(self.induction_algorithm)
        self.selection_config = self._default_config(self.selection_algorithm)
        self.classifier_config = self._default_config(self.classifier_algorithm)
        self.regressor_config = self._default_config(self.regressor_algorithm)
        self.prediction_config = self._default_config(self.prediction_algorithm)
        self.inference_config = self._default_config(self.inference_algorithm)
        self.evaluation_config = self._default_config(self.evaluation_algorithm)

        # Aggregate all configurations
        self.configuration = dict(
            imputation=self.imputer_config,
            induction=self.induction_config,
            selection=self.selection_config,
            classification=self.classifier_config,
            regression=self.regressor_config,
            prediction=self.prediction_config,
            inference=self.inference_config,
        )

        # Add MixedRandomForest configuration if the algorithm has been selected
        self.mixed_config = None
        if self.mixed_algorithm:
            self.mixed_config = self._default_config(self.mixed_algorithm)
            self.configuration["mixed"] = self.mixed_config

        # Update config based on random_state and kwargs
        self._update_config(random_state=random_state, **kwargs)
        self._extra_checks_on_config()

    def fit(self, X, y=None, m_codes=None, **kwargs):
        """
        Fits n models for the given dataset, based on its metadata(n of models, desc and targ attributes, etc.)
        The models are fitted by the induction algorithm.
        The m_codes are generated by the selection algorithm.
        Missing data is handled by the imputer algorithm.
        A score is calculated by the evaluation algorithm.

        Args:
            X: dataset
            y: labels(optional). They will be used as another column of X.
            m_codes: model codes(optional). If not given, they will be randomly generated.
            **kwargs: metadata

        """
        assert isinstance(X, np.ndarray)

        # If labels are provided, they are added to the X data as a new column at the end
        # MERCS does not care about labels and will treat them as a new variable
        if y is not None:
            assert isinstance(y, np.ndarray)
            X = np.c_[X, y]
        tic = default_timer()

        self.metadata = self._default_metadata(X)
        self._update_metadata(**kwargs)

        self.i_list = self.imputer_algorithm(X, self.metadata.get("nominal_attributes"))

        # N.b.: 'random state' parameter is in 'self.sel_config'
        if m_codes is None:
            generate_mixed_codes = True if self.mixed_algorithm else False
            self.m_codes = self.selection_algorithm(self.metadata, generate_mixed_codes, **self.selection_config)
        else:
            self.m_codes = m_codes

        self.m_list = self.induction_algorithm(
            X,
            self.m_codes,
            self.metadata,
            self.classifier_algorithm,
            self.regressor_algorithm,
            self.mixed_algorithm,
            self.classifier_config,
            self.regressor_config,
            self.mixed_config,
            **self.induction_config
        )

        self._filter_m_list_m_codes()

        self._consistent_data_structures()

        if self.imputer_algorithm == self.imputer_algorithms.get("nan"):
            # If you do no have imputers, you cannot use them as a baseline evaluation
            self.evaluation_config["consider_imputations"] = False

        self.m_score = self.evaluation_algorithm(
            X, self.m_codes, self.m_list, self.i_list, **self.evaluation_config
        )

        toc = default_timer()
        self.model_data["ind_time"] = toc - tic
        self.metadata["n_component_models"] = len(self.m_codes)

    def predict(
            self,
            X,
            q_code=None,
            inference_algorithm=None,
            prediction_algorithm=None,
            **kwargs
    ):
        """

        Args:
            X: dataset.
            q_code: query code(indicates descriptive and target attributes).
                A default code will be used if not provided.
            inference_algorithm: the chosen inference algorithm. Adds data to the graph.
            prediction_algorithm: the chosen prediction algorithm. Indicates the prediction strategy.
                Used to determine which models should be used for prediction.
            **kwargs: prediction algorithm metadata

        Returns: the prediction

        """
        # Update configuration if necessary
        if q_code is None:
            q_code = self._default_q_code()

        if inference_algorithm is not None:
            self._reconfig_inference(inference_algorithm=inference_algorithm)

        if prediction_algorithm is not None:
            self._reconfig_prediction(
                prediction_algorithm=prediction_algorithm, **kwargs
            )

        # Adjust data
        self.q_code = q_code
        self.q_desc_ids, self.q_targ_ids, _ = code_to_query(
            self.q_code, return_list=True
        )

        # Make query-diagram
        tic_prediction = default_timer()
        m_selection = self.prediction_algorithm(
            self.m_codes, self.m_feature_importances, self.m_score, q_code=self.q_code, **self.prediction_config
        )
        toc_prediction = default_timer()

        tic_diagram = default_timer()
        # Builds empty diagram with the selected models' structure
        self.q_diagram = self._build_q_diagram(self.m_list, m_selection)
        toc_diagram = default_timer()

        tic_inference = default_timer()
        # Fills the diagram nodes with the needed data and methods
        if isinstance(self.q_diagram, tuple):
            q_diagrams = self.q_diagram

            self.c_list = [self._build_q_model(X, d) for d in q_diagrams]
            self.c_sel = list(range(len(self.c_list)))
            self.c_diagram = self._build_q_diagram(
                self.c_list, self.c_sel, composition=True
            )

            self.q_model = self._build_q_model(X, self.c_diagram)
        else:
            self.q_model = self._build_q_model(X, self.q_diagram)
        toc_inference = default_timer()

        tic_dask = default_timer()
        X = X[:, self.q_model.desc_ids]
        result = self.q_model.predict(X)
        toc_dask = default_timer()

        self.model_data["prd_time"] = toc_prediction - tic_prediction
        self.model_data["dia_time"] = toc_diagram - tic_diagram
        self.model_data["inference_time"] = toc_inference - tic_inference
        self.model_data["dsk_time"] = toc_dask - tic_dask
        self.model_data["inf_time"] = toc_dask - tic_prediction

        return result

    def show_q_diagram(self, kind="svg", fi=False, ortho=False, index=None, **kwargs):
        if isinstance(self.q_diagram, tuple) and index is None:
            return show_diagram(self.c_diagram, kind=kind, fi=fi, ortho=ortho, **kwargs)
        elif isinstance(self.q_diagram, tuple):
            return show_diagram(
                self.q_diagram[index], kind=kind, fi=fi, ortho=ortho, **kwargs
            )
        else:
            return show_diagram(self.q_diagram, kind=kind, fi=fi, ortho=ortho, **kwargs)

    # Diagrams
    def _build_q_diagram(self, m_list, m_selection, composition=False):
        if isinstance(m_selection, tuple):
            diagrams = [
                build_diagram(
                    m_list,
                    m_selection_instance,
                    self.q_code,
                    prune=True,
                    composition=composition,
                )
                for m_selection_instance in m_selection
            ]
            return tuple(diagrams)
        else:
            return build_diagram(
                m_list, m_selection, self.q_code, prune=True, composition=composition
            )

    # Inference
    def _build_q_model(self, X, diagram):
        try:
            self.inference_algorithm(
                diagram,
                self.m_list,
                self.i_list,
                self.c_list,
                X,
                self.metadata.get("nominal_attributes"),
            )
        except NetworkXUnfeasible:
            cycle = find_cycle(self.q_diagram, orientation="original")
            msg = """
            Topological sort failed, investigate diagram to debug.
            
            I will never be able to squeeze a prediction out of a diagram with a loop.
            
            Cycle was:  {}
            """.format(
                cycle
            )
            raise RecursionError(msg)

        n_component_models = self.metadata["n_component_models"]
        q_model = CompositeModel(
            diagram,
            nominal_attributes=self.metadata["nominal_attributes"],
            n_component_models=n_component_models,
        )
        return q_model

    def _filter_m_list_m_codes(self):
        """Filtering out the failed models.

        This happens when TODO: EXPLAIN
        """
        fail_m_idxs = [i for i, m in enumerate(self.m_list) if m is None]
        self.m_codes = np.delete(self.m_codes, fail_m_idxs, axis=0)
        self.m_list = [m for m in self.m_list if m is not None]

    # Graphs
    def _consistent_data_structures(self):
        self._update_m_codes()
        self._update_m_feature_importances()

    def _expand_m_list(self):
        self.m_list = list(itertools.chain.from_iterable(self.m_list))

    def _update_m_codes(self):
        self.m_codes = np.array(
            [
                query_to_code(
                    list(model.desc_ids),
                    list(model.targ_ids),
                    attributes=self.metadata["attributes"],
                )
                for model in self.m_list
            ]
        )

    def _update_m_feature_importances(self):

        init = np.zeros(self.m_codes.shape)

        for m_idx, mod in enumerate(self.m_list):
            init[m_idx, list(mod.desc_ids)] = mod.feature_importances_

        self.m_feature_importances = init

    # Imputer
    def _add_imputer_function(self, g):

        for n in g.nodes:
            if g.nodes[n]["kind"] == "imputation":
                idx = g.nodes[n]["idx"]

                f_1 = self._dummy_array  # Artificial input
                f_2 = self.i_list[idx].transform  # Actual imputation
                f_3 = np.ravel  # Return a vector, not array

                g.nodes[n]["function"] = o(f_3, o(f_2, f_1))

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

    # Configuration
    def _reconfig_prediction(self, prediction_algorithm="mi", **kwargs):
        self.prediction_algorithm = self.prediction_algorithms[prediction_algorithm]
        self.prediction_config = self._default_config(self.prediction_algorithm)

        self.configuration["prediction"] = self.prediction_config
        self._update_config(**kwargs)

    def _reconfig_inference(self, inference_algorithm="base", **kwargs):
        self.inference_algorithm = self.inference_algorithms[inference_algorithm]

        self.inf_config = self._default_config(self.inference_algorithm)

        self.configuration["inference"] = self.inf_config
        self._update_config(**kwargs)

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

    def _extra_checks_on_config(self):
        self._check_xgb_single_target()

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
        is_nominal = t == np.dtype(int)
        return is_nominal

    @staticmethod
    def _is_numeric(t):
        is_numeric = t == np.dtype(float)
        return is_numeric

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
