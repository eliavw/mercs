import itertools
import warnings
from inspect import signature
from timeit import default_timer

from sklearn.preprocessing import normalize

import dask
import numpy as np
import shap
from dask import delayed
from networkx import NetworkXUnfeasible, find_cycle, topological_sort
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ..algo import (
    evaluation,
    imputation,
    inference,
    inference_v3,
    new_inference,
    new_prediction,
    selection,
    vector_prediction,
)
from ..algo.induction import base_induction_algorithm, expand_induction_algorithm
from ..composition import CompositeModel, NewCompositeModel, o, x
from ..graph import build_diagram, compose_all, get_targ, model_to_graph
from ..utils import (
    DESC_ENCODING,
    MISS_ENCODING,
    TARG_ENCODING,
    DecoratedDecisionTreeClassifier,
    DecoratedDecisionTreeRegressor,
    DecoratedRandomForestClassifier,
    DecoratedRandomForestRegressor,
    code_to_query,
    get_i_o,
    query_to_code,
)
from ..visuals import save_diagram, show_diagram

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


class Mercs(object):
    delimiter = "_"

    selection_algorithms = dict(
        default=selection.base_selection_algorithm,
        base=selection.base_selection_algorithm,
        random=selection.random_selection_algorithm,
    )

    induction_algorithms = dict(
        base=base_induction_algorithm,
        default=base_induction_algorithm,
        expand=expand_induction_algorithm,
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
        mi=vector_prediction.mi,
        mrai=vector_prediction.mrai,
        it=vector_prediction.it,
        rw=vector_prediction.rw,
    )

    inference_algorithms = dict(
        base=inference.base_inference_algorithm,
        dask=inference_v3.inference_algorithm,
        own=inference_v3.inference_algorithm,
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
        inference_algorithm="own",
        imputer_algorithm="default",
        evaluation_algorithm="default",
        random_state=42,
        **kwargs
    ):
        self.params = dict(
            selection_algorithm=selection_algorithm,
            induction_algorithm=induction_algorithm,
            classifier_algorithm=classifier_algorithm,
            regressor_algorithm=regressor_algorithm,
            prediction_algorithm=prediction_algorithm,
            inference_algorithm=inference_algorithm,
            imputer_algorithm=imputer_algorithm,
            evaluation_algorithm=evaluation_algorithm,
            random_state=random_state,
        )
        self.params = {**self.params, **kwargs}

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
        self.induction_algorithm = self.induction_algorithms[
            induction_algorithm
        ]  # For now, we only have one.
        self.imputer_algorithm = self.imputer_algorithms[imputer_algorithm]
        self.evaluation_algorithm = self.evaluation_algorithms[evaluation_algorithm]

        # Data-structures
        self.m_codes = np.array([])
        self.m_list = []
        self.c_list = []
        self.g_list = []
        self.i_list = []

        self.m_fimps = np.array([])
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

        # Configurations
        self.imp_cfg = self._default_config(self.imputer_algorithm)
        self.ind_cfg = self._default_config(self.induction_algorithm)
        self.sel_cfg = self._default_config(self.selection_algorithm)
        self.clf_cfg = self._default_config(self.classifier_algorithm)
        self.rgr_cfg = self._default_config(self.regressor_algorithm)
        self.prd_cfg = self._default_config(self.prediction_algorithm)
        self.inf_cfg = self._default_config(self.inference_algorithm)
        self.evl_cfg = self._default_config(self.evaluation_algorithm)

        self.configuration = dict(
            imputation=self.imp_cfg,
            induction=self.ind_cfg,
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

    def fit(self, X, y=None, m_codes=None, **kwargs):
        assert isinstance(X, np.ndarray)

        if y is not None:
            assert isinstance(y, np.ndarray)
            X = np.c_[X, y]

        tic = default_timer()

        self.metadata = self._default_metadata(X)
        self._update_metadata(**kwargs)

        self.i_list = self.imputer_algorithm(X, self.metadata.get("nominal_attributes"))

        # N.b.: `random state` parameter is in `self.sel_cfg`
        if m_codes is None:
            self.m_codes = self.selection_algorithm(self.metadata, **self.sel_cfg)
        else:
            self.m_codes = m_codes

        self.m_list = self.induction_algorithm(
            X,
            self.m_codes,
            self.metadata,
            self.classifier_algorithm,
            self.regressor_algorithm,
            self.clf_cfg,
            self.rgr_cfg,
            **self.ind_cfg
        )

        self._filter_m_list_m_codes()

        self._consistent_datastructures()

        if self.imputer_algorithm == self.imputer_algorithms.get("nan"):
            # If you do no have imputers, you cannot use them as a baseline evaluation
            self.evl_cfg["consider_imputations"] = False

        self.m_score = self.evaluation_algorithm(
            X, self.m_codes, self.m_list, self.i_list, **self.evl_cfg
        )

        toc = default_timer()
        self.model_data["ind_time"] = toc - tic
        self.metadata["n_component_models"] = len(self.m_codes)
        return

    def predict(
        self,
        X,
        q_code=None,
        inference_algorithm=None,
        prediction_algorithm=None,
        **kwargs
    ):
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
        self.m_sel = self.prediction_algorithm(
            self.m_codes, self.m_fimps, self.m_score, q_code=self.q_code, **self.prd_cfg
        )
        toc_prediction = default_timer()

        tic_diagram = default_timer()
        self.q_diagram = self._build_q_diagram(self.m_list, self.m_sel)
        toc_diagram = default_timer()

        tic_infalgo = default_timer()
        if isinstance(self.q_diagram, tuple):
            self.q_diagrams = self.q_diagram

            # for d in self.q_diagrams:
            #    print(d.nodes)
            #   self.c_list.append(self._build_q_model(X, d))

            self.c_list = [self._build_q_model(X, d) for d in self.q_diagrams]
            self.c_sel = list(range(len(self.c_list)))
            self.c_diagram = self._build_q_diagram(
                self.c_list, self.c_sel, composition=True
            )

            self.q_model = self._build_q_model(X, self.c_diagram)
        else:
            self.q_model = self._build_q_model(X, self.q_diagram)

        toc_infalgo = default_timer()

        tic_dask = default_timer()
        X = X[:, self.q_model.desc_ids]
        result = self.q_model.predict(X)
        toc_dask = default_timer()

        self.model_data["prd_time"] = toc_prediction - tic_prediction
        self.model_data["dia_time"] = toc_diagram - tic_diagram
        self.model_data["infalgo_time"] = toc_infalgo - tic_infalgo
        self.model_data["dsk_time"] = toc_dask - tic_dask
        self.model_data["inf_time"] = toc_dask - tic_prediction

        return result

    def get_params(self, deep=False):
        return self.params

    # Diagrams
    def _build_q_diagram(self, m_list, m_sel, composition=False):
        if isinstance(m_sel, tuple):
            diagrams = [
                build_diagram(
                    m_list,
                    m_sel_instance,
                    self.q_code,
                    prune=True,
                    composition=composition,
                )
                for m_sel_instance in m_sel
            ]
            return tuple(diagrams)
        else:
            return build_diagram(
                m_list, m_sel, self.q_code, prune=True, composition=composition
            )

    def show_q_diagram(self, kind="svg", fi=False, ortho=False, index=None, **kwargs):

        if isinstance(self.q_diagram, tuple) and index is None:
            return show_diagram(self.c_diagram, kind=kind, fi=fi, ortho=ortho, **kwargs)
        elif isinstance(self.q_diagram, tuple):
            return show_diagram(
                self.q_diagram[index], kind=kind, fi=fi, ortho=ortho, **kwargs
            )
        else:
            return show_diagram(self.q_diagram, kind=kind, fi=fi, ortho=ortho, **kwargs)

    def save_diagram(self, fname=None, kind="svg", fi=False, ortho=False):
        return save_diagram(self.q_diagram, fname, kind=kind, fi=fi, ortho=ortho)

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
        q_model = NewCompositeModel(
            diagram,
            nominal_attributes=self.metadata["nominal_attributes"],
            n_component_models=n_component_models,
        )
        return q_model

    def _merge_q_models(self, q_models):
        q_diagram = build_diagram(self.c_list, self.c_sel, self.q_code, prune=True)
        return q_diagram

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

    # Filter
    def _filter_m_list_m_codes(self):
        """Filtering out the failed models.

        This happens when TODO: EXPLAIN
        """

        fail_m_idxs = [i for i, m in enumerate(self.m_list) if m is None]
        self.m_codes = np.delete(self.m_codes, fail_m_idxs, axis=0)
        self.m_list = [m for m in self.m_list if m is not None]

        return

    # Graphs
    def _consistent_datastructures(self, binary_scores=False):
        self._update_m_codes()
        self._update_m_fimps()
        return

    def _expand_m_list(self):
        self.m_list = list(itertools.chain.from_iterable(self.m_list))
        return

    def _add_model(self, model, binary_scores=False):
        self.m_list.append(model)
        self._consistent_datastructures(binary_scores=binary_scores)
        return

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
        return

    def _update_m_fimps(self):

        init = np.zeros(self.m_codes.shape)

        for m_idx, mod in enumerate(self.m_list):
            init[m_idx, list(mod.desc_ids)] = mod.feature_importances_

        self.m_fimps = init

        return

    def _update_m_score(self, binary_scores=False):
        if binary_scores:
            self.m_score = (self.m_codes == TARG_ENCODING).astype(float)
        return

    # Imputer
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

        # All attributes should be accounted for and none should be double.
        if (len(nominal) + len(numeric) - len(att_ids)) != 0:
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

    def _reconfig_inference(self, inference_algorithm="own", **kwargs):
        self.inference_algorithm = self.inference_algorithms[inference_algorithm]

        self.inf_cfg = self._default_config(self.inference_algorithm)

        self.configuration["inference"] = self.inf_cfg
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

    # SYNTH
    def autocomplete(self, X, **kwargs):
        return

    # Legacy (delete when I am sure they can go)
    def predict_old(
        self, X, q_code=None, prediction_algorithm=None, beta=False, **kwargs
    ):
        # Update configuration if necessary
        if q_code is None:
            q_code = self._default_q_code()

        if prediction_algorithm is not None:
            reuse = False
            self._reconfig_prediction(
                prediction_algorithm=prediction_algorithm, **kwargs
            )

        # Adjust data
        tic_prediction = default_timer()
        self.q_code = q_code
        self.q_desc_ids, self.q_targ_ids, _ = code_to_query(
            self.q_code, return_list=True
        )

        # Make query-diagram
        self.q_diagram = self.prediction_algorithm(
            self.g_list, q_code, self.fi, self.t_codes, **self.prd_cfg
        )

        toc_prediction = default_timer()

        tic_dask = default_timer()

        toc_dask = default_timer()

        tic_compute = default_timer()
        res = self.q_model.predict.compute()
        toc_compute = default_timer()

        # Diagnostics
        self.model_data["prd_time"] = toc_prediction - tic_prediction
        self.model_data["dsk_time"] = toc_dask - tic_dask
        self.model_data["cmp_time"] = toc_compute - tic_compute
        self.model_data["inf_time"] = toc_compute - tic_prediction
        self.model_data["ratios"] = (
            self.model_data["prd_time"] / self.model_data["inf_time"],
            self.model_data["dsk_time"] / self.model_data["inf_time"],
            self.model_data["cmp_time"] / self.model_data["inf_time"],
        )
        return res

    def _update_g_list(self):
        types = self._get_types(self.metadata)
        self.g_list = [
            model_to_graph(m, types=types, idx=idx) for idx, m in enumerate(self.m_list)
        ]
        return

    def _update_t_codes(self):
        self.t_codes = (self.m_codes == TARG_ENCODING).astype(int)
        return

    # AVATAR-TOOLS
    def avatar(
        self,
        explainer_data,
        background_data=None,
        check_additivity=True,
        keep_abs_shaps=False,
        **shap_kwargs
    ):

        self._init_avatar()

        for m_idx in range(len(self.m_list)):
            # Extract tree and m_code
            tree = self.m_list[m_idx].model
            m_code = self.m_codes[m_idx]

            # Filter data
            attribute_filter = m_code == DESC_ENCODING
            X = explainer_data[:, attribute_filter]

            if background_data is not None:
                B = background_data[:, attribute_filter]
            else:
                B = background_data

            # Shap Calculation
            explainer = shap.TreeExplainer(
                tree, data=B, check_additivity=check_additivity, **shap_kwargs
            )
            raw_shaps = explainer.shap_values(X)

            # Process Shap values
            tsr_shaps = np.array(raw_shaps)  # tensor
            abs_shaps = np.abs(tsr_shaps)  # absolute

            if len(abs_shaps.shape) == 3:
                # In case of nominal target, sum shap values across target classes
                abs_shaps = np.sum(abs_shaps, axis=0)

            avg_shaps = np.mean(
                abs_shaps, axis=0
            )  # Avg over instances (of explainer data!)

            nrm_shaps = np.squeeze(
                normalize(avg_shaps.reshape(1, -1), norm="l1")
            )  # Normalize (between 0 and 1)

            if keep_abs_shaps:
                self.abs_shaps.append(abs_shaps)
            self.nrm_shaps.append(nrm_shaps)

        self._format_abs_shaps()
        self._format_nrm_shaps()

        return

    def _init_avatar(self):
        """Initialize avatar-datastructures that are used there.
        """
        self.abs_shaps = []
        self.nrm_shaps = []
        return

    def _format_nrm_shaps(self):
        if isinstance(self.nrm_shaps, list) and len(self.nrm_shaps) > 0:
            init = np.zeros(self.m_codes.shape)

            for m_idx, (mod, nrm_shap) in enumerate(zip(self.m_list, self.nrm_shaps)):
                init[m_idx, list(mod.desc_ids)] = nrm_shap

            self.nrm_shaps = init
        else:
            return

    def _format_abs_shaps(self):
        if isinstance(self.abs_shaps, list) and len(self.abs_shaps) > 0:
            n_models, n_attributes = self.m_codes.shape
            n_instances = self.abs_shaps[0].shape[0]
            init = np.zeros((n_models, n_instances, n_attributes))

            for m_idx, (mod, abs_shap) in enumerate(zip(self.m_list, self.abs_shaps)):
                init_abs = np.zeros((n_instances, n_attributes))
                init_abs[:, list(mod.desc_ids)] = abs_shap
                init[m_idx, :, :] = init_abs

            self.abs_shaps = init
        else:
            return

