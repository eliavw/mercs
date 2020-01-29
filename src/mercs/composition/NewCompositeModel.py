import numpy as np
from dask import delayed
from sklearn.preprocessing import normalize

from ..algo.inference_v3 import compute
from .compose import o, x
from ..graph.network import get_ids, node_label, get_nodes

from ..utils import debug_print

VERBOSITY = 0


class NewCompositeModel(object):
    def __init__(self, diagram, desc_ids=None, targ_ids=None, nominal_attributes=None, n_component_models=0):

        # Assign desc and targ ids

        for k, idx in diagram.nodes():
            if k == 'M' and (idx >= n_component_models):
                diagram.nodes[(k, idx)]['shape'] = 'square'

        if desc_ids is not None:
            self.desc_ids = desc_ids
        else:
            self.desc_ids = diagram.desc_ids

        if targ_ids is not None:
            self.targ_ids = targ_ids
        else:
            self.targ_ids = diagram.targ_ids

        self.desc_ids = sorted(list(self.desc_ids))
        self.targ_ids = sorted(list(self.targ_ids))

        self.feature_importances_ = [1/len(self.desc_ids) for _ in self.desc_ids]

        self.predict = _get_predict(diagram, self.targ_ids)

        if nominal_attributes is not None:
            self.nominal_targ_ids = nominal_attributes.intersection(self.targ_ids)
            self.classes_ = _get_classes_(diagram, self.nominal_targ_ids)
            self.predict_proba = _get_predict_proba(diagram, self.nominal_targ_ids)

        self.score = 1

        return


    def get_confidences(self, X=None, redo=False, normalize_outputs=True):

        confidences = [np.array([[1.0]]) for t in self.targ_ids]
        nominal_prb = self.predict_proba(X, redo=redo)

        for targ_id, proba in zip(self.nominal_targ_ids, nominal_prb):
            targ_idx = self.targ_ids.index(targ_id)
            confidences[targ_idx] = proba

        if normalize_outputs:
            confidences = [np.max(normalize(c, norm='l1')) for c in confidences]
        return confidences


def _get_predict(diagram, targ_ids):
    def predict(X, redo=True):
        diagram.data = X

        if redo:
            clean_cache(diagram)

        collector = [compute(diagram, ("D", n)) for n in targ_ids]
        if len(targ_ids) == 1:
            return collector.pop()
        else:
            return np.stack(collector, axis=1)

    return predict


def _get_predict_proba(diagram, nominal_targ_ids):
    def predict_proba(X, redo=True):
        diagram.data = X

        if redo:
            clean_cache(diagram)

        collector = [compute(diagram, ("D", n), proba=True) for n in nominal_targ_ids]
        return collector

    return predict_proba


def _get_classes_(diagram, nominal_targ_ids):
    collector = [diagram.nodes[("D", n)]["classes"] for n in nominal_targ_ids]
    return collector


def clean_cache(diagram):
    for n in diagram.nodes:
        if diagram.nodes[n].get("result", None) is not None:
            diagram.nodes[n]["result"] = None
        if diagram.nodes[n].get("result_proba", None) is not None:
            diagram.nodes[n]["result_proba"] = None
    return
