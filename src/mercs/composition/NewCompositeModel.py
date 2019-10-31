import numpy as np
from dask import delayed

from ..algo.inference_v3 import compute
from .compose import o, x
from ..graph.network import get_ids, node_label, get_nodes

from ..utils import debug_print

VERBOSITY = 0


class NewCompositeModel(object):
    def __init__(self, diagram, desc_ids=None, targ_ids=None):

        # Assign desc and targ ids

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

        self.feature_importances_ = None

        self.predict = _get_predict(diagram, self.targ_ids)

        self.predict_proba = _get_predict_proba(diagram, self.targ_ids)

        self.classes_ = _get_classes_(diagram, self.targ_ids)

        return


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
    collector = [diagram.node[("D", n)]["classes"] for n in nominal_targ_ids]
    return collector


def clean_cache(diagram):
    for n in diagram.nodes:
        if diagram.node[n].get("result", None) is not None:
            diagram.node[n]["result"] = None
        if diagram.node[n].get("result_proba", None) is not None:
            diagram.node[n]["result_proba"] = None
    return
