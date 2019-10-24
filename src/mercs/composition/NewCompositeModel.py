import numpy as np
from dask import delayed

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

        self.desc_ids = list(self.desc_ids)
        self.targ_ids = sorted(list(self.targ_ids))

        self.feature_importances_ = None

        
        self.predict = _get_predict(diagram, self.targ_ids)

        return

def _get_predict(diagram, targ_ids):

    tgt_methods = [diagram.node[('D', n)]["dask"] for n in targ_ids]

    if len(tgt_methods) == 1:
        # We can safely return the only method
        return tgt_methods.pop()
    else:
        # We need to package them together
        collector = delayed(np.stack)(tgt_methods, axis=1)
        return collector


def _get_predict_proba(diagram, nominal_targ_ids):

    tgt_methods = [diagram.node[('D', n)]["dask_proba"] for n in nominal_targ_ids]

    if len(tgt_methods) == 1:
        # We can safely return the only method
        return tgt_methods.pop()
    else:
        # We need to package them together
        collector = delayed(x)(*tgt_methods, return_type=list)
        return collector


def _get_classes(diagram, nominal_targ_ids):

    nominal_targ_ids.sort()

    # The prob and vote nodes have complete information on the classes
    targ_classes_ = [
        diagram.node[node_label(t, kind="prob")]["classes"] for t in nominal_targ_ids
    ]

    if len(targ_classes_) == 1:
        return targ_classes_.pop()
    else:
        return targ_classes_
