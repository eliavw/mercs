from functools import partial
import numpy as np

from sklearn.preprocessing import normalize
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class DecoratedRandomForestRegressor(RandomForestRegressor):
    def fit(self, X, y, **kwargs):
        super(DecoratedRandomForestRegressor, self).fit(X, y, **kwargs)

        decorate_forest(self)
        return


class DecoratedRandomForestClassifier(RandomForestClassifier):
    def fit(self, X, y, **kwargs):
        super(DecoratedRandomForestClassifier, self).fit(X, y, **kwargs)

        decorate_forest(self)
        return


class DecoratedDecisionTreeClassifier(DecisionTreeClassifier):
    def fit(self, X, y, **kwargs):
        super(DecoratedDecisionTreeClassifier, self).fit(X, y, **kwargs)

        decorate_tree(self)
        return


class DecoratedDecisionTreeRegressor(DecisionTreeRegressor):
    def fit(self, X, y, **kwargs):
        super(DecoratedDecisionTreeRegressor, self).fit(X, y, **kwargs)

        decorate_tree(self)
        return


class Node(object):
    def __init__(
        self,
        idx=0,
        feature=-1,
        threshold=0,
        left_child=-1,
        right_child=-1,
        value=None,
        n_samples=-1,
    ):
        self.idx = idx
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.value = value
        self.leaf = left_child == right_child
        self.n_samples = n_samples
        return

    def __str__(self):
        msg = "Node {0}. Rule: X[:, {1}] <= {2:.2f}. Leaf: {3}".format(
            self.idx, self.feature, self.threshold, self.leaf
        )
        return msg


def get_nodes(tree):
    n_nodes = tree.node_count
    children_left = tree.children_left
    children_right = tree.children_right
    feature = tree.feature
    threshold = tree.threshold
    value = tree.value
    n_node_samples = tree.n_node_samples

    nodes = []

    for n_idx in range(n_nodes):
        node = Node(
            idx=n_idx,
            feature=feature[n_idx],
            threshold=threshold[n_idx],
            left_child=children_left[n_idx],
            right_child=children_right[n_idx],
            value=value[n_idx],
            n_samples=n_node_samples[n_idx],
        )

        nodes.append(node)

    return nodes


def add_nodes(clf):
    clf.nodes = get_nodes(clf.tree_)
    return


def compute_average(left_value, right_value, left_samples, right_samples):
    nominator = (left_samples * left_value + right_samples * right_value)
    denominator = left_samples + right_samples
    return  nominator/denominator


def apply_tree_on_sample(sample, nodes, n_idx=0):

    node = nodes[n_idx]

    if node.leaf:
        return node.value
    else:
        if sample[node.feature] <= node.threshold:
            return apply_tree_on_sample(sample, nodes, n_idx=node.left_child)
        elif sample[node.feature] > node.threshold:
            return apply_tree_on_sample(sample, nodes, n_idx=node.right_child)
        else:
            left_samples = nodes[node.left_child].n_samples
            right_samples = nodes[node.right_child].n_samples
            left_value = apply_tree_on_sample(sample, nodes, n_idx=node.left_child)
            right_value = apply_tree_on_sample(sample, nodes, n_idx=node.right_child)
            return compute_average(left_value, right_value, left_samples, right_samples)


def apply(X, nodes):
    out = np.zeros((X.shape[0], nodes[0].value.shape[1]))
    for i in range(X.shape[0]):
        out[i, :] = apply_tree_on_sample(X[i, :], nodes)
    return out


def predict(X, clf=None):
    if isinstance(clf, DecisionTreeClassifier):
        return predict_clf(clf, X)
    else:
        return predict_rgr(clf, X)


def predict_clf(clf, X):
    out = predict_proba(X, clf=clf, normalize_out=False)
    return clf.classes_.take(np.argmax(out, axis=1), axis=0)


def predict_proba(X, clf=None, normalize_out=True):

    if hasattr(clf, "nodes"):
        out = apply(X, clf.nodes)
    else:
        add_nodes(clf)
        out = apply(X, clf.nodes)

    if normalize_out:
        normalize(out, norm="l1", copy=False)
    return out


def predict_rgr(clf, X):
    if hasattr(clf, "nodes"):
        out = apply(X, clf.nodes)
    else:
        add_nodes(clf)
        out = apply(X, clf.nodes)
    return out


def fancy_predict(X, tree=None, **kwargs):
    try:
        y_pred = tree.classic_predict(X, **kwargs)
        return y_pred
    except ValueError:
        return predict(X, clf=tree)


def fancy_predict_proba(X, tree=None, **kwargs):
    try:
        y_proba = tree.classic_predict_proba(X, **kwargs)
        return y_proba
    except ValueError:
        return predict_proba(X, clf=tree)


def _validate_X_predict_overwrite(X):
    return X.astype(np.float32)


def decorate_tree(tree):
    add_nodes(tree)
    tree.classic_predict = tree.predict
    tree.predict = partial(fancy_predict, tree=tree)

    if isinstance(tree, DecisionTreeClassifier):
        tree.classic_predict_proba = tree.predict_proba
        tree.predict_proba = partial(fancy_predict_proba, tree=tree)

    return


def decorate_forest(forest):
    for dt in forest:
        decorate_tree(dt)

    forest._validate_X_predict = _validate_X_predict_overwrite

    return

