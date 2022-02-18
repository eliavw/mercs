# MERCS

MERCS stands for **multi-directional ensembles of classification and regression trees**. It is a novel ML-paradigm under active development at the [DTAI-lab at KU Leuven](https://dtai.cs.kuleuven.be/).

## Installation

Easy via pip;

```
pip install mercs
```

## Website

Our (very small) website can be found [here](https://eliavw.github.io/mercs/).


## Tutorials

Cf. the [quickstart section](https://eliavw.github.io/mercs/quickstart) of the website.

## Code

MERCS is fully open-source cf. our [github-repository](https://github.com/eliavw/mercs/)

## Publications

MERCS is an active research project, hence we periodically publish our findings;

### MERCS: Multi-Directional Ensembles of Regression and Classification Trees

**Abstract**
*Learning a function f(X) that predicts Y from X is the archetypal Machine Learning (ML) problem. Typically, both sets of attributes (i.e., X,Y) have to be known before a model can be trained. When this is not the case, or when functions f(X) that predict Y from X are needed for varying X and Y, this may introduce significant overhead (separate learning runs for each function). In this paper, we explore the possibility of omitting the specification of X and Y at training time altogether, by learning a multi-directional, or versatile model, which will allow prediction of any Y from any X. Specifically, we introduce a decision tree-based paradigm that generalizes the well-known Random Forests approach to allow for multi-directionality. The result of these efforts is a novel method called MERCS: Multi-directional Ensembles of Regression and Classification treeS. Experiments show the viability of the approach.*

**Authors**
Elia Van Wolputte, Evgeniya Korneva, Hendrik Blockeel

**Open Access**
A pdf version can be found at [AAAI-publications](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/16875/16735)


ISO-690
```
VAN WOLPUTTE, Elia; KORNEVA, Evgeniya; BLOCKEEL, Hendrik. MERCS: multi-directional ensembles of regression and classification trees. In: Thirty-Second AAAI Conference on Artificial Intelligence. 2018.
```

Bibtex
```bibtex
@inproceedings{van2018mercs,
  title={MERCS: multi-directional ensembles of regression and classification trees},
  author={Van Wolputte, Elia and Korneva, Evgeniya and Blockeel, Hendrik},
  booktitle={Thirty-Second AAAI Conference on Artificial Intelligence},
  year={2018}
}

```

### Model Selection for Multi-Directional Ensemble of Regression and Classification Trees

**Abstract**
*Multi-directional ensembles of Classification and Regression treeS (MERCS) extend random forests towards multi-directional prediction. The current work discusses different strategies of induction of such a model, which comes down to selecting sets of input and output attributes for each tree in the ensemble. It has been previously shown that employing multi-targets trees as MERCS component models helps reduce both model induction and inference time. In the current work, we present a novel output selection strategy for MERCS component model that takes relatedness between the attributes into account and compare it to the random output selection. We observe that accounting for relatedness between targets has a limited effect on performance and discuss the reasons why it is inherently difficult to improve the overall performance of a multi-directional model by altering target selection strategy for its component models.*

**Authors**
Evgeniya Korneva, Hendrik Blockeel

**Open Access**
A pdf version can be found at [KU Leuven](https://lirias.kuleuven.be/retrieve/529405)


ISO-690
```
KORNEVA, Evgeniya; BLOCKEEL, Hendrik. Model Selection for Multi-directional Ensemble of Regression and Classification Trees. In: Benelux Conference on Artificial Intelligence. Springer, Cham, 2018. p. 52-64.
```

Bibtex
```bibtex
@inproceedings{korneva2018model,
  title={Model Selection for Multi-directional Ensemble of Regression and Classification Trees},
  author={Korneva, Evgeniya and Blockeel, Hendrik},
  booktitle={Benelux Conference on Artificial Intelligence},
  pages={52--64},
  year={2018},
  organization={Springer}
}
```

### Missing value imputation with MERCS: a faster alternative to MissForest

**Abstract**
*Fundamentally, many problems in Machine Learning are understood as some form of function approximation; given a dataset D, learn a function fX→Y . However, this overlooks the ubiquitous problem of missing data. E.g., if afterwards an unseen instance has missing input variables, we actually need a function f:X′→Y with X′⊂X to predict its label. Strategies to deal with missing data come in three kinds: naive, probabilistic and iterative. The naive case replaces missing values with a fixed value (e.g. the mean), then uses f:X→Y as if nothing was ever missing. The probabilistic case has a generative model M of D and uses probabilistic inference to find the most likely value of Y, given values for any subset of X. The iterative approach consists of a loop: according to some model M, fill in all the missing values based on the given ones, retrain M on the completed data and redo your predictions, until these converge. MissForest is a well-known realization of this idea using Random Forests. In this work, we establish the connection between MissForest and MERCS (a multi-directional generalization of Random Forests). We go on to show that under certain (realistic) conditions where the retraining step in MissForest becomes a bottleneck, MERCS (which is trained only once) offers at-par predictive performance at a fraction of the time cost.*

**Authors**
Elia Van Wolputte, Hendrik Blockeel

**Open Access**
A pdf version can be found at [KU Leuven](https://lirias.kuleuven.be/retrieve/583955)

**Cite**

ISO-690
```
VAN WOLPUTTE, Elia; BLOCKEEL, Hendrik. Missing value imputation with MERCS: a faster alternative to MissForest. In: International Conference on Discovery Science. Springer, Cham, 2020. p. 502-516.
```

Bibtex
```bibtex
@inproceedings{wolputte2020missing,
  title={Missing value imputation with MERCS: a faster alternative to MissForest},
  author={Van Wolputte, Elia and Blockeel, Hendrik},
  booktitle={International Conference on Discovery Science},
  pages={502--516},
  year={2020},
  organization={Springer}
}
```

## People

People involved in this project:

* [Elia Van Wolputte](https://eliavw.github.io/personal-site/)
* [Evgeniya Korneva](https://scholar.google.com/citations?user=5trsrZUAAAAJ&hl=nl&oi=ao)
* [Prof. Hendrik Blockeel](https://people.cs.kuleuven.be/~hendrik.blockeel/)

