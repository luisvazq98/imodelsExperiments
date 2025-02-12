from functools import partial
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from imodels import (
    GreedyTreeClassifier, GreedyTreeRegressor, HSTreeRegressorCV, HSTreeClassifier, HSTreeRegressor
)
from util import ModelConfig
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# ================= PATCH HSTreeClassifier FOR LAMBDA = 0 =================

from copy import deepcopy

from hierarchical_shrinkage import HSTreeClassifierCV






RANDOM_FOREST_DEFAULT_KWARGS = {'random_state': 0}
TREE_DEPTHS_CLASS = [2]

# ESTIMATORS_CLASSIFICATION = [
#     [ModelConfig('CART', GreedyTreeClassifier, 'max_depth', n)
#      for n in TREE_DEPTHS_CLASS],
#     [ModelConfig('HSCART', partial(HSTreeClassifierCV, estimator=GreedyTreeClassifier(max_depth=n)),
#                  'max_depth', n)
#      for n in TREE_DEPTHS_CLASS],
#     [ModelConfig('Random_Forest', RandomForestClassifier, 'n_estimators', n, other_params=RANDOM_FOREST_DEFAULT_KWARGS)
#      for n in TREE_DEPTHS_CLASS],
#     [ModelConfig('Gradient_Boosting', GradientBoostingClassifier, 'n_estimators', n,
#                  other_params=RANDOM_FOREST_DEFAULT_KWARGS)
#      for n in TREE_DEPTHS_CLASS],
# ]


#TREE_LEAF_NODES = [4]
TREE_LEAF_NODES = [2, 4, 8, 12, 15, 20, 24, 28, 30, 32]

# ESTIMATORS_CLASSIFICATION = [
#     # CART
#     [ModelConfig(
#         'CART',
#         DecisionTreeClassifier,
#         'max_leaf_nodes',
#         max_leaf
#     ) for max_leaf in TREE_LEAF_NODES],
#
#     # HSCART
#     [ModelConfig(
#         'HSCART',
#         partial(HSTreeClassifierCV, estimator_=DecisionTreeClassifier(max_leaf_nodes=max_leaf)),
#         'max_leaf_nodes',
#         max_leaf
#     ) for max_leaf in TREE_LEAF_NODES],
#
#     # Random Forest
#     [ModelConfig(
#         'Random_Forest',
#         RandomForestClassifier,
#         'n_estimators',
#         n,
#         other_params=RANDOM_FOREST_DEFAULT_KWARGS
#     ) for n in TREE_LEAF_NODES],
#
#     # Gradient Boosting
#     [ModelConfig(
#         'Gradient_Boosting',
#         GradientBoostingClassifier,
#         'n_estimators',
#         n,
#         other_params=RANDOM_FOREST_DEFAULT_KWARGS
#     ) for n in TREE_LEAF_NODES],
# ]


ESTIMATORS_CLASSIFICATION = [
    # CART
    [ModelConfig(
        'CART',
        DecisionTreeClassifier,
        'max_leaf_nodes',
        max_leaf
    ) for max_leaf in TREE_LEAF_NODES],

    # HSCART
    [ModelConfig(
        'HSCART',
        partial(HSTreeClassifierCV, estimator_=DecisionTreeClassifier(max_leaf_nodes=max_leaf)),
        'max_leaf_nodes',
        max_leaf
    ) for max_leaf in TREE_LEAF_NODES],

    # Random Forest
    [ModelConfig(
        'Random_Forest',
        RandomForestClassifier,
        'n_estimators',
        n,
        other_params=RANDOM_FOREST_DEFAULT_KWARGS
    ) for n in TREE_LEAF_NODES],

    # Gradient Boosting
    [ModelConfig(
        'Gradient_Boosting',
        GradientBoostingClassifier,
        'n_estimators',
        n,
        other_params=RANDOM_FOREST_DEFAULT_KWARGS
    ) for n in TREE_LEAF_NODES],
]





# ESTIMATORS_CLASSIFICATION = [
#     [ModelConfig('CART', DecisionTreeClassifier, 'max_depth', n)
#      for n in TREE_DEPTHS_CLASS],
#     [ModelConfig('HSCART', partial(HSTreeClassifierCV, estimator_=DecisionTreeClassifier(max_depth=n)),
#                  'max_depth', n)
#      for n in TREE_DEPTHS_CLASS],
#     [ModelConfig('Random_Forest', RandomForestClassifier, 'n_estimators', n, other_params=RANDOM_FOREST_DEFAULT_KWARGS)
#      for n in TREE_DEPTHS_CLASS],
#     [ModelConfig('Gradient_Boosting', GradientBoostingClassifier, 'n_estimators', n,
#                  other_params=RANDOM_FOREST_DEFAULT_KWARGS)
#      for n in TREE_DEPTHS_CLASS],
# ]

#TREE_DEPTHS_CLASS = [1, 2, 3, 5, 7, 10]
# TREE_DEPTHS_CLASS = [2]
# ESTIMATORS_CLASSIFICATION = [
#     # CART (Gini)
#     [ModelConfig('CART', DecisionTreeClassifier, 'max_depth', n)
#      for n in TREE_DEPTHS_CLASS],
#
#     # HSCART using HSTreeClassifierCV
#     [ModelConfig(
#         'HSCART',
#         partial(HSTreeClassifierCV, estimator_=DecisionTreeClassifier(max_depth=n)),
#         'max_depth',
#         n
#     )
#      for n in TREE_DEPTHS_CLASS],
#
#     # HSTreeClassifier as an additional option (without CV)
#     [ModelConfig(
#         'HSTreeClassifier',
#         partial(HSTreeClassifier, estimator_=DecisionTreeClassifier(max_depth=n)),
#         None,
#         None
#     )
#         for n in TREE_DEPTHS_CLASS],
#
#     # Random Forest
#     [ModelConfig('Random_Forest', RandomForestClassifier, 'n_estimators', n)
#      for n in [3, 10, 25, 50]],
#
#     # Gradient Boosting
#     [ModelConfig(
#         'Gradient_Boosting',
#         GradientBoostingClassifier,
#         'n_estimators',
#         n
#     )
#      for n in [10, 50, 100]],
# ]



# ESTIMATORS_CLASSIFICATION = [
#     [ModelConfig('CART', DecisionTreeClassifier, 'max_depth', n)
#      for n in [1, 2, 3, 5, 7, 10]],  # Replaced GreedyTreeClassifier with DecisionTreeClassifier from sklearn
#     [ModelConfig('HSCART', partial(HSTreeClassifierCV, estimator=DecisionTreeClassifier(max_depth=n)),
#                  'max_depth', n)
#      for n in [1, 2, 3, 5, 7, 10]],
#     [ModelConfig('Random_Forest', RandomForestClassifier, 'n_estimators', n, other_params=RANDOM_FOREST_DEFAULT_KWARGS)
#      for n in [3, 10, 25, 50]],
#     [ModelConfig('Gradient_Boosting', GradientBoostingClassifier, 'n_estimators', n,
#                  other_params=RANDOM_FOREST_DEFAULT_KWARGS)
#      for n in [10, 50, 100]],
# ]


# ESTIMATORS_CLASSIFICATION = [
#     [ModelConfig('CART', GreedyTreeClassifier, 'max_depth', n)
#      for n in [1, 2, 3, 5, 7, 10]],
#     [ModelConfig('HSCART', partial(HSTreeClassifierCV, estimator=DecisionTreeClassifier(max_depth=n)),
#                  'max_depth', n)
#      for n in [1, 2, 3, 5, 7, 10]],
#     [ModelConfig('Random_Forest', RandomForestClassifier, 'n_estimators', n, other_params=RANDOM_FOREST_DEFAULT_KWARGS)
#      for n in [3, 10, 25, 50]],
#     [ModelConfig('Gradient_Boosting', GradientBoostingClassifier, 'n_estimators', n,
#                  other_params=RANDOM_FOREST_DEFAULT_KWARGS)
#      for n in [10, 50, 100]],
# ]

ENSEMBLE_ESTIMATOR_NUMS = [3, 10, 25, 50]
TREE_DEPTHS = [1, 2, 3, 4, 5, 7, 8, 10, 15, 20, 25]

ESTIMATORS_REGRESSION = [
    # CART (MSE)
    [ModelConfig('CART_(MSE)', DecisionTreeRegressor, 'max_depth', n)
     for n in TREE_DEPTHS],

    # HSCART using HSTreeRegressorCV
    [ModelConfig(
        'HSCART',
        partial(HSTreeRegressorCV, estimator_=DecisionTreeRegressor(max_depth=n)),
        'max_depth',
        n
    )
     for n in TREE_DEPTHS],

    # HSTreeRegressor as an additional option (without CV)
    [ModelConfig(
        'HSTreeRegressor',
        HSTreeRegressor,
        'max_depth',
        n,
        other_params={
            'estimator_': DecisionTreeRegressor(max_depth=n)  # Pass the configured estimator directly
        }
    )
     for n in TREE_DEPTHS],

    # Random Forest
    [ModelConfig('Random_Forest', RandomForestRegressor, other_params={'n_estimators': n})
     for n in ENSEMBLE_ESTIMATOR_NUMS],

    # HSRandom_Forest using HSTreeRegressorCV
    [ModelConfig(
        'HSRandom_Forest',
        partial(HSTreeRegressorCV, estimator_=RandomForestRegressor(n_estimators=n))
    )
     for n in ENSEMBLE_ESTIMATOR_NUMS],

    # Gradient Boosting
    [ModelConfig(
        'Gradient_Boosting',
        GradientBoostingRegressor,
        'n_estimators',
        n,
        other_params=RANDOM_FOREST_DEFAULT_KWARGS
    )
     for n in ENSEMBLE_ESTIMATOR_NUMS],

    # HSGradient_Boosting using HSTreeRegressorCV
    [ModelConfig(
        'HSGradient_Boosting',
        partial(HSTreeRegressorCV, estimator_=GradientBoostingRegressor(n_estimators=n))
    )
     for n in ENSEMBLE_ESTIMATOR_NUMS],
]



# ESTIMATORS_REGRESSION = [
#     [ModelConfig('CART_(MSE)', DecisionTreeRegressor, 'max_depth', n)
#      for n in TREE_DEPTHS],  # Replaced GreedyTreeRegressor with DecisionTreeRegressor
#     [ModelConfig(
#         'HSCART',
#         partial(HSTreeRegressorCV, estimator=DecisionTreeRegressor(max_depth=n)),
#         'max_depth',
#         n
#     )
#         for n in TREE_DEPTHS],
#     [ModelConfig('Random_Forest', RandomForestRegressor, other_params={'n_estimators': n})
#      for n in ENSEMBLE_ESTIMATOR_NUMS],
#     [ModelConfig('HSRandom_Forest',
#                  partial(HSTreeRegressorCV, estimator_=RandomForestRegressor(n_estimators=n)))
#      for n in ENSEMBLE_ESTIMATOR_NUMS],
#     [ModelConfig('Gradient_Boosting', GradientBoostingRegressor, 'n_estimators', n,
#                  other_params=RANDOM_FOREST_DEFAULT_KWARGS)
#      for n in ENSEMBLE_ESTIMATOR_NUMS],
#     [ModelConfig('HSGradient_Boosting',
#                  partial(HSTreeRegressorCV, estimator_=GradientBoostingRegressor(n_estimators=n)))
#      for n in ENSEMBLE_ESTIMATOR_NUMS],
# ]

# ESTIMATORS_REGRESSION = [
#     [ModelConfig('CART_(MSE)', DecisionTreeRegressor, 'max_depth', n)
#      for n in TREE_DEPTHS],  # Replaced GreedyTreeRegressor with DecisionTreeRegressor
#     [ModelConfig('HSCART', partial(HSTreeRegressorCV, estimator_=DecisionTreeRegressor(max_depth=n)))
#      for n in TREE_DEPTHS],
#     [ModelConfig('Random_Forest', RandomForestRegressor, other_params={'n_estimators': n})
#      for n in ENSEMBLE_ESTIMATOR_NUMS],
#     [ModelConfig('HSRandom_Forest',
#                  partial(HSTreeRegressorCV, estimator_=RandomForestRegressor(n_estimators=n)))
#      for n in ENSEMBLE_ESTIMATOR_NUMS],
#     [ModelConfig('Gradient_Boosting', GradientBoostingRegressor, 'n_estimators', n,
#                  other_params=RANDOM_FOREST_DEFAULT_KWARGS)
#      for n in ENSEMBLE_ESTIMATOR_NUMS],
#     [ModelConfig('HSGradient_Boosting',
#                  partial(HSTreeRegressorCV, estimator_=GradientBoostingRegressor(n_estimators=n)))
#      for n in ENSEMBLE_ESTIMATOR_NUMS],
# ]

#

"""
# gosdt experiments
from imodels import OptimalTreeClassifier
from imodels.experimental import HSOptimalTreeClassifierCV

ESTIMATORS_CLASSIFICATION = [
    [ModelConfig("OptimalTreeClassifier", OptimalTreeClassifier, "regularization", 0.3)],
    [ModelConfig("HSOptimalTreeClassifierCV", HSOptimalTreeClassifierCV, "reg_param", r)
     for r in np.arange(0, 0.0051, 0.001)]
]

# bart experiments
from imodels.experimental.bartpy import BART, HSBARTCV
ESTIMATORS_CLASSIFICATION = [
    [ModelConfig("BART", BART,
                 other_params={"classification": True, "n_trees": 30, "n_samples": 100, "n_chains": 4})],
    [ModelConfig("HSBARTCV", HSBARTCV)]
]

ESTIMATORS_REGRESSION = [
    [ModelConfig("BART", BART,
                 other_params={"classification": False, "n_trees": 30, "n_samples": 100, "n_chains": 4})]
]
"""