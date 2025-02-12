import sklearn.tree
from config.shrinkage.models import ESTIMATORS_CLASSIFICATION
from imodels.util.data_util import get_clean_dataset
import numpy as np
from sklearn import tree
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from imodels import get_clean_dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.metrics import accuracy_score
import csv

########################################################################################################################

# Imitialize a list of classifier that consists only of CART and HSCART classifiers
cart_hscart_estimators = [
    model for model_group in ESTIMATORS_CLASSIFICATION
    for model in model_group
    if model.name in ['CART', 'HSCART']
]


########################################################################################################################
# Define hyperparameters
split_seeds = range(10)  # Ten random splits
#lambda_list = [0.0, 0.1, 1.0, 10, 25.0, 50.0, 100.0]  # Lambda value list equal to the HS paper
#lambda_list=[0.0]

######################
#
# DATASETS
#
##################
specific_dataset_name = "credit_card_clean"
X, y, feat_names = get_clean_dataset(specific_dataset_name, data_source="imodels")
n_samples, n_features = X.shape
#print(f"n_samples: {n_samples}, n_features: {n_features}")

# Results storage
results = []

#print(f"Processing dataset: {specific_dataset_name}")

accuracy_results = {"CART": [], "HSCART": []}  # Store accuracy scores

for seed in split_seeds:
    # Create train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 / 3.0, random_state=seed
    )

    for model_config in cart_hscart_estimators:  # Loop through CART and HSCART models
        model_name = model_config.name
        model_class = model_config.cls
        model_kwargs = model_config.kwargs.copy()

        # Handle CART
        if model_name == 'CART':
            cart_model = model_class(**model_kwargs)
            cart_model.fit(x_train_credit, y_train_credit)
            y_pred_proba = cart_model.predict_proba(x_test_credit)[:, 1]
            auc_cart = roc_auc_score(y_test_credit, y_pred_proba) #  multi_class='ovo'

            # Predict and compute accuracy
            predictions = cart_model.predict(x_test_credit)
            accuracy = accuracy_score(y_test_credit, predictions)
            accuracy_results["CART"].append(accuracy)

            # Append CART results
            results.append({
                'Dataset': "Adult",
                'Model': 'CART',
                'Max Leaves': model_kwargs['max_leaf_nodes'],
                'Lambda': None,
                'AUC': auc_cart,
                'Split Seed': seed
            })

        # Handle HSCART
        if model_name == 'HSCART':
            model = model_class(**model_kwargs)
            start_time = time.time()
            model.fit(x_train_cifar_flat, y_train_cifar.ravel())
            end_time = time.time()
            print(f"Total time {model_kwargs['max_leaf_nodes']}: {(end_time - start_time) / 60}")

            y_pred_proba = model.predict_proba(x_test_cifar_flat)
            predictions = model.predict(x_test_cifar_flat)
            accuracy = accuracy_score(y_test_cifar, predictions)
            accuracy_results["HSCART"].append(accuracy)

            auc_hscart = roc_auc_score(y_test_cifar, y_pred_proba, multi_class='ovo')

            # Append HSCART results
            results.append({
                'Dataset': "Adult",
                'Model': 'HSCART',
                'Max Leaves': model_kwargs['max_leaf_nodes'],
                'Lambda': model.reg_param,
                'AUC': auc_hscart,
                'Split Seed': seed
            })

# Compute and display the average accuracy over five runs
avg_accuracy_cart = sum(accuracy_results["CART"]) / len(accuracy_results["CART"])
avg_accuracy_hscart = sum(accuracy_results["HSCART"]) / len(accuracy_results["HSCART"])
print(f"Average Accuracy (CART): {avg_accuracy_cart:.4f}")
print(f"Average Accuracy (HSCART): {avg_accuracy_hscart:.4f}")

# Convert results to DataFrame and save
results_df = pd.DataFrame(results)
