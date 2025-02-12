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

# Perform experiments for random splits
for seed in split_seeds:
    # Create train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 / 3.0, random_state=seed
    )
    #print(f"X_train.shape: {X_train.shape}")

    for model_config in cart_hscart_estimators:  # Loop through CART and HSCART models
        #print("fit models")
        model_name = model_config.name
        model_class = model_config.cls
        model_kwargs = model_config.kwargs.copy()  # Copy to safely modify

        # Handle CART
        if model_name == 'CART':
            # Train CART
            cart_model = model_class(**model_kwargs)
            #print("CART prin to fit")
            cart_model.fit(x_train_student, y_train_student)
            #print("CART meta to fit")
            y_pred_proba = cart_model.predict_proba(x_test_student) #[:, 1]
            auc_cart = roc_auc_score(y_test_student, y_pred_proba, multi_class='ovo')

            # Append CART results
            results.append({
                'Dataset': "Adult",
                'Model': 'CART',
                'Max Leaves': model_kwargs['max_leaf_nodes'],  # Directly taken from ModelConfig
                'Lambda': None,  # CART does not use lambda
                'AUC': auc_cart,
                'Split Seed': "na"
            })
            #tree.plot_tree(cart_model)

        # Handle HSCART
        if model_name == 'HSCART':
            # Set lambda list for HSCART
            print("ksekiname")

            #model_kwargs['reg_param_list'] = lambda_list
            #print("ksekiname 1.5")
            #print(str(model_kwargs))
            model = model_class(**model_kwargs)
            #print("ksekiname 2")
            #print("lambda prin : " + str(model.get_params))
            start_time = time.time()
            model.fit(x_train_student, y_train_student)
            end_time = time.time()
            print(f"Total time {model_kwargs['max_leaf_nodes']}: {(end_time - start_time) / 60}")
            #print("lambda meta : " + str(model.get_params))

            # Predict and calculate AUC
            y_pred_proba = model.predict_proba(x_test_student) # [:, 1]

            predictions = model.predict(x_test_student)
            accuracy = accuracy_score(y_test_student, predictions)

            auc_hscart = roc_auc_score(y_test_student, y_pred_proba, multi_class='ovo')
            #print(f"HSCART selected lambda: {model.reg_param}")

            # Append HSCART results
            results.append({
                'Dataset': "Adult",
                'Model': 'HSCART',
                'Max Leaves': model_kwargs['max_leaf_nodes'],  # Directly taken from ModelConfig
                'Lambda': model.reg_param,  # Save the selected lambda
                'AUC': auc_hscart,
                'Split Seed': "na"
            })
            #print(model)
            tree.plot_tree(model)

# Convert results to DataFrame and save
results_df = pd.DataFrame(results)
results_df.to_csv(f"{specific_dataset_name}_cart_hscart_results_combined_GTC.csv", index=False)
#print(f"Results saved to {specific_dataset_name}_cart_hscart_results_combined_GTC.csv")



########################################################################################################################
# Load results of models
results_df = pd.read_csv("credit_card_clean_cart_hscart_results_combined_GTC.csv")

# Step 1: Filter and organize data
cart_results = results_df[results_df["Model"] == "CART"]
hscart_results = results_df[results_df["Model"] == "HSCART"]

# Step 2: Group by Max Leaves and calculate mean AUC and standard error
def calculate_mean_and_sem(data):
    grouped = data.groupby("Max Leaves")["AUC"]
    mean = grouped.mean()
    sem = grouped.sem()  # Standard error of the mean
    return mean, sem

cart_mean, cart_sem = calculate_mean_and_sem(cart_results)
hscart_mean, hscart_sem = calculate_mean_and_sem(hscart_results)

all_values = pd.concat([cart_mean, hscart_mean])
y_min = all_values.min() - 0.01  # Slightly below the min value of y axis for plotting
y_max = all_values.max() + 0.01  # Slightly above the max value of y axis for plotting

# Generate custom ticks dynamically
ticks_step = 0.025  # Step size for ticks 9same as paper's for heart classification
custom_ticks = [round(y_min + i * ticks_step, 3) for i in range(int((y_max - y_min) / ticks_step) + 1)]

# Plot
plt.figure(figsize=(8, 6))

# Plot CART results
plt.plot(
    cart_mean.index, cart_mean, label="CART", marker="o", linestyle="-", color="blue"
)

# Plot HSCART results
plt.plot(
    hscart_mean.index, hscart_mean, label="HSCART", marker="s", linestyle="-", color="red"
)

# Set custom y-axis ticks
plt.yticks(custom_ticks, [f"{tick:.3f}" for tick in custom_ticks])

# Add labels, title, and legend
plt.xlabel("Number of Leaves", fontsize=12)
plt.ylabel("AUC", fontsize=12)
plt.title("FASHION NO PCA", fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()

# Save and show the plot
#plt.savefig("heart_dataset_plot_GTC.png", dpi=300)
plt.show()
