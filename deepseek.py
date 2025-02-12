import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn import tree
import time
from ucimlrepo import fetch_ucirepo
from config.shrinkage.models import ESTIMATORS_CLASSIFICATION
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import kagglehub
import os
from sklearn.decomposition import PCA



# Splitting datasets
def split_data(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test


# fetch dataset
multivariate_gait_data = fetch_ucirepo(id=760)

# data (as pandas dataframes)
X = multivariate_gait_data.data.features
y = multivariate_gait_data.data.targets

features = X.drop(columns=['condition'])
features = features.columns

x = X.drop(columns=['condition'])
y = X['condition']

x = x.to_numpy()
y = y.to_numpy()

x_train_gait, x_test_gait, y_train_gait, y_test_gait = split_data(x, y)



##########################################################################################
# # Download dataset
dataset_path = kagglehub.dataset_download("rabieelkharoua/students-performance-dataset")

print("Path to dataset files:", dataset_path)

# List all files in the dataset directory
dataset_files = os.listdir(dataset_path)

# Find the CSV file(s)
csv_files = [f for f in dataset_files if f.endswith('.csv')]

if not csv_files:
    raise FileNotFoundError("No CSV file found in dataset folder.")

# Load the first CSV file
csv_path = os.path.join(dataset_path, csv_files[0])
df = pd.read_csv(csv_path)

# Display the first few rows of the DataFrame
print(df.head())
X = df.drop(columns=['GradeClass'])  # Drop the target column to get features
y = df['GradeClass']  # Target is the GradeClass column



pca = PCA(0.99)
pca.fit(X)
X = pca.transform(X)



features = df.drop(columns=['GradeClass'])
features = features.columns



##########################################################################################


# # Fetch dataset
predict_students_dropout_and_academic_success = fetch_ucirepo(id=697)

# Data (as pandas dataframes)
X = predict_students_dropout_and_academic_success.data.features
y = predict_students_dropout_and_academic_success.data.targets

pca = PCA(0.99)
pca.fit(X)
X = pca.transform(X)

# Convert categorical variables to numerical using Label Encoding
categorical_columns = y.select_dtypes(include=['object']).columns
for col in categorical_columns:
    le = LabelEncoder()
    y.loc[:, col] = le.fit_transform(y[col])  # Fix SettingWithCopyWarning




y = y.squeeze()  # Converts DataFrame with one column to Series
y = y.astype(int)
y = y.dropna()  # Remove NaNs

X = X.to_numpy()
y = y.to_numpy()

# # Split data
x_train_student, x_test_student, y_train_student, y_test_student = split_data(X, y)





##########################################################################################



# Initialize a list of classifiers that consists only of CART and HSCART classifiers
cart_hscart_estimators = [
    model for model_group in ESTIMATORS_CLASSIFICATION
    for model in model_group
    if model.name in ['CART', 'HSCART']
]

results = []

for model_config in cart_hscart_estimators:  # Loop through CART and HSCART models
    model_name = model_config.name
    model_class = model_config.cls
    model_kwargs = model_config.kwargs.copy()  # Copy to safely modify

    if model_name == 'CART':
        # Train CART
        cart_model = model_class(**model_kwargs)
        cart_model.fit(x_train_gait, y_train_gait)
        y_pred_proba = cart_model.predict_proba(x_test_gait)
        predictions = cart_model.predict(x_test_gait)
        accuracy = accuracy_score(y_test_gait, predictions)
        #print(f"Shape of y_pred_proba: {y_pred_proba.shape}")

        # Calculate AUC for CART
        #auc_cart = roc_auc_score(y_test_student, y_pred_proba, multi_class='ovo')

        # Get tree size (number of nodes) and depth
        tree_size = cart_model.tree_.node_count
        tree_depth = cart_model.tree_.max_depth
        if model_kwargs['max_leaf_nodes'] == 20:
            plt.figure(figsize=(20, 10))
            plot_tree(cart_model)
            plt.title('CART: Gait', fontsize=35)
            plt.savefig('cart_gait')
            plt.show()


        # Append CART results
        results.append({
            'Dataset': "Adult",
            'Model': 'CART',
            'Max Leaves': model_kwargs['max_leaf_nodes'],
            'Lambda': None,  # CART does not use lambda
            #'AUC': auc_cart,
            'Tree Size': tree_size,
            'Tree Depth': tree_depth,
            'Split Seed': "na",
            'Accuracy': accuracy
        })

    elif model_name == 'HSCART':
        # Train HSCART
        hscart_model = model_class(**model_kwargs)
        start_time = time.time()
        hscart_model.fit(x_train_gait, y_train_gait)
        end_time = time.time()
        #print(f"Total time {model_kwargs['max_leaf_nodes']}: {(end_time - start_time) / 60}")

        # Predict and calculate AUC for HSCART
        y_pred_proba = hscart_model.predict_proba(x_test_gait)
        predictions = hscart_model.predict(x_test_gait)
        accuracy = accuracy_score(y_test_gait, predictions)
        #auc_hscart = roc_auc_score(y_test_student, y_pred_proba, multi_class='ovo')

        # Get tree size (number of nodes) and depth
        if hasattr(hscart_model, 'estimator_'):
            decision_tree = hscart_model.estimator_
            tree_size = decision_tree.tree_.node_count
            tree_depth = decision_tree.tree_.max_depth
        else:
            raise AttributeError("HSCART model does not contain an attribute 'estimator_' for tree visualization.")

        # Append HSCART results
        results.append({
            'Dataset': "Adult",
            'Model': 'HSCART',
            'Max Leaves': model_kwargs['max_leaf_nodes'],
            'Lambda': hscart_model.reg_param,  # Save the selected lambda
            #'AUC': auc_hscart,
            'Tree Size': tree_size,
            'Tree Depth': tree_depth,
            'Split Seed': "na",
            'Accuracy': accuracy
        })


# Convert results to DataFrame and save
results_df = pd.DataFrame(results)

################################################################################################


# Extract the underlying decision tree
if hasattr(hscart_model, 'estimator_'):
    decision_tree = hscart_model.estimator_
else:
    raise AttributeError("HSCART model does not contain an attribute 'estimator_' for tree visualization.")

# Plot the tree
plt.figure(figsize=(20, 10))
plot_tree(decision_tree, filled=True, feature_names=features)
plt.title("HSCART: Gait", fontsize=35)
plt.savefig('HSCART_gait')
plt.show()


##################################################################################################



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Filter results for CART and HSCART
cart_results = results_df[results_df['Model'] == 'CART']
hscart_results = results_df[results_df['Model'] == 'HSCART']

# Extract tree sizes and depths
cart_tree_sizes = cart_results['Tree Size']
cart_tree_depths = cart_results['Tree Depth']
hscart_tree_sizes = hscart_results['Tree Size']
hscart_tree_depths = hscart_results['Tree Depth']

# Combine data for box plots
tree_sizes = pd.DataFrame({
    'Model': ['CART'] * len(cart_tree_sizes) + ['HSCART'] * len(hscart_tree_sizes),
    'Tree Size': list(cart_tree_sizes) + list(hscart_tree_sizes)
})

tree_depths = pd.DataFrame({
    'Model': ['CART'] * len(cart_tree_depths) + ['HSCART'] * len(hscart_tree_depths),
    'Tree Depth': list(cart_tree_depths) + list(hscart_tree_depths)
})

# Create box plots
plt.figure(figsize=(12, 6))

# Box plot for Tree Size
plt.subplot(1, 2, 1)
sns.boxplot(x='Model', y='Tree Size', data=tree_sizes, palette='Set2')
plt.title('Tree Size Comparison')
plt.xlabel('Model')
plt.ylabel('Tree Size (Number of Nodes)')

# Box plot for Tree Depth
plt.subplot(1, 2, 2)
sns.boxplot(x='Model', y='Tree Depth', data=tree_depths, palette='Set2')
plt.title('Tree Depth Comparison')
plt.xlabel('Model')
plt.ylabel('Tree Depth')

plt.tight_layout()
plt.show()

