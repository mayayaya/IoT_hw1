import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Step 1: Load the dataset
file_path = r"C:\Users\User\Desktop\Maya\IoT\Hw_01\Hw_1-2\BodyFatDataset.csv"
data = pd.read_csv(file_path)

# Define the features (X) and the target (y)
X = data.drop(columns=['BodyFat'])  # Using all other columns as features
y = data['BodyFat']  # Target is BodyFat

# Step 2: Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Define feature selection methods
feature_selection_methods = {
    'Lasso': Lasso(alpha=0.1),
    'Ridge': Ridge(alpha=1.0),
    'RFE': RFE(estimator=Ridge(alpha=1.0), n_features_to_select=1),
    'SelectKBest': SelectKBest(score_func=f_regression, k=1)
}

# Initialize dictionaries to store results
rmse_dict = {method: [] for method in feature_selection_methods.keys()}
r2_dict = {method: [] for method in feature_selection_methods.keys()}
selected_features_dict = {method: [] for method in feature_selection_methods.keys()}

# Step 4: Iterate through each feature selection method
for method_name, model in feature_selection_methods.items():
    for i in range(1, X_train.shape[1] + 1):  # Iterate over feature numbers from 1 to all
        if method_name == 'Lasso':
            model = Lasso(alpha=0.1).fit(X_train, y_train)
            important_features_idx = np.argsort(-np.abs(model.coef_))[:i]
        elif method_name == 'Ridge':
            model = Ridge(alpha=1.0).fit(X_train, y_train)
            important_features_idx = np.argsort(-np.abs(model.coef_))[:i]
        elif method_name == 'RFE':
            model = RFE(estimator=Ridge(alpha=1.0), n_features_to_select=i).fit(X_train, y_train)
            important_features_idx = np.where(model.support_)[0]
        elif method_name == 'SelectKBest':
            model = SelectKBest(score_func=f_regression, k=i).fit(X_train, y_train)
            important_features_idx = model.get_support(indices=True)

        selected_X_train = X_train.iloc[:, important_features_idx]
        selected_X_test = X_test.iloc[:, important_features_idx]

        # Train a linear regression model with the selected features
        lasso_model = Lasso(alpha=0.1)
        lasso_model.fit(selected_X_train, y_train)
        
        # Make predictions and calculate RMSE and R²
        y_pred = lasso_model.predict(selected_X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Store the results
        rmse_dict[method_name].append(rmse)
        r2_dict[method_name].append(r2)
        selected_features_dict[method_name].append(X_train.columns[important_features_idx].tolist())

# Step 5: Display selected features for each method in a table format
selected_features_df = pd.DataFrame(selected_features_dict)
print("Selected Features Table:")
print(selected_features_df)

# Step 6: Plot RMSE and R² for each feature selection method in combined charts
methods = list(feature_selection_methods.keys())


# Iterate through each method and plot both RMSE and R² in subplots
for idx, method_name in enumerate(methods):
    # Plot RMSE and R² in a single subplot
    # plt.subplot(4, 1, idx + 1)
    plt.figure(figsize=(10, 6))

    # Plot RMSE
    plt.plot(range(1, X_train.shape[1] + 1), rmse_dict[method_name], marker='o', label='RMSE', color='blue')

    # Plot R² on the same graph
    plt.plot(range(1, X_train.shape[1] + 1), r2_dict[method_name], marker='o', label='R²', color='orange')

    # Add title and labels
    plt.title(f'RMSE and R² vs Number of Variables ({method_name} Feature Selection)')
    plt.xlabel('Number of Variables')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    # Adjust layout for better visibility
    plt.tight_layout()
    plt.savefig(f"RMSE and R² vs Number of Variables ({method_name} Feature Selection).jpg")
    plt.show()

# Step 6: Plot RMSE results for each algorithm on the same graph for comparison
plt.figure(figsize=(10, 6))

for method_name, rmse_values in rmse_dict.items():
    plt.plot(range(1, X_train.shape[1] + 1), rmse_values, marker='o', label=method_name)

# Add title and labels
plt.title('RMSE vs Number of Variables for Different Feature Selection Methods')
plt.xlabel('Number of Variables')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True)
plt.savefig("RMSE vs Number of Variables for Different Feature Selection Methods.jpg")
plt.show()
