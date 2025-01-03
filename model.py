"""import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Step 1: Load the dataset
data = pd.read_csv('datasets/computed_insight_success_of_active_sellers.csv')

# Step 2: Split the data into features (X) and target (y)
X = data.drop(columns=['totalunitssold'])
y = data['totalunitssold']

# Step 3: Identify categorical columns (assuming they are object type)
categorical_cols = X.select_dtypes(include=['object']).columns

# Step 4: Preprocessing pipeline
numeric_transformer = SimpleImputer(strategy='mean')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, ~X.columns.isin(categorical_cols)),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Model Selection and Training
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

model.fit(X_train, y_train)

# Step 7: Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)"""


# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.impute import SimpleImputer

# # Step 1: Load the dataset
# data = pd.read_csv('datasets\computed_insight_success_of_active_sellers.csv')

# # Step 2: Data Preprocessing
# # Step 3: Impute missing values
# imputer = SimpleImputer(strategy='mean')
# data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# # Step 4: Split the data into training and testing sets
# X = data_imputed.drop(columns=['totalunitssold'])
# y = data_imputed['totalunitssold']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Step 5: Model Selection and Training
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Step 6: Model Evaluation
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)

# # Step 7: Save the trained model (if needed)
# # Add code to save the model to a file for future use

# # Proceed with further analysis or deployment using the trained model


# # Step 8: Model Fine-Tuning (Optional)
# # Adjust hyperparameters or try different algorithms to improve performance

# # Step 9: Deployment
# # Deploy the trained model for making predictions on new data
# import pickle

# # Save the model to a file
# with open('trained_model.pkl', 'wb') as file:
#     pickle.dump(model, file)

# # Load the model from the file
# with open('trained_model.pkl', 'rb') as file:
#     loaded_model = pickle.load(file)

"""import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 1: Load the dataset
data = pd.read_csv('datasets/computed_insight_success_of_active_sellers.csv')

# Step 2: Data Preprocessing
# Drop non-numeric columns or encode them if they are relevant for the analysis
numeric_data = data.select_dtypes(include=['number'])

# Step 3: Handling Missing Values
# Impute missing values using SimpleImputer with a strategy (e.g., median)
imputer = SimpleImputer(strategy='median')
data_imputed = pd.DataFrame(imputer.fit_transform(numeric_data), columns=numeric_data.columns)

# Step 4: Split the data into training and testing sets
X = data_imputed.drop(columns=['totalunitssold'])
y = data_imputed['totalunitssold']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Step 7: Model Deployment
# Save the trained model for future use
# For example, using joblib
# from joblib import dump
# dump(model, 'random_forest_model.joblib')
import pickle

# Save the model to a file
with open('trained_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Load the model from the file
with open('trained_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)
"""
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier  # Import classifier instead of regressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score  # Import accuracy_score for classification
import pickle
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(data):
    try:
        # Duplicate Detection
        duplicate_orders = data[data.duplicated(subset=['orderid'])]
        if not duplicate_orders.empty:
            logger.info("Duplicate Order IDs found:")
            logger.info(duplicate_orders)
            data.drop_duplicates(subset=['orderid'], inplace=True)
            logger.info("Duplicate Order IDs removed.")

        # Data Preprocessing
        numeric_data = data.select_dtypes(include=['number'])
        imputer = SimpleImputer(strategy='median')
        data_imputed = pd.DataFrame(imputer.fit_transform(numeric_data), columns=numeric_data.columns)

        # Model Training
        X = data_imputed.drop(columns=['totalunitssold'])  # Features
        y = data_imputed['totalunitssold']  # Target variable
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)  # Use RandomForestClassifier
        model.fit(X_train, y_train)

        # Model Evaluation
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
        logger.info(f"Model accuracy: {accuracy}")

        # Save the trained model to a file
        with open('trained_model.pkl', 'wb') as file:
            pickle.dump(model, file)

        logger.info("Model saved successfully.")
    except Exception as e:
        logger.error(f"Error during model training: {e}")

if __name__ == "__main__":
    # Step 1: Load the dataset
    data = pd.read_csv('datasets/computed_insight_success_of_active_sellers.csv')

    # Train the model
    train_model(data)
