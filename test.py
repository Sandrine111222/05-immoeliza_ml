# import libraries

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

import xgboost as xgb

# import dataset

file_path = r"C:\Users\sandy\Desktop\cleaned_output.csv"
df = pd.read_csv(file_path)

target = "price" 
df = df.dropna(subset=[target])    

X = df.drop(columns=[target])
y = df[target]

if "number_rooms" in df.columns:
    df["number_rooms"] = df["number_rooms"].clip(upper=15)
else:
    raise KeyError("Column 'number_rooms' was not found in the dataset.")

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object', 'category']).columns

print("Numeric features:", list(numeric_features))
print("Categorical features:", list(categorical_features))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# Numeric: impute missing + standardize
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical: impute missing + one-hot encode
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Support Vector Regressor": SVR(),
    "XGBoost": xgb.XGBRegressor(n_estimators=300, random_state=42, verbosity=0)
}
 

 # Small grid for LR (no parameters)
param_grids = {
    "Linear Regression": {},

    "Decision Tree": {
        "regressor__max_depth": [5, 10, 20, None],
        "regressor__min_samples_split": [2, 5, 10]
    },

    # Large models → better to use RandomizedSearch
    "Random Forest": {
        "regressor__n_estimators": [100, 200, 300, 500],
        "regressor__max_depth": [10, 20, 30, None],
        "regressor__min_samples_split": [2, 5, 10]
    },

    "Support Vector Regressor": {
        "regressor__C": [0.1, 1, 10],
        "regressor__epsilon": [0.01, 0.1, 0.2],
        "regressor__kernel": ["rbf", "linear"]
    },

    "XGBoost": {
        "regressor__n_estimators": [200, 300, 400],
        "regressor__learning_rate": [0.05, 0.1, 0.2],
        "regressor__max_depth": [3, 5, 7]
    }
}



cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42) # cross-validation strategy



pipelines = {}                    # hyperparameter-tuned pipelines
best_params = {}

print("\n===== Running Hyperparameter Tuning =====")

for name, model in models.items():
    print(f"\n🔍 Tuning {name}...")

    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("regressor", model)
    ])

    param_grid = param_grids[name]

    # Use GridSearch for smaller models, RandomizedSearch for large grids
    if name in ["Random Forest", "XGBoost"]:
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_grid,
            cv=cv_strategy,
            n_iter=10,                     # faster
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
    else:
        search = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            cv=cv_strategy,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )

    search.fit(X_train, y_train)

    pipelines[name] = search.best_estimator_
    best_params[name] = search.best_params_

    print(f"Best Parameters for {name}:\n{search.best_params_}\n")



def evaluate(model, X_train, y_train, X_test, y_test, name):
    # Predictions
    train_preds = model.predict(X_train)
    test_preds  = model.predict(X_test)

    # Metrics
    rmse_test = np.sqrt(mean_squared_error(y_test, test_preds))
    mae_test  = mean_absolute_error(y_test, test_preds)

    r2_train = r2_score(y_train, train_preds)
    r2_test  = r2_score(y_test, test_preds)

    print(f"\n===== {name} =====")
    print(f"RMSE (test): {rmse_test:.3f}")
    print(f"MAE  (test): {mae_test:.3f}")
    print(f"R²   (train): {r2_train:.3f}")
    print(f"R²   (test):  {r2_test:.3f}")
    return preds



results = {}      # evaluate models

for name, pipe in pipelines.items():
    results[name] = evaluate(pipe, X_test, y_test, name)


    import joblib

# Save the best model to disk
joblib.dump(best_model, "best_house_price_model.pkl")

print("\nModel saved as 'best_house_price_model.pkl'")