# test.py
import os
import traceback
import joblib
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

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

# Try import xgboost but continue if not installed
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False
    print("⚠ xgboost not available; skipping XGBoost model.")



# LOAD DATA

def load_data(file_path, target="price"):
    df = pd.read_csv(file_path)
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in {file_path}")
    df = df.dropna(subset=[target])
    if "number_rooms" in df.columns:
        df["number_rooms"] = df["number_rooms"].clip(upper=15)
    else:
        raise KeyError("Column 'number_rooms' was not found in the dataset.")
    X = df.drop(columns=[target])
    y = df[target]
    return X, y, df




# PREPROCESSING PIPELINE

def make_preprocessor(X):
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # FIXED: Use sparse_output=False for scikit-learn >= 1.2
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )
    return preprocessor, numeric_features, categorical_features




# CREATE SEARCH OBJECT

def build_search(name, model, param_grid, preprocessor, cv_strategy):
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("regressor", model)
    ])

    if name in ["Random Forest", "XGBoost"]:
        return RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_grid,
            cv=cv_strategy,
            n_iter=10,
            scoring="neg_root_mean_squared_error",
            n_jobs=1,
            verbose=1,
            random_state=42
        )

    return GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
        verbose=1
    )




# TUNE A SINGLE MODEL (RUNS IN THREAD)

def tune_model(name, model, param_grid, preprocessor, cv_strategy, X_train, y_train):
    try:
        search = build_search(name, model, param_grid, preprocessor, cv_strategy)
        print(f"[{name}] Starting tuning…")
        search.fit(X_train, y_train)
        print(f"[{name}] Done — best score: {search.best_score_}")
        return name, search.best_estimator_, search.best_params_
    except Exception:
        print(f"Error while tuning {name}:")
        traceback.print_exc()
        raise




# MODEL EVALUATION

def evaluate(model, X_train, y_train, X_test, y_test, name):
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    rmse_test = float(np.sqrt(mean_squared_error(y_test, test_preds)))
    mae_test  = float(mean_absolute_error(y_test, test_preds))
    r2_train  = float(r2_score(y_train, train_preds))
    r2_test   = float(r2_score(y_test, test_preds))

    print(f"\n===== {name} =====")
    print(f"RMSE (test): {rmse_test:.3f}")
    print(f"MAE  (test): {mae_test:.3f}")
    print(f"R²   (train): {r2_train:.3f}")
    print(f"R²   (test):  {r2_test:.3f}")

    return {
        "rmse_test": rmse_test,
        "mae_test": mae_test,
        "r2_train": r2_train,
        "r2_test": r2_test
    }




# MAIN

if __name__ == "__main__":

    file_path = r"C:\Users\sandy\Desktop\cleaned_output.csv"
    X, y, df = load_data(file_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor, num_ftrs, cat_ftrs = make_preprocessor(X)



   
    # Models
  
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Support Vector Regressor": SVR()
    }

    if XGB_AVAILABLE:
        models["XGBoost"] = xgb.XGBRegressor(
            random_state=42,
            verbosity=0,
            n_jobs=1
        )



    # Hyperparameter grids
   
    param_grids = {
        "Linear Regression": {},

        "Decision Tree": {
            "regressor__max_depth": [5, 10, 20, None],
            "regressor__min_samples_split": [2, 5, 10]
        },

        "Random Forest": {
            "regressor__n_estimators": [100, 200, 300],
            "regressor__max_depth": [10, 20, None],
            "regressor__min_samples_split": [2, 5, 10]
        },

        "Support Vector Regressor": {
            "regressor__C": [0.1, 1, 10],
            "regressor__epsilon": [0.01, 0.1, 0.2],
            "regressor__kernel": ["rbf", "linear"]
        }
    }

    if XGB_AVAILABLE:
        param_grids["XGBoost"] = {
            "regressor__n_estimators": [200, 300],
            "regressor__learning_rate": [0.05, 0.1],
            "regressor__max_depth": [3, 5, 7]
        }



    cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)



    # TUNE MODELS IN PARALLEL
   
    print("\n===== Parallel Hyperparameter Search =====")

    best_estimators = {}
    best_params = {}

    max_workers = min(4, len(models))

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(tune_model, name, model, param_grids[name],
                      preprocessor, cv_strategy, X_train, y_train): name
            for name, model in models.items()
        }

        for future in as_completed(futures):
            name = futures[future]
            try:
                n, best_est, params = future.result()
                best_estimators[n] = best_est
                best_params[n] = params
            except Exception:
                print(f"Failed tuning {name}")



    # EVALUATION
   
    print("\n===== Evaluating Models =====")

    evaluation_results = {}
    for name, est in best_estimators.items():
        evaluation_results[name] = {
            "metrics": evaluate(est, X_train, y_train, X_test, y_test, name),
            "best_params": best_params[name]
        }



   
    # SAVE RESULTS
   
    joblib.dump(evaluation_results, "evaluation_results.pkl")
    print("\nSaved evaluation_results.pkl")

    # pick best model by RMSE
    best_model_name = min(
        evaluation_results,
        key=lambda n: evaluation_results[n]["metrics"]["rmse_test"]
    )

    best_model = best_estimators[best_model_name]
    joblib.dump(best_model, "best_house_price_model.pkl")

    print(f"\nBest model: {best_model_name}")
    print("Saved best_house_price_model.pkl")

    joblib.dump(best_estimators, "all_best_estimators.pkl")
    joblib.dump(best_params, "all_best_params.pkl")

    print("\nSaved all_best_estimators.pkl and all_best_params.pkl")

    print("\nDONE — all models tuned, evaluated, and saved.")