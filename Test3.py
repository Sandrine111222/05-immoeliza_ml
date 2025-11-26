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
import joblib


# ===============================
# 📥 Load dataset
# ===============================

file_path = r"C:\Users\sandy\Desktop\cleaned_output.csv"
df = pd.read_csv(file_path)

target = "price"
df = df.dropna(subset=[target])

X = df.drop(columns=[target])
y = df[target]


# ===============================
# 🧹 Clean custom values
# ===============================

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


# ===============================
# 🔧 Preprocessing pipelines
# ===============================

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# ===============================
# 🤖 Model definitions
# ===============================

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42, n_jobs=10),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Support Vector Regressor": SVR(),
    "XGBoost": xgb.XGBRegressor(
        n_estimators=300,
        random_state=42,
        verbosity=0,
        n_jobs=10      # full core power
    )
}


# ===============================
# 🎯 Hyperparameter search grids
# ===============================

param_grids = {
    "Linear Regression": {},

    "Decision Tree": {
        "regressor__criterion": ["squared_error", "friedman_mse"],
        "regressor__max_depth": [5, 10, 20, 30, None],
        "regressor__min_samples_split": [2, 5, 10, 20],
        "regressor__min_samples_leaf": [1, 2, 4, 6],
        "regressor__max_features": ["auto", "sqrt", "log2", None]
    },

    "Random Forest": {
        "regressor__n_estimators": [200, 300, 500, 800],
        "regressor__max_depth": [10, 20, 30, 40, None],
        "regressor__min_samples_split": [2, 5, 10, 20],
        "regressor__min_samples_leaf": [1, 2, 4],
        "regressor__max_features": ["auto", "sqrt", "log2"],
        "regressor__bootstrap": [True, False]
    },

    "Support Vector Regressor": {
        "regressor__kernel": ["rbf", "linear", "poly"],
        "regressor__C": [0.1, 1, 10, 50, 100],
        "regressor__epsilon": [0.01, 0.05, 0.1, 0.2],
        "regressor__gamma": ["scale", "auto"],
        "regressor__degree": [2, 3, 4]
    },

    "XGBoost": {
        "regressor__n_estimators": [300, 500, 800],
        "regressor__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "regressor__max_depth": [3, 5, 7, 10],
        "regressor__subsample": [0.6, 0.8, 1.0],
        "regressor__colsample_bytree": [0.6, 0.8, 1.0],
        "regressor__gamma": [0, 1, 5],
        "regressor__reg_alpha": [0, 0.1, 1],
        "regressor__reg_lambda": [1, 2, 5]
    }
}


cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)



# ===============================
# 🔎 Hyperparameter tuning
# ===============================

pipelines = {}
best_params = {}

print("\n===== Running Hyperparameter Tuning =====")

for name, model in models.items():
    print(f"\n🔍 Tuning {name}...")

    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("regressor", model)
    ])

    param_grid = param_grids[name]

    # Faster search for large grids
    if name in ["Random Forest", "XGBoost", "Support Vector Regressor"]:
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_grid,
            cv=cv_strategy,
            n_iter=20,
            scoring='neg_root_mean_squared_error',
            n_jobs=10,
            verbose=1,
            random_state=42
        )
    else:
        search = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            cv=cv_strategy,
            scoring='neg_root_mean_squared_error',
            n_jobs=10,
            verbose=1
        )

    search.fit(X_train, y_train)

    pipelines[name] = search.best_estimator_
    best_params[name] = search.best_params_

    print(f"Best Parameters for {name}:\n{search.best_params_}\n")



# ===============================
# 📊 Evaluation
# ===============================

def evaluate(model, X_test, y_test, name):
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"\n===== {name} =====")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE:  {mae:.3f}")
    print(f"R²:   {r2:.3f}")

    return preds


results = {}

for name, pipe in pipelines.items():
    results[name] = evaluate(pipe, X_test, y_test, name)


# ===============================
# 💾 Save best model
# ===============================

best_model_name = max(results, key=lambda k: r2_score(y_test, results[k]))
best_model = pipelines[best_model_name]

joblib.dump(best_model, "best_house_price_model.pkl")
print("\nModel saved as 'best_house_price_model.pkl'")
print(f"🏆 Best model: {best_model_name}")
