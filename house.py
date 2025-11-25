# ============================================
# 1. Import Libraries
# ============================================

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


# ============================================
# 2. Load Dataset
# ============================================

df = pd.read_csv(r"C:\Users\sandy\Desktop\cleaned_output.csv")

target = "price"
df = df.dropna(subset=[target])


# ============================================
# 3. Filter: ONLY HOUSES
# ============================================

house_types = ["residence", "bungalow", "master"]

if "property_type_name" not in df.columns:
    raise KeyError("Column 'property_type_name' not found. Cannot filter houses.")

df = df[df["property_type_name"].isin(house_types)]

print("House rows:", df.shape[0])

if df.shape[0] < 5:
    raise ValueError("Not enough house rows to train a model.")


# ============================================
# 4. Limit number of rooms
# ============================================

if "number_rooms" in df.columns:
    df["number_rooms"] = df["number_rooms"].clip(upper=15)


# ============================================
# 5. Define Features
# ============================================

X = df.drop(columns=[target])
y = df[target]

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object', 'category']).columns

print("Numeric:", list(numeric_features))
print("Categorical:", list(categorical_features))


# ============================================
# 6. Train/Test Split
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ============================================
# 7. Preprocessing
# ============================================

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])


# ============================================
# 8. Models
# ============================================

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Support Vector Regressor": SVR(),
    "XGBoost": xgb.XGBRegressor(random_state=42, verbosity=0)
}

param_grids = {
    "Linear Regression": {},
    "Decision Tree": {
        "regressor__max_depth": [5, 10, 20, None],
        "regressor__min_samples_split": [2, 5, 10]
    },
    "Random Forest": {
        "regressor__n_estimators": [100, 200, 300],
        "regressor__max_depth": [10, 20, None],
        "regressor__min_samples_split": [2, 5]
    },
    "Support Vector Regressor": {
        "regressor__C": [0.1, 1, 10],
        "regressor__epsilon": [0.01, 0.1],
        "regressor__kernel": ["rbf", "linear"]
    },
    "XGBoost": {
        "regressor__n_estimators": [200, 300],
        "regressor__learning_rate": [0.05, 0.1],
        "regressor__max_depth": [3, 5]
    }
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)


# ============================================
# 9. Training
# ============================================

pipelines = {}

for name, model in models.items():
    print(f"\nTraining {name}...")

    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("regressor", model)
    ])

    param_grid = param_grids[name]

    if name in ["Random Forest", "XGBoost"]:
        search = RandomizedSearchCV(
            pipe, param_grid, n_iter=10,
            scoring='neg_root_mean_squared_error',
            cv=cv, random_state=42, n_jobs=-1
        )
    else:
        search = GridSearchCV(
            pipe, param_grid,
            scoring='neg_root_mean_squared_error',
            cv=cv, n_jobs=-1
        )

    search.fit(X_train, y_train)
    pipelines[name] = search.best_estimator_

    print("Best params:", search.best_params_)


# ============================================
# 10. Evaluation
# ============================================

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


for name, mdl in pipelines.items():
    evaluate(mdl, X_test, y_test, name)


