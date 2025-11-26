import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import joblib

# ------------------------------
# 1. Load dataset
# ------------------------------
file_path = r"C:\Users\sandy\Desktop\cleaned_output.csv"
df = pd.read_csv(file_path)

# Remove useless columns
columns_to_drop = ["property_id", "url", "postal_code"]
df = df.drop(columns=[c for c in columns_to_drop if c in df.columns])

target = "price"
df = df.dropna(subset=[target])

X = df.drop(columns=[target])
y = df[target]

# Identify numeric & categorical columns
numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

# ------------------------------
# 2. Preprocessing Fix: Proper Categorical Encoding
# ------------------------------
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
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ],
    remainder="drop"
)

# ------------------------------
# 3. Models to evaluate
# ------------------------------
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42)
}

results = {}

# ------------------------------
# 4. Train/test
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

for name, model in models.items():
    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])
    
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    results[name] = preds

scores = {name: r2_score(y_test, results[name]) for name in results}
best_model_name = max(scores, key=scores.get)

print("\nScores:", scores)
print("Best model:", best_model_name)

# ------------------------------
# 5. Retrain on full dataset & save
# ------------------------------
best_model = models[best_model_name]

final_pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", best_model)
])

final_pipeline.fit(X, y)
joblib.dump(final_pipeline, "best_house_price_model.pkl")

print("\nSaved: best_house_price_model.pkl")
