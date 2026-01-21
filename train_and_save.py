import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("diabetes_dataset.csv")

y = df["diabetes"]
X = df.drop(columns=["diabetes"])

num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object", "bool"]).columns

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ]
)

# Balanced = catches more diabetes cases
model = LogisticRegression(max_iter=1000, class_weight="balanced")

clf = Pipeline(steps=[("preprocess", preprocess),
                     ("model", model)])

# Train on full data (for demo)
clf.fit(X, y)

# Save model to file
joblib.dump(clf, "diabetes_model.pkl")

print("✅ Model saved as diabetes_model.pkl")
print("Features needed:", list(X.columns))
