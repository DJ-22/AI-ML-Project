# 🧹 Imports
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# 🎲 Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# 📥 Load dataset (change path if needed)
filename = "winequality-red.csv"
df = pd.read_csv(filename, delimiter=';')
df.columns = df.columns.str.strip()  # Remove any column name whitespace

# 🧪 Feature Engineering: original + custom features
df["total_acidity"] = df["fixed acidity"] + df["volatile acidity"]
df["sugar_per_acid"] = df["residual sugar"] / (df["total_acidity"] + 1e-6)
df["density_alcohol_ratio"] = df["density"] / (df["alcohol"] + 1e-6)
df["sulphates_sulfur_ratio"] = df["sulphates"] / (df["total sulfur dioxide"] + 1e-6)
df["free_total_sulfur_ratio"] = df["free sulfur dioxide"] / (df["total sulfur dioxide"] + 1e-6)
df["acid_density_ratio"] = df["total_acidity"] / (df["density"] + 1e-6)
df["alcohol_sugar_ratio"] = df["alcohol"] / (df["residual sugar"] + 1e-6)
df["acidity_sulphates_interaction"] = df["total_acidity"] * df["sulphates"]

# 🎯 Define features and target
X = df.drop("quality", axis=1)

# 🔄 Convert quality scores to 3-class labels
def group_quality(q):
    if q <= 5:
        return 0  # Low
    elif q == 6:
        return 1  # Medium
    else:
        return 2  # High

y = df["quality"].apply(group_quality)

# 🧠 Encode target labels as integers
y_encoder = LabelEncoder()
y = y_encoder.fit_transform(y)

# 📊 Split data into train/test sets with stratified sampling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 🔍 Feature importance using Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
importances = pd.Series(rf.feature_importances_, index=X_train.columns)

# 🔝 Select top 12 most important features
top_features = importances.sort_values(ascending=False).head(12).index.tolist()
X_train = X_train[top_features]
X_test = X_test[top_features]

# ⚙️ Objective function for Optuna tuning
def objective(trial, model_name):
    if model_name == "xgb":
        model = XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 400),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
            eval_metric='mlogloss',
            random_state=42
        )
    elif model_name == "lgbm":
        model = LGBMClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 400),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
            min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 10, 50),
            verbosity=-1,
            random_state=42
        )
    elif model_name == "cat":
        model = CatBoostClassifier(
            iterations=trial.suggest_int("iterations", 100, 400),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
            depth=trial.suggest_int("depth", 3, 10),
            verbose=0,
            random_seed=42
        )
    # Return average CV accuracy
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy")
    return scores.mean()

# 🔁 Run Optuna optimization for a given model
def run_optuna(model_name):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, model_name), n_trials=50)
    return study.best_params

# 🔍 Tune each model individually
print("Tuning XGBoost...")
xgb_best = run_optuna("xgb")
print("Tuning LightGBM...")
lgbm_best = run_optuna("lgbm")
print("Tuning CatBoost...")
cat_best = run_optuna("cat")

# 🧠 Initialize models with best params
xgb = XGBClassifier(**xgb_best, eval_metric='mlogloss', random_state=42)
lgbm = LGBMClassifier(**lgbm_best, verbosity=-1, random_state=42)
cat = CatBoostClassifier(**cat_best, verbose=0, random_seed=42)

# 🔗 Stacking ensemble with CatBoost as final estimator
stack = StackingClassifier(
    estimators=[('xgb', xgb), ('lgbm', lgbm), ('cat', cat)],
    final_estimator=CatBoostClassifier(verbose=0, random_seed=42),
    passthrough=True
)

# 📈 Cross-validate stacked model
cv_score = cross_val_score(stack, X_train, y_train, cv=5, scoring='accuracy').mean()
print(f"📊 Cross-validated stacking accuracy: {cv_score * 100:.2f}%")

# 🚂 Fit stacking model and predict on test set
stack.fit(X_train, y_train)
preds = stack.predict(X_test)
accuracy = accuracy_score(y_test, preds) * 100
print(f"\n✅ Test Set Stacking Accuracy: {accuracy:.4f}%")

# 💾 Save predictions to Excel
results_df = pd.DataFrame({
    "Actual Quality": y_encoder.inverse_transform(y_test),
    "Predicted Quality": y_encoder.inverse_transform(preds)
})
results_df.to_excel("wine_quality_predictions.xlsx", index=False)
print("\n📁 Predictions saved to 'wine_quality_predictions.xlsx'")

# 📊 Confusion matrix plot
cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Low", "Medium", "High"],
            yticklabels=["Low", "Medium", "High"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("📊 Confusion Matrix")
plt.show()

# 🔥 XGBoost feature importances
xgb.fit(X_train, y_train)
importances = xgb.feature_importances_
feat_imp = pd.Series(importances, index=X_train.columns)
feat_imp.sort_values().plot(kind='barh', figsize=(8, 6), color='teal')
plt.title("🔥 XGBoost Feature Importances")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# 🎯 Classification report plot (precision, recall, F1-score)
report = classification_report(
    y_test, preds,
    target_names=["Low", "Medium", "High"],
    output_dict=True
)
class_acc = pd.DataFrame(report).T.drop(["accuracy", "macro avg", "weighted avg"], errors="ignore")
plt.figure(figsize=(10, 6))
class_acc[["precision", "recall", "f1-score"]].plot(kind='barh', ax=plt.gca(), colormap="Set2")
plt.title("🎯 Class-wise Performance (Precision, Recall, F1-score)")
plt.xlabel("Score")
plt.tight_layout()
plt.show()
