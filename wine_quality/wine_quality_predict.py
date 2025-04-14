# Import necessary libraries for data handling, machine learning, and visualization
import os  # For handling file paths
import json  # For working with JSON files
import random  # For setting random seeds and generating random numbers
import joblib  # For saving and loading machine learning models
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting graphs and visualizations
import seaborn as sns  # For statistical data visualization

# Import specific functions and classes from scikit-learn for model training and evaluation
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold  # For data splitting and cross-validation
from sklearn.ensemble import RandomForestClassifier, StackingClassifier  # For Random Forest and Stacking Classifiers
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # For model evaluation metrics
from sklearn.linear_model import LogisticRegression  # For logistic regression in stacking
from sklearn.preprocessing import LabelEncoder  # For encoding categorical labels

# Import machine learning models from popular libraries
from xgboost import XGBClassifier  # For XGBoost model
from lightgbm import LGBMClassifier  # For LightGBM model
from catboost import CatBoostClassifier  # For CatBoost model

# Import SMOTE for handling imbalanced datasets
from imblearn.over_sampling import SMOTE  # For Synthetic Minority Over-sampling Technique

# Import Optuna for hyperparameter optimization
import optuna  # For hyperparameter tuning using Optuna

# Set seed for reproducibility
np.random.seed(42)  # Set NumPy random seed for reproducibility
random.seed(42)  # Set Python random seed for reproducibility

# Load dataset
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get script directory
fname = os.path.join(script_dir, "winequality-red.csv")  # Path to the dataset
df = pd.read_csv(fname, delimiter=';')  # Load the dataset
df.columns = df.columns.str.strip()  # Strip any leading/trailing spaces in column names

# Feature engineering
df["total_acidity"] = df["fixed acidity"] + df["volatile acidity"]  # Calculate total acidity
df["sugar_per_acid"] = df["residual sugar"] / (df["total_acidity"] + 1e-6)  # Sugar to acidity ratio
df["density_alcohol_ratio"] = df["density"] / (df["alcohol"] + 1e-6)  # Density to alcohol ratio
df["sulphates_sulfur_ratio"] = df["sulphates"] / (df["total sulfur dioxide"] + 1e-6)  # Sulphates to sulfur ratio
df["free_total_sulfur_ratio"] = df["free sulfur dioxide"] / (df["total sulfur dioxide"] + 1e-6)  # Free to total sulfur ratio
df["acid_density_ratio"] = df["total_acidity"] / (df["density"] + 1e-6)  # Acidity to density ratio
df["alcohol_sugar_ratio"] = df["alcohol"] / (df["residual sugar"] + 1e-6)  # Alcohol to sugar ratio
df["acidity_sulphates_interaction"] = df["total_acidity"] * df["sulphates"]  # Interaction feature

# Define features and target
X = df.drop("quality", axis=1)  # Features: all columns except 'quality'

# Function to group wine quality into 3 categories (Low, Medium, High)
def group_quality(q):
    return 0 if q <= 5 else 1 if q == 6 else 2  # Low=0, Medium=1, High=2

y = df["quality"].apply(group_quality)  # Apply the quality grouping function
y_encoder = LabelEncoder()  # Initialize label encoder
y = y_encoder.fit_transform(y)  # Encode the target labels as integers

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # Stratified split to maintain class balance
)

# Feature selection using Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)  # Initialize Random Forest
rf.fit(X_train, y_train)  # Fit the model to training data
importances = pd.Series(rf.feature_importances_, index=X_train.columns)  # Get feature importances
top_features = importances.sort_values(ascending=False).head(12).index.tolist()  # Select top 12 features
X_train = X_train[top_features]  # Update training features to only top features
X_test = X_test[top_features]  # Update test features to only top features

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)  # Initialize SMOTE
X_train, y_train = smote.fit_resample(X_train, y_train)  # Resample the training data to balance classes

# Optuna objective function for hyperparameter tuning
def objective(trial, model_name):
    if model_name == "xgb":  # If model is XGBoost
        model = XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 400),  # Number of estimators
            max_depth=trial.suggest_int("max_depth", 3, 10),  # Max depth of trees
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),  # Learning rate
            eval_metric='mlogloss',  # Evaluation metric for XGBoost
            verbosity=0,  # Suppress detailed logging
            random_state=42  # Set random seed for reproducibility
        )
    elif model_name == "lgbm":  # If model is LightGBM
        model = LGBMClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 400),  # Number of estimators
            max_depth=trial.suggest_int("max_depth", 3, 10),  # Max depth of trees
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),  # Learning rate
            min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 10, 50),  # Minimum data in leaf nodes
            verbosity=-1,  # Suppress detailed logging
            random_state=42  # Set random seed for reproducibility
        )
    elif model_name == "cat":  # If model is CatBoost
        model = CatBoostClassifier(
            iterations=trial.suggest_int("iterations", 100, 400),  # Number of iterations
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),  # Learning rate
            depth=trial.suggest_int("depth", 3, 10),  # Depth of trees
            verbose=0,  # Suppress detailed logging
            random_seed=42  # Set random seed for reproducibility
        )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # 3-fold cross-validation

    return cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy").mean()  # Return the mean accuracy

# Run or resume Optuna optimization
def run_optuna(model_name, param_path):
    study = optuna.create_study(direction="maximize")  # Create a new Optuna study for maximizing accuracy

    # If a previous set of parameters exists, load it and resume tuning
    if os.path.exists(param_path):
        with open(param_path, "r") as f:
            best_params = json.load(f)  # Load the previous best parameters
        study.enqueue_trial(best_params)  # Enqueue the previous best parameters as a trial

        # Initialize model with the best parameters
        model_cls = {
            "xgb": lambda **params: XGBClassifier(**params, verbosity=0, eval_metric='mlogloss', random_state=42),
            "lgbm": lambda **params: LGBMClassifier(**params, verbosity=-1, random_state=42),
            "cat": lambda **params: CatBoostClassifier(**params, verbose=0, random_seed=42)
        }[model_name]

        model = model_cls(**best_params)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        prev_score = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy").mean()  # Evaluate previous score
    else:
        prev_score = -1  # No previous best parameters, set score to -1

    study.optimize(lambda trial: objective(trial, model_name), n_trials=50)  # Optimize hyperparameters using Optuna
    new_best = study.best_params  # Get the best parameters after optimization

    model_cls = {
        "xgb": lambda **params: XGBClassifier(**params, verbosity=0, eval_metric='mlogloss', random_state=42),
        "lgbm": lambda **params: LGBMClassifier(**params, verbosity=-1, random_state=42),
        "cat": lambda **params: CatBoostClassifier(**params, verbose=0, random_seed=42)
    }[model_name]

    model = model_cls(**new_best)  # Initialize the model with the new best parameters
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    new_score = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy").mean()  # Evaluate the new score

    # If the new score is better than the previous score, save the new parameters
    if new_score > prev_score:
        with open(param_path, "w") as f:
            json.dump(new_best, f)  # Save the new best parameters
        print(f"Improved accuracy for {model_name}: {prev_score:.4f} → {new_score:.4f}")
        
        return new_best
    else:
        print(f"No improvement for {model_name}: keeping previous best {prev_score:.4f}")
        
        return best_params  # Return the previous best parameters if no improvement

# File paths for saving model and parameters
model_path = os.path.join(script_dir, "stacking_model.pkl")  # Path to save the final stacked model
xgb_path = os.path.join(script_dir, "xgb_best_params.json")  # Path to save XGBoost parameters
lgbm_path = os.path.join(script_dir, "lgbm_best_params.json")  # Path to save LightGBM parameters
cat_path = os.path.join(script_dir, "cat_best_params.json")  # Path to save CatBoost parameters

# Hyperparameter tuning for each model
print("Tuning XGBoost...")
xgb_best = run_optuna("xgb", xgb_path)  # Tune XGBoost model

print("Tuning LightGBM...")
lgbm_best = run_optuna("lgbm", lgbm_path)  # Tune LightGBM model

print("Tuning CatBoost...")
cat_best = run_optuna("cat", cat_path)  # Tune CatBoost model

# Stacking model combining XGBoost, LightGBM, and CatBoost
xgb = XGBClassifier(**xgb_best, verbosity=0, eval_metric='mlogloss', random_state=42)  # XGBoost model
lgbm = LGBMClassifier(**lgbm_best, verbosity=-1, random_state=42)  # LightGBM model
cat = CatBoostClassifier(**cat_best, verbose=0, random_seed=42)  # CatBoost model

# Create stacking classifier
stack = StackingClassifier(
    estimators=[('xgb', xgb), ('lgbm', lgbm), ('cat', cat)],  # Stacked models
    final_estimator=LogisticRegression(max_iter=1000),  # Final estimator: Logistic Regression
    passthrough=False,  # Do not pass the original features
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Cross-validation for stacking
)

print("Training improved stacking model...")
stack.fit(X_train, y_train)  # Train the stacking model on the training data
joblib.dump(stack, model_path)  # Save the trained model
print("Improved stacking model saved to 'stacking_model.pkl'")

# Predictions and evaluation
preds = stack.predict(X_test)  # Predict on the test set
accuracy = accuracy_score(y_test, preds) * 100  # Calculate accuracy
print(f"\nImproved Test Set Stacking Accuracy: {accuracy:.2f}%")

# Save predictions to Excel
results_df = pd.DataFrame({
    "Actual Quality": y_encoder.inverse_transform(y_test),
    "Predicted Quality": y_encoder.inverse_transform(preds)
})
results_df.to_excel("wine_quality_predictions.xlsx", index=False)  # Save to Excel
print("\nPredictions saved to 'wine_quality_predictions.xlsx'")

# Confusion matrix visualization
cm = confusion_matrix(y_test, preds)  # Generate confusion matrix
plt.figure(figsize=(8, 6))  # Set figure size
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",  # Plot heatmap for confusion matrix
            xticklabels=["Low", "Medium", "High"],
            yticklabels=["Low", "Medium", "High"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Feature importances from XGBoost
xgb.fit(X_train, y_train)  # Train XGBoost on the training data
feat_imp = pd.Series(xgb.feature_importances_, index=X_train.columns)  # Get feature importances
feat_imp.sort_values().plot(kind='barh', figsize=(8, 6), color='teal')  # Plot feature importances
plt.title("XGBoost Feature Importances")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# Class-wise performance metrics visualization
report = classification_report(
    y_test, preds,
    target_names=["Low", "Medium", "High"],
    output_dict=True
)  # Generate classification report
class_acc = pd.DataFrame(report).T.drop(["accuracy", "macro avg", "weighted avg"], errors="ignore")  # Extract class-wise metrics
plt.figure(figsize=(10, 6))  # Set figure size
class_acc[["precision", "recall", "f1-score"]].plot(kind='barh', ax=plt.gca(), colormap="Set2")  # Plot metrics
plt.title("Class-wise Performance (Precision, Recall, F1)")
plt.xlabel("Score")
plt.ylabel("Class")
plt.tight_layout()
plt.show()
