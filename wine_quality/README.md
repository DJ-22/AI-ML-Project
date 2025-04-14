
# Wine Quality Prediction Model

## How to Use

First, install all required packages using:

```bash
pip install --user -r requirements.txt
```

Make sure you have Python 3.8+ installed.

## Setup

1. Place the `winequality-red.csv` dataset file in the **same directory** as the `.py` script.
2. Run the python script.

## Working

This project predicts the quality of red wine based on physicochemical features using advanced machine learning techniques.

### Features

- **Stacked Ensemble Model** using:

  - **XGBoost**
  - **LightGBM**
  - **CatBoost**
- **Optuna Hyperparameter Optimization** across base learners and the final stacking classifier
- **SMOTE** for handling class imbalance
- **Cross-Validation** to prevent overfitting
- **Feature Engineering**, Scaling, and Selection
- **Confusion Matrix** and classification report visualization
- **Optuna Resume Support** to continue tuning from the best saved parameters

## Concepts Used

#### **Stacking Classifier**

Combines predictions from multiple machine learning models (XGBoost, LightGBM, and CatBoost) to improve accuracy. The predictions are fed into a final **Logistic Regression** model to optimize the results.

#### **Optuna for Hyperparameter Optimization**

Optuna is used to automatically tune the hyperparameters of each individual model (XGBoost, LightGBM, and CatBoost) to achieve the best results.

#### **SMOTE for Handling Class Imbalance**

SMOTE (Synthetic Minority Over-sampling Technique) is used to balance the dataset by creating synthetic examples of the minority classes.

#### **Cross-validation and Model Evaluation**

Stratified K-fold cross-validation is used to evaluate the models on different data splits, ensuring that the evaluation process is robust and avoids overfitting.

## Output

- Displays accuracy, precision, recall, and F1-score for each class.
- Saves Optuna tuning results to `optuna_results.xlsx`.
- Shows confusion matrix heatmap and prediction count bar chart.
- Automatically resumes previous Optuna session if available.

## Files Generated

- **stacking_model.pkl**: The trained stacked model.
- **wine_quality_predictions.xlsx**: A file containing the predictions for the test dataset along with the actual labels.
- **optuna_results.xlsx**: A file containing the results of the Optuna optimization process.
- **Confusion Matrix**: A heatmap of the confusion matrix for model evaluation.

## Notes

- The model aims to achieve **over 75% accuracy** on the wine quality classification task.
- You can modify the number of Optuna trials in the script to adjust the depth of optimization.
- The script is tailored for the **red wine dataset** (`winequality-red.csv`).
- If you have any issues or questions, feel free to open an issue or pull request in the GitHub repository.

---
