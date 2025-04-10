# Wine Quality Prediction Model

## How to Use
Run `pip install --user -r requirements.txt`

Simple put the .csv file into the same directory as the .py file and run the code.

## Working
The code will use 3 different machine learning models, **XGBoost**, **LightGBM**, **CatBoost** and then combines them together and tunes their output using **Optuna** to give the best results.