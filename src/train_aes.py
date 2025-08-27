import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, cohen_kappa_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import pearsonr
import numpy as np
import optuna
import argparse
import csv
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path 
import sys


try:
    ROOT_DIR = Path(__file__).parent.parent
except NameError:
    ROOT_DIR = Path.cwd()

DATA_DIR = ROOT_DIR / 'data'
MODELS_DIR = ROOT_DIR / 'models'

MODELS_DIR.mkdir(parents=True, exist_ok=True)
# -------------------------


def load_data(filepath):
    data = pd.read_csv(filepath)
    return data


def preprocess_data(data):
    data = data.dropna()  
    return data


def create_pipeline(params, n_features):
    regressor = XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_estimators=params['n_estimators'],
        learning_rate=params['learning_rate'],
        max_depth=params['max_depth'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        gamma=params['gamma'],
        reg_alpha=params['reg_alpha'], 
        reg_lambda=params['reg_lambda']  
    )
    pipeline = Pipeline([
        ('scaler', StandardScaler()), 
        ('feature_selector', RFE(regressor, n_features_to_select=n_features))  
    ])
    return pipeline, regressor



def objective(trial, X_train, y_train, X_test, y_test, n_features):
    """
    Objective function for Optuna to optimize hyperparameters based on QWK.
    """
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),  
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10) 
    }

    pipeline, regressor = create_pipeline(params, n_features)

    X_train_transformed = pipeline.fit_transform(X_train, y_train)
    X_test_transformed = pipeline.transform(X_test)

    regressor.fit(X_train_transformed, y_train)

    predictions = regressor.predict(X_test_transformed)
    predictions_rounded = np.round(predictions).astype(int)
    y_test_rounded = np.round(y_test).astype(int)
    
    qwk = cohen_kappa_score(y_test_rounded, predictions_rounded, weights='quadratic')
    
    return -qwk

def make_predictions(model, X):
    """
    Make predictions using the trained model.
    """
    return model.predict(X)


def evaluate_model(model, X, y):
    """
    Evaluate the model using RMSE, MAE, Pearson Correlation, QWK, Precision, and Recall.
    """
    predictions = make_predictions(model, X)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    mae = mean_absolute_error(y, predictions)
    pearson_corr, _ = pearsonr(y, predictions)

    predictions_rounded_float = np.round(predictions * 2) / 2
    y_test_float = y.to_numpy()

    y_true_int = (y_test_float * 10).astype(int)
    y_pred_int = (predictions_rounded_float * 10).astype(int)

    qwk = cohen_kappa_score(y_true_int, y_pred_int, weights='quadratic')


    within_tolerance = (np.abs(y_true_int - y_pred_int) <= 10).astype(int)
    tp = np.sum(within_tolerance)
    fn = len(y) - tp
    accuracy_within_1_band = tp / len(y) if len(y) > 0 else 0
    
    print(f"Accuracy (within 1.0 score band): {accuracy_within_1_band:.4f}")
    
    precision = accuracy_within_1_band 
    recall = accuracy_within_1_band 
    return rmse, mae, pearson_corr, qwk, precision, recall, predictions

def predict_grade(prompt, essay_file, best_params, n_features):
    with open(essay_file, 'r') as f:
        essay_text = f.read()

    # Preprocess 
    essay_features = preprocess_essay_text(essay_text)

    pipeline, regressor = create_pipeline(best_params, n_features)

    essay_features_transformed = pipeline.transform(essay_features)

    predicted_grade = regressor.predict(essay_features_transformed)

    print(f'Predicted grade for prompt "{prompt}" and essay "{essay_file}": {predicted_grade:.2f}')
    
def preprocess_essay_text(text):
    """
    Preprocess the text by tokenizing, removing stop words, etc.
    """
    # Example preprocessing steps (customize as needed):
    # Tokenization, removing stop words, etc.
    return text  # Replace with actual preprocessing


def main(args=None):
    # Load extracted features from feature_extract.py
    features = preprocess_data(load_data(DATA_DIR / 'extracted_features.csv')).dropna()
    
    # Feature engineering
    X = features.drop(columns=['score'])
    y = features['score']  

    column_names = ['prompt', 'essay', 'score']
    df_raw = pd.read_csv(DATA_DIR / 'ielts_data.csv', header=None, names=column_names)
    df_raw = df_raw.iloc[:len(features)]
    corpus = df_raw['prompt'].astype(str).tolist() + df_raw['essay'].astype(str).tolist()
    
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(corpus)
    
    # Perform 80-10-10 train-test-validation split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=20)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=20)

    rmse_scores = []
    mae_scores = []
    pearson_scores = []
    qwk_scores = []
    precision_scores = []
    recall_scores = []

    n_features = 56

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val, n_features), n_trials=35, n_jobs=-1)

    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])

    best_params = study.best_params
    pipeline, regressor = create_pipeline(best_params, n_features)
    pipeline.fit(X_train_full, y_train_full)

    X_train_full_transformed = pipeline.transform(X_train_full)
    regressor.fit(X_train_full_transformed, y_train_full)

    # Evaluate the model
    X_test_transformed = pipeline.transform(X_test)
    rmse, mae, pearson_corr, qwk, precision, recall, predictions = evaluate_model(regressor, X_test_transformed, y_test)
    rmse_scores.append(rmse)
    mae_scores.append(mae)
    pearson_scores.append(pearson_corr)
    qwk_scores.append(qwk)
    precision_scores.append(precision)
    recall_scores.append(recall)

    # average metrics
    average_rmse = np.mean(rmse_scores)
    average_mae = np.mean(mae_scores)
    average_pearson = np.mean(pearson_scores)
    average_qwk = np.mean(qwk_scores)
    average_precision = np.mean(precision_scores)
    average_recall = np.mean(recall_scores)

    print(f'Average Root Mean Squared Error: {average_rmse}')
    print(f'Average Mean Absolute Error: {average_mae}')
    print(f'Average Pearson Correlation: {average_pearson}')
    print(f'Average QWK: {average_qwk}')
    print(f'Average Precision: {average_precision}')
    print(f'Average Recall: {average_recall}')
    
    metrics = [
        ("Average Root Mean Squared Error", average_rmse),
        ("Average Mean Absolute Error", average_mae),
        ("Average Pearson Correlation", average_pearson),
        ("Average QWK", average_qwk),
        ("Average Precision", average_precision),
        ("Average Recall", average_recall)
    ]

    # Write to CSV
    results_path = ROOT_DIR / 'xgboost_results.csv'
    with open(results_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(["Metric", "Value"])
        
        # Write metric data
        writer.writerows(metrics)

    print("CSV file saved successfully!")

    artifacts = {
        'pipeline': pipeline,
        'regressor': regressor,
        'tfidf_vectorizer': tfidf_vectorizer,
        'training_columns': X_train_full.columns.tolist() 
    }
    
    joblib.dump(artifacts, MODELS_DIR / 'aes_full_pipeline_artifacts.pkl')
    
    print("Artifacts saved successfully to models/aes_full_pipeline_artifacts.pkl")
    
if __name__ == "__main__":
    main()