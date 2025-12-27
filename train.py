# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sb
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn import metrics
# from sklearn.svm import SVC
# from xgboost import XGBClassifier
# from sklearn.linear_model import LogisticRegression
# from imblearn.over_sampling import RandomOverSampler
# from sklearn.metrics import ConfusionMatrixDisplay, classification_report
# from sklearn.impute import SimpleImputer
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# import joblib
# import os

# # Suppress warnings
# import warnings
# warnings.filterwarnings('ignore')

# # Ensure working directory is set to Kaggle's working directory
# working_dir = '/kaggle/working'
# os.makedirs(working_dir, exist_ok=True)

# def load_and_preprocess_data(file_path):
#     # Load the data
#     df = pd.read_csv(file_path)
    
#     # Replace values
#     df = df.replace({'yes':1, 'no':0, '?':'Others', 'others':'Others'})
    
#     # Age conversion function
#     def convertAge(age):
#         if age < 4:
#             return 'Toddler'
#         elif age < 12:
#             return 'Kid'
#         elif age < 18:
#             return 'Teenager'
#         elif age < 40:
#             return 'Young'
#         else:
#             return 'Senior'
    
#     # Feature engineering
#     df['ageGroup'] = df['age'].apply(convertAge)
    
#     def add_feature(data):
#         # Creating a column with sum of scores
#         data['sum_score'] = data.loc[:,'A1_Score':'A10_Score'].sum(axis=1)
        
#         # Creating an indicator feature
#         data['ind'] = data['austim'] + data['used_app_before'] + data['jaundice']
        
#         return data
    
#     df = add_feature(df)
    
#     # Log transformation of age
#     df['age'] = np.log(df['age'])
    
#     # Label encoding
#     def encode_labels(data):
#         encoders = {}
#         for col in data.columns:
#             if data[col].dtype == 'object':
#                 le = LabelEncoder()
#                 data[col] = le.fit_transform(data[col])
#                 encoders[col] = le
#         return data, encoders

    
#     df,encoders = encode_labels(df)
    
#     return df,encoders

# def prepare_data(df):
#     # Remove unnecessary columns
#     removal = ['ID', 'age_desc', 'used_app_before', 'austim']
#     features = df.drop(removal + ['Class/ASD'], axis=1)
#     target = df['Class/ASD']
    
#     # Split the data
#     X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size=0.2, random_state=10)
    
#     # Impute missing values
#     imputer = SimpleImputer(strategy='mean')
#     X_train_imputed = imputer.fit_transform(X_train)
#     X_val_imputed = imputer.transform(X_val)
    
#     # Oversample minority class
#     ros = RandomOverSampler(sampling_strategy='minority', random_state=0)
#     X_resampled, Y_resampled = ros.fit_resample(X_train_imputed, Y_train)
    
#     # Scale features
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X_resampled)
#     X_val_scaled = scaler.transform(X_val_imputed)
    
#     return X_scaled, Y_resampled, X_val_scaled, Y_val, scaler, X_train.columns

# # def train_and_save_models(X, Y, X_val, Y_val):
# #     # Define models
# #     models = {
# #         'logistic_regression': LogisticRegression(max_iter=1000),
# #         'xgboost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
# #         'svm': SVC(kernel='rbf', probability=True)
# #     }
    
# #     results = {}
# #     for name, model in models.items():
# #         # Train model
# #         model.fit(X, Y)

# #         # Train predictions
# #         train_pred = model.predict(X)
# #         train_probs = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X)

# #         # Validation predictions
# #         val_pred = model.predict(X_val)
# #         val_probs = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_val)

# #         # Predict probabilities (for AUC)
# #         if hasattr(model, "predict_proba"):
# #             val_probs = model.predict_proba(X_val)[:, 1]
# #         else:
# #             val_probs = model.decision_function(X_val)

# #         # Store metrics
# #         results[name] = {
# #             # Train metrics
# #             'train_accuracy': accuracy_score(Y, train_pred),
# #             'train_precision': precision_score(Y, train_pred),
# #             'train_recall': recall_score(Y, train_pred),
# #             'train_f1': f1_score(Y, train_pred),
# #             'train_auc': roc_auc_score(Y, train_probs),
# #             'train_report': classification_report(Y, train_pred),

# #             # Validation metrics
# #             'val_accuracy': accuracy_score(Y_val, val_pred),
# #             'val_precision': precision_score(Y_val, val_pred),
# #             'val_recall': recall_score(Y_val, val_pred),
# #             'val_f1': f1_score(Y_val, val_pred),
# #             'val_auc': roc_auc_score(Y_val, val_probs),
# #             'val_report': classification_report(Y_val, val_pred),

# #             # Trained model
# #             'model': model
# #         }

# #         # Save model
# #         joblib.dump(model, os.path.join(working_dir, f'{name}_model.joblib'))

# #         # 🔥 Save confusion matrix plot for each model
# #         plt.figure(figsize=(6, 5))
# #         ConfusionMatrixDisplay.from_estimator(
# #             model, X_val, Y_val,
# #             display_labels=["No ASD", "ASD"],
# #             cmap="Blues",
# #             values_format="d"
# #         )
# #         plt.title(f'Confusion Matrix - {name.upper()}')
# #         plt.tight_layout()
# #         plt.savefig(os.path.join(working_dir, f'{name}_confusion_matrix.png'))
# #         plt.close()

# #     return results
# def train_and_save_models(X, Y, X_val, Y_val):
#     # Define models
#     models = {
#         'logistic_regression': LogisticRegression(max_iter=1000),
#         'xgboost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
#         'svm': SVC(kernel='rbf', probability=True)
#     }
    
#     results = {}

#     # --- Helper for best threshold ---
#     from sklearn.metrics import precision_recall_curve

#     def find_best_threshold(y_true, y_probs):
#         precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
#         f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
#         best_idx = f1_scores.argmax()
#         return thresholds[best_idx], f1_scores[best_idx]

#     for name, model in models.items():
#         # Train model
#         model.fit(X, Y)

#         # --- Training predictions ---
#         train_pred = model.predict(X)
#         train_probs = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X)

#         # --- Validation predictions (raw probs) ---
#         val_probs = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_val)

#         # --- Find best threshold ---
#         best_thresh, best_f1 = find_best_threshold(Y_val, val_probs)
#         val_pred_custom = (val_probs >= best_thresh).astype(int)

#         # --- Store metrics ---
#         results[name] = {
#             # Train metrics
#             'train_accuracy': accuracy_score(Y, train_pred),
#             'train_precision': precision_score(Y, train_pred),
#             'train_recall': recall_score(Y, train_pred),
#             'train_f1': f1_score(Y, train_pred),
#             'train_auc': roc_auc_score(Y, train_probs),
#             'train_report': classification_report(Y, train_pred),

#             # Validation metrics (with tuned threshold)
#             'val_accuracy': accuracy_score(Y_val, val_pred_custom),
#             'val_precision': precision_score(Y_val, val_pred_custom),
#             'val_recall': recall_score(Y_val, val_pred_custom),
#             'val_f1': f1_score(Y_val, val_pred_custom),
#             'val_auc': roc_auc_score(Y_val, val_probs),
#             'val_report': classification_report(Y_val, val_pred_custom),

#             # Best threshold info
#             'best_threshold': best_thresh,
#             'best_f1': best_f1,

#             # Trained model
#             'model': model
#         }

#         # Save model
#         joblib.dump(model, os.path.join(working_dir, f'{name}_model.joblib'))

#         # 🔥 Save confusion matrix plot (with tuned threshold)
#         plt.figure(figsize=(6, 5))
#         ConfusionMatrixDisplay.from_predictions(
#             Y_val, val_pred_custom,
#             display_labels=["No ASD", "ASD"],
#             cmap="Blues",
#             values_format="d"
#         )
#         plt.title(f'Confusion Matrix - {name.upper()} (Threshold={best_thresh:.2f})')
#         plt.tight_layout()
#         plt.savefig(os.path.join(working_dir, f'{name}_confusion_matrix.png'))
#         plt.close()

#     return results

# def main():
#     # Load and preprocess data
#     if os.path.exists("/kaggle/input/autismprediction/train.csv"):
#         train_path = "/kaggle/input/autismprediction/train.csv"   # Kaggle path
#     else:
#         train_path = r"E:/MyProjects/Sumanth/autism-prediction/data/train.csv"  # Local path

#     df,encoders = load_and_preprocess_data(train_path)
    
#     # Prepare data
#     X, Y, X_val, Y_val, scaler, feature_columns = prepare_data(df)
    
#     # Save scaler, feature columns, and label encoder
#     joblib.dump(scaler, os.path.join(working_dir, 'feature_scaler.joblib'))
#     joblib.dump(feature_columns.tolist(), os.path.join(working_dir, 'feature_columns.joblib'))
#     joblib.dump(encoders, os.path.join(working_dir, 'label_encoders.joblib'))

#     # Train and save models
#     results = train_and_save_models(X, Y, X_val, Y_val)
    
#     # Print results
#     for model_name, model_results in results.items():
#         print(f"\n{model_name.upper()} Model Results:")
#         print(f"Training AUC: {model_results['train_auc']}")
#         print(f"Validation AUC: {model_results['val_auc']}")
#         print("\nTraining Classification Report:")
#         print(model_results['train_report'])
#         print("\nValidation Classification Report:")
#         print(model_results['val_report'])
    
#     # Visualize Confusion Matrix for Logistic Regression
#     plt.figure(figsize=(8,6))
#     ConfusionMatrixDisplay.from_estimator(results['logistic_regression']['model'], X_val, Y_val)
#     plt.title('Confusion Matrix - Logistic Regression')
#     plt.tight_layout()
#     plt.savefig(os.path.join(working_dir, 'confusion_matrix.png'))
#     plt.close()

# if __name__ == '__main__':
#     main()


# new updated code

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

working_dir = './'
os.makedirs(working_dir, exist_ok=True)


# ----------------------------
# Load & Preprocess Data
# ----------------------------
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Replace values
    df = df.replace({'yes': 1, 'no': 0, '?': 'Others', 'others': 'Others'})

    # Age group
    def convertAge(age):
        if age < 4: return 'Toddler'
        elif age < 12: return 'Kid'
        elif age < 18: return 'Teenager'
        elif age < 40: return 'Young'
        else: return 'Senior'

    df['ageGroup'] = df['age'].apply(convertAge)

    # Add engineered features
    df['sum_score'] = df.loc[:, 'A1_Score':'A10_Score'].sum(axis=1)
    df['ind'] = df['austim'] + df['used_app_before'] + df['jaundice']

    # Log transform age
    df['age'] = np.log(df['age'])

    # Encode categorical features
    encoders = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

    return df, encoders


# ----------------------------
# Prepare Features & Target
# ----------------------------
def prepare_data(df):
    removal = ['ID', 'age_desc', 'used_app_before', 'austim']
    features = df.drop(removal + ['Class/ASD'], axis=1)
    target = df['Class/ASD']

    X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size=0.2, random_state=10)

    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)

    ros = RandomOverSampler(sampling_strategy='minority', random_state=0)
    X_resampled, Y_resampled = ros.fit_resample(X_train, Y_train)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_resampled)
    X_val_scaled = scaler.transform(X_val)

    return X_scaled, Y_resampled, X_val_scaled, Y_val, scaler, features.columns


# ----------------------------
# Train Models & Save
# ----------------------------
def train_and_save_models(X, Y, X_val, Y_val):
    models = {
        'logistic_regression': LogisticRegression(max_iter=1000),
        'xgboost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        'svm': SVC(kernel='rbf', probability=True)
    }

    results = {}
    for name, model in models.items():
        model.fit(X, Y)

        val_pred = model.predict(X_val)
        val_probs = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_val)

        results[name] = {
            'val_accuracy': accuracy_score(Y_val, val_pred),
            'val_f1': f1_score(Y_val, val_pred),
            'val_auc': roc_auc_score(Y_val, val_probs),
            'val_report': classification_report(Y_val, val_pred),
            'model': model
        }

        # 🔹 Save with consistent names
        if name == 'logistic_regression':
            joblib.dump(model, os.path.join(working_dir, 'logistic_regression_model.joblib'))
        elif name == 'xgboost':
            joblib.dump(model, os.path.join(working_dir, 'xgboost_model.joblib'))
        elif name == 'svm':
            joblib.dump(model, os.path.join(working_dir, 'svm_model.joblib'))

        # Confusion matrix image
        plt.figure(figsize=(6, 5))
        ConfusionMatrixDisplay.from_estimator(
            model, X_val, Y_val,
            display_labels=["No ASD", "ASD"],
            cmap="Blues",
            values_format="d"
        )
        plt.title(f'Confusion Matrix - {name.upper()}')
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f'{name}_confusion_matrix.png'))
        plt.close()

    return results


# ----------------------------
# Main
# ----------------------------
def main():
    if os.path.exists("./data/train.csv"):
        train_path = "./data/train.csv"
    else:
        raise FileNotFoundError("train.csv not found!")

    df, encoders = load_and_preprocess_data(train_path)
    X, Y, X_val, Y_val, scaler, feature_columns = prepare_data(df)

    # 🔹 Save preprocessing artifacts
    joblib.dump(scaler, os.path.join(working_dir, 'feature_scaler.joblib'))
    joblib.dump(feature_columns.tolist(), os.path.join(working_dir, 'feature_columns.joblib'))
    joblib.dump(encoders, os.path.join(working_dir, 'label_encoders.joblib'))

    results = train_and_save_models(X, Y, X_val, Y_val)

    for model_name, res in results.items():
        print(f"\n{model_name.upper()} Validation Results:")
        print(f"Accuracy: {res['val_accuracy']:.3f}, F1: {res['val_f1']:.3f}, AUC: {res['val_auc']:.3f}")
        print(res['val_report'])


if __name__ == '__main__':
    main()
