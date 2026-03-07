"""
the Model Trainer Module
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings
import mlflow
warnings.filterwarnings('ignore')


class ModelTrainer:
    """the Model Trainer Class"""
    
    def __init__(self, n_folds: int = 5, random_state: int = 42):
        """
        the Model Trainer Initialization
        
        Args:
            n_folds: the number of folds for cross-validation
            random_state: the random seed
        """
        self.n_folds = n_folds
        self.random_state = random_state
        self.models = {}
        self.cv_scores = {}
        self.feature_importance = {}
        
    def prepare_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                     target_col: str = 'TARGET') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        the Data Preparation
        
        Args:
            train_df: the training data
            test_df: the testing data
            target_col: the target column name

        Returns:
            X_train, y_train, X_test: the features and labels
        """
        # the Feature Selection
        feature_cols = [col for col in train_df.columns 
                       if col not in ['SK_ID_CURR', target_col]]
        
        X_train = train_df[feature_cols].copy()
        y_train = train_df[target_col]
        X_test = test_df[feature_cols].copy()
        
        # the Handle Missing Values
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)

        # the Handle Missing Values
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns

        # the Fill Missing Values for Numeric Columns
        if len(numeric_cols) > 0:
            X_train[numeric_cols] = X_train[numeric_cols].fillna(X_train[numeric_cols].median())
            X_test[numeric_cols] = X_test[numeric_cols].fillna(X_train[numeric_cols].median())

        # the Fill Missing Values for Categorical Columns
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                mode_value = X_train[col].mode()
                if len(mode_value) > 0:
                    X_train[col] = X_train[col].fillna(mode_value[0])
                    X_test[col] = X_test[col].fillna(mode_value[0])
                else:
                    X_train[col] = X_train[col].fillna('Unknown')
                    X_test[col] = X_test[col].fillna('Unknown')

        print(f"the training set shape: {X_train.shape}")
        print(f"the testing set shape: {X_test.shape}")
        print(f"the numeric features: {len(numeric_cols)}")
        print(f"the categorical features: {len(categorical_cols)}")
        print(f"the positive class ratio: {y_train.mean():.4f}")

        return X_train.values, y_train.values, X_test.values
    

    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray, 
                      X_test: np.ndarray, params: Dict = None) -> Dict:
        """
        the Train XGBoost Model
        
        Args:
            X_train: the training features
            y_train: the training labels
            X_test: the testing features
            params: the model parameters
            
        Returns:
            the model results dictionary
        """
        print("the Train XGBoost Model...")
        
        if params is None:
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'eta': 0.05,
                'max_depth': 6,
                'min_child_weight': 1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.random_state,
                'verbosity': 0
            }

        # the Create Datasets
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, 
                             random_state=self.random_state)
        
        cv_scores = []
        oof_predictions = np.zeros(len(X_train))
        test_predictions = np.zeros(len(X_test))
        feature_importance = {}
        models = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            print(f"the training fold {fold + 1} ...")

            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

            # the Create Datasets
            train_data = xgb.DMatrix(X_fold_train, label=y_fold_train)
            val_data = xgb.DMatrix(X_fold_val, label=y_fold_val)

            # the Train Model
            mlflow.xgboost.autolog()
            model = xgb.train(
                params,
                train_data,
                evals=[(train_data, 'train'), (val_data, 'val')],
                num_boost_round=10000,
                early_stopping_rounds=200,
                verbose_eval=0
            )

            # the Predict
            oof_predictions[val_idx] = model.predict(val_data)
            test_predictions += model.predict(xgb.DMatrix(X_test)) / self.n_folds

            # the Calculate AUC
            fold_auc = roc_auc_score(y_fold_val, oof_predictions[val_idx])
            cv_scores.append(fold_auc)
            print(f"the training fold {fold + 1} AUC: {fold_auc:.6f}")

            # the Feature Importance
            fold_importance = model.get_score(importance_type='gain')
            for feature, importance in fold_importance.items():
                if feature in feature_importance:
                    feature_importance[feature] += importance / self.n_folds
                else:
                    feature_importance[feature] = importance / self.n_folds
            
            models.append(model)

        # the Calculate OOF AUC
        oof_auc = roc_auc_score(y_train, oof_predictions)
        print(f"OOF AUC: {oof_auc:.6f}")
        print(f"CV AUC: {np.mean(cv_scores):.6f} +/- {np.std(cv_scores):.6f}")

        # the Save Results
        result = {
            'models': models,
            'oof_predictions': oof_predictions,
            'test_predictions': test_predictions,
            'cv_scores': cv_scores,
            'oof_auc': oof_auc,
            'feature_importance': feature_importance
        }
        
        self.models['xgboost'] = result
        self.cv_scores['xgboost'] = cv_scores
        self.feature_importance['xgboost'] = feature_importance
        
        return result
    
    
    def ensemble_models(self, models_to_ensemble: List[str] = None) -> Dict:
        """
        the Ensemble multiple models

        Args:
            models_to_ensemble: the list of model names to ensemble

        Returns:
            the ensemble model results
        """
        if models_to_ensemble is None:
            models_to_ensemble = list(self.models.keys())
        
        print(f"the Ensemble models: {models_to_ensemble}")

        # the Get OOF predictions and test predictions
        oof_predictions = []
        test_predictions = []
        
        for model_name in models_to_ensemble:
            if model_name in self.models:
                oof_predictions.append(self.models[model_name]['oof_predictions'])
                test_predictions.append(self.models[model_name]['test_predictions'])

        # the Simple Average Ensemble
        ensemble_oof = np.mean(oof_predictions, axis=0)
        ensemble_test = np.mean(test_predictions, axis=0)
        
        return {
            'oof_predictions': ensemble_oof,
            'test_predictions': ensemble_test
        }
    
    def get_feature_importance_df(self, model_name: str, feature_names: List[str]) -> pd.DataFrame:
        """
        the Get Feature Importance DataFrame

        Args:
            model_name: the model name
            feature_names: the list of feature names

        Returns:
            the feature importance DataFrame
        """
        if model_name not in self.feature_importance:
            print(f"the model {model_name} not found or has no feature importance.")
            return None
        
        importance_data = self.feature_importance[model_name]

        # the Process different formats of feature importance
        if isinstance(importance_data, dict):
            # XGBoost format - dict
            features = []
            importances = []
            for i, feature_name in enumerate(feature_names):
                # XGBoost feature names are like 'f0', 'f1', ...
                xgb_feature_name = f'f{i}'
                if xgb_feature_name in importance_data:
                    features.append(feature_name)
                    importances.append(importance_data[xgb_feature_name])
                else:
                    features.append(feature_name)
                    importances.append(0.0)
            
            importance_df = pd.DataFrame({
                'feature': features,
                'importance': importances
            })
        else:
            # LightGBM/CatBoost format - array-like
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance_data
            })
        
        return importance_df.sort_values('importance', ascending=False)
    
    def create_submission(self, test_predictions: np.ndarray, 
                         test_ids: np.ndarray, filename: str) -> None:
        """
        the Create Submission File
        
        Args:
            test_predictions: the test set predictions
            test_ids: the test set IDs
            filename: the filename to save
        """
        import os
        
        submission = pd.DataFrame({
            'SK_ID_CURR': test_ids,
            'TARGET': test_predictions
        })

        # the Get project root directory path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        submissions_dir = os.path.join(project_root, 'submissions')

        # the Ensure submissions directory exists
        os.makedirs(submissions_dir, exist_ok=True)

        # the Save file
        filepath = os.path.join(submissions_dir, filename)
        submission.to_csv(filepath, index=False)
        print(f"the submission file has been saved: {filepath}")