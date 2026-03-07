"""
the Feature Engineering class for Home Credit Default Risk dataset.
This class includes methods to create new features, aggregate features from
different datasets, and preprocess the data for modeling.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """Feature Engineering class for Home Credit Default Risk dataset."""
    
    def __init__(self):
        self.label_encoders = {}
        self.scalers = {}
    
    def engineer_application_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for the application data.

        Args:
            df: application data

        Returns:
            Engineered application data
        """
        df = df.copy()
        
        # Create new features
        # 1. Credit to Income Ratio
        df['CREDIT_INCOME_PERCENT'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']

        # 2. Annuity to Income Ratio
        df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']

        # 3. Credit Term
        df['CREDIT_TERM'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']

        # 4. Employment Days Feature (Convert to Positive)
        df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
        df['INCOME_CREDIT_PERCENT'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
        df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
        df['ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
        df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']

        # 5. External Sources Features
        df['EXT_SOURCES_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
        df['EXT_SOURCES_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
        df['EXT_SOURCES_MAX'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].max(axis=1)
        df['EXT_SOURCES_MIN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].min(axis=1)

        # 6. Age Group
        df['AGE_YEARS'] = -df['DAYS_BIRTH'] / 365
        # Create numerical age group feature
        df['AGE_GROUP'] = pd.cut(df['AGE_YEARS'], bins=[0, 25, 35, 50, 65, 100], 
                                labels=[0, 1, 2, 3, 4])
        df['AGE_GROUP'] = df['AGE_GROUP'].astype(float)

        # 7. Employment Duration (Handle Outliers)
        df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace({365243: np.nan})
        df['EMPLOYED_YEARS'] = -df['DAYS_EMPLOYED'] / 365

        # 8. Car and Real Estate Age
        df['CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / df['AGE_YEARS']
        df['CAR_TO_EMPLOYED_RATIO'] = df['OWN_CAR_AGE'] / df['EMPLOYED_YEARS']
        
        return df
    
    def aggregate_bureau_features(self, bureau: pd.DataFrame, 
                                 bureau_balance: pd.DataFrame = None) -> pd.DataFrame:
        """
        Aggregate bureau features

        Args:
            bureau: bureau data
            bureau_balance: bureau balance data

        Returns:
            Aggregated features
        """
        # Basic aggregation
        agg_funcs = {
            'DAYS_CREDIT': ['count', 'mean', 'max', 'min', 'sum'],
            'CREDIT_DAY_OVERDUE': ['mean', 'max'],
            'DAYS_CREDIT_ENDDATE': ['mean', 'max', 'min'],
            'DAYS_ENDDATE_FACT': ['mean', 'max', 'min'],
            'AMT_CREDIT_MAX_OVERDUE': ['mean', 'max'],
            'CNT_CREDIT_PROLONG': ['sum', 'mean'],
            'AMT_CREDIT_SUM': ['sum', 'mean', 'max'],
            'AMT_CREDIT_SUM_DEBT': ['sum', 'mean', 'max'],
            'AMT_CREDIT_SUM_LIMIT': ['sum', 'mean', 'max'],
            'AMT_CREDIT_SUM_OVERDUE': ['sum', 'mean', 'max'],
            'DAYS_CREDIT_UPDATE': ['mean', 'max', 'min']
        }
        
        bureau_agg = bureau.groupby('SK_ID_CURR').agg(agg_funcs)
        bureau_agg.columns = ['BUREAU_' + '_'.join(col).upper() for col in bureau_agg.columns]
        
        # Active credit count
        bureau_active = bureau[bureau['CREDIT_ACTIVE'] == 'Active']
        bureau_active_agg = bureau_active.groupby('SK_ID_CURR').agg({
            'AMT_CREDIT_SUM': 'sum',
            'AMT_CREDIT_SUM_DEBT': 'sum'
        })
        bureau_active_agg.columns = ['BUREAU_ACTIVE_' + col for col in bureau_active_agg.columns]
        
        # Merge
        bureau_features = bureau_agg.merge(bureau_active_agg, left_index=True, 
                                          right_index=True, how='left')
        
        # If there is balance data, add balance features
        if bureau_balance is not None:
            balance_agg = bureau_balance.groupby('SK_ID_BUREAU').agg({
                'MONTHS_BALANCE': ['count', 'mean', 'max', 'min'],
                'STATUS': lambda x: (x == 'C').sum()  # Closed accounts count
            })
            balance_agg.columns = ['BALANCE_' + '_'.join(col).upper() for col in balance_agg.columns]

            # Merge with bureau data
            bureau_with_balance = bureau.merge(balance_agg, left_on='SK_ID_BUREAU',
                                              right_index=True, how='left')
            
            balance_features = bureau_with_balance.groupby('SK_ID_CURR')[
                [col for col in bureau_with_balance.columns if 'BALANCE_' in col]
            ].mean()
            
            bureau_features = bureau_features.merge(balance_features, left_index=True, 
                                                   right_index=True, how='left')
        
        return bureau_features
    
    def aggregate_previous_application_features(self, prev_app: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate previous application features

        Args:
            prev_app: previous application data

        Returns:
            Aggregated features
        """
        # Basic aggregation
        agg_funcs = {
            'AMT_ANNUITY': ['mean', 'max', 'min'],
            'AMT_APPLICATION': ['mean', 'max', 'min'],
            'AMT_CREDIT': ['mean', 'max', 'min'],
            'AMT_DOWN_PAYMENT': ['mean', 'max'],
            'AMT_GOODS_PRICE': ['mean', 'max', 'min'],
            'HOUR_APPR_PROCESS_START': ['mean', 'max', 'min'],
            'RATE_DOWN_PAYMENT': ['mean', 'max', 'min'],
            'DAYS_DECISION': ['mean', 'max', 'min'],
            'CNT_PAYMENT': ['mean', 'max', 'min', 'sum']
        }
        
        prev_agg = prev_app.groupby('SK_ID_CURR').agg(agg_funcs)
        prev_agg.columns = ['PREV_' + '_'.join(col).upper() for col in prev_agg.columns]
        
        # Approval/Refusal statistics
        prev_agg['PREV_APP_COUNT'] = prev_app.groupby('SK_ID_CURR').size()
        prev_agg['PREV_APP_APPROVED'] = prev_app[prev_app['NAME_CONTRACT_STATUS'] == 'Approved'].groupby('SK_ID_CURR').size()
        prev_agg['PREV_APP_REFUSED'] = prev_app[prev_app['NAME_CONTRACT_STATUS'] == 'Refused'].groupby('SK_ID_CURR').size()
        prev_agg['PREV_APP_APPROVAL_RATE'] = prev_agg['PREV_APP_APPROVED'] / prev_agg['PREV_APP_COUNT']
        
        return prev_agg
    
    def aggregate_pos_cash_features(self, pos_cash: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate POS and cash loan features

        Args:
            pos_cash: POS cash data

        Returns:
            Aggregated features
        """
        agg_funcs = {
            'MONTHS_BALANCE': ['count', 'mean', 'max', 'min'],
            'CNT_INSTALMENT': ['mean', 'max', 'min'],
            'CNT_INSTALMENT_FUTURE': ['mean', 'max', 'min'],
            'SK_DPD': ['mean', 'max', 'sum'],
            'SK_DPD_DEF': ['mean', 'max', 'sum']
        }
        
        pos_agg = pos_cash.groupby('SK_ID_CURR').agg(agg_funcs)
        pos_agg.columns = ['POS_' + '_'.join(col).upper() for col in pos_agg.columns]
        
        return pos_agg
    
    def aggregate_credit_card_features(self, credit_card: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate credit card features

        Args:
            credit_card: credit card data

        Returns:
            Aggregated features
        """
        agg_funcs = {
            'MONTHS_BALANCE': ['count', 'mean', 'max', 'min'],
            'AMT_BALANCE': ['mean', 'max', 'min', 'sum'],
            'AMT_CREDIT_LIMIT_ACTUAL': ['mean', 'max', 'min'],
            'AMT_DRAWINGS_ATM_CURRENT': ['mean', 'max', 'min', 'sum'],
            'AMT_DRAWINGS_CURRENT': ['mean', 'max', 'min', 'sum'],
            'AMT_DRAWINGS_OTHER_CURRENT': ['mean', 'max', 'min', 'sum'],
            'AMT_DRAWINGS_POS_CURRENT': ['mean', 'max', 'min', 'sum'],
            'AMT_INST_MIN_REGULARITY': ['mean', 'max', 'min'],
            'AMT_PAYMENT_CURRENT': ['mean', 'max', 'min', 'sum'],
            'AMT_PAYMENT_TOTAL_CURRENT': ['mean', 'max', 'min', 'sum'],
            'AMT_RECEIVABLE_PRINCIPAL': ['mean', 'max', 'min', 'sum'],
            'AMT_RECIVABLE': ['mean', 'max', 'min', 'sum'],
            'AMT_TOTAL_RECEIVABLE': ['mean', 'max', 'min', 'sum'],
            'CNT_DRAWINGS_ATM_CURRENT': ['mean', 'max', 'min', 'sum'],
            'CNT_DRAWINGS_CURRENT': ['mean', 'max', 'min', 'sum'],
            'CNT_DRAWINGS_OTHER_CURRENT': ['mean', 'max', 'min', 'sum'],
            'CNT_DRAWINGS_POS_CURRENT': ['mean', 'max', 'min', 'sum'],
            'CNT_INSTALMENT_MATURE_CUM': ['mean', 'max', 'min'],
            'SK_DPD': ['mean', 'max', 'sum'],
            'SK_DPD_DEF': ['mean', 'max', 'sum']
        }
        
        cc_agg = credit_card.groupby('SK_ID_CURR').agg(agg_funcs)
        cc_agg.columns = ['CC_' + '_'.join(col).upper() for col in cc_agg.columns]
        
        return cc_agg
    
    def aggregate_installments_features(self, installments: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate installments features

        Args:
            installments: installments data

        Returns:
            Aggregated features
        """
        # Create new features
        installments['PAYMENT_PERC'] = installments['AMT_PAYMENT'] / installments['AMT_INSTALMENT']
        installments['PAYMENT_DIFF'] = installments['AMT_INSTALMENT'] - installments['AMT_PAYMENT']
        installments['DPD'] = installments['DAYS_ENTRY_PAYMENT'] - installments['DAYS_INSTALMENT']
        installments['DBD'] = installments['DAYS_INSTALMENT'] - installments['DAYS_ENTRY_PAYMENT']
        
        agg_funcs = {
            'NUM_INSTALMENT_VERSION': ['nunique'],
            'DPD': ['mean', 'max', 'sum'],
            'DBD': ['mean', 'max', 'sum'],
            'PAYMENT_PERC': ['mean', 'max', 'min', 'std'],
            'PAYMENT_DIFF': ['mean', 'max', 'min', 'std'],
            'AMT_INSTALMENT': ['mean', 'max', 'min', 'sum'],
            'AMT_PAYMENT': ['mean', 'max', 'min', 'sum'],
            'DAYS_ENTRY_PAYMENT': ['mean', 'max', 'min']
        }
        
        inst_agg = installments.groupby('SK_ID_CURR').agg(agg_funcs)
        inst_agg.columns = ['INST_' + '_'.join(col).upper() for col in inst_agg.columns]
        
        return inst_agg
    
    def encode_categorical_features(self, df: pd.DataFrame, 
                                  categorical_cols: List[str] = None) -> pd.DataFrame:
        """
        Encode categorical features

        Args:
            df: Input data
            categorical_cols: List of categorical column names

        Returns:
            Encoded data
        """
        df = df.copy()
        
        if categorical_cols is None:
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    # then transform using existing encoder 
                    unique_vals = df[col].unique()
                    known_vals = self.label_encoders[col].classes_
                    new_vals = set(unique_vals) - set(known_vals)
                    
                    if new_vals:
                        # the new values are added to the classes_
                        self.label_encoders[col].classes_ = np.append(known_vals, list(new_vals))
                    
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        return df
    
    def create_all_features(self, datasets: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create all features

        Args:
            datasets: Dictionary containing all datasets

        Returns:
            train_features, test_features: Features after engineering for training and testing sets
        """
        print("the feature engineering starts...")
        
        # 1. the main application features
        train_df = self.engineer_application_features(datasets['application_train'])
        test_df = self.engineer_application_features(datasets['application_test'])

        print(f"the main application features: {train_df.shape[1]}")

        # 2. Aggregate features from other tables
        feature_tables = []

        # Bureau features
        if 'bureau' in datasets and datasets['bureau'] is not None:
            bureau_balance = datasets.get('bureau_balance', None)
            bureau_features = self.aggregate_bureau_features(datasets['bureau'], bureau_balance)
            feature_tables.append(bureau_features)
            print(f"Bureau features: {bureau_features.shape[1]}")

        # Previous application features
        if 'previous_application' in datasets and datasets['previous_application'] is not None:
            prev_features = self.aggregate_previous_application_features(datasets['previous_application'])
            feature_tables.append(prev_features)
            print(f"Previous application features: {prev_features.shape[1]}")

        # POS cash features
        if 'pos_cash' in datasets and datasets['pos_cash'] is not None:
            pos_features = self.aggregate_pos_cash_features(datasets['pos_cash'])
            feature_tables.append(pos_features)
            print(f"POS cash features: {pos_features.shape[1]}")

        # Credit card features
        if 'credit_card' in datasets and datasets['credit_card'] is not None:
            cc_features = self.aggregate_credit_card_features(datasets['credit_card'])
            feature_tables.append(cc_features)
            print(f"Credit card features: {cc_features.shape[1]}")

        # Installments features
        if 'installments' in datasets and datasets['installments'] is not None:
            inst_features = self.aggregate_installments_features(datasets['installments'])
            feature_tables.append(inst_features)
            print(f"Installments features: {inst_features.shape[1]}")
        
        # 3. the merging features
        for features in feature_tables:
            train_df = train_df.merge(features, left_on='SK_ID_CURR', 
                                    right_index=True, how='left')
            test_df = test_df.merge(features, left_on='SK_ID_CURR', 
                                   right_index=True, how='left')

        # 4. Encode categorical features
        categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
        categorical_cols = [col for col in categorical_cols if col != 'SK_ID_CURR']

        # Merge train and test sets for encoding
        combined_df = pd.concat([train_df[categorical_cols], test_df[categorical_cols]], 
                               axis=0, ignore_index=True)
        combined_encoded = self.encode_categorical_features(combined_df, categorical_cols)

        # Separate train and test sets
        train_encoded = combined_encoded.iloc[:len(train_df)]
        test_encoded = combined_encoded.iloc[len(train_df):]

        # Update categorical columns
        for col in categorical_cols:
            train_df[col] = train_encoded[col].values
            test_df[col] = test_encoded[col].values

        print(f"the final feature count: {train_df.shape[1]}")
        print("Feature engineering completed!")

        return train_df, test_df