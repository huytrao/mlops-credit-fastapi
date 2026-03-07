"""
# the code below in order to load and preprocess the Home Credit Default Risk dataset.
# It includes functions to load each individual data file, reduce memory usage,
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class HomeCreditDataLoader:
    """Home Credit Data Loader"""
    
    def __init__(self, data_path: str = "data/raw"):
        """
        Initialize the data loader
        
        Args:
            data_path: Path to the raw data
        """
        self.data_path = data_path
        self.datasets = {}
        
    def load_application_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load the main application data
        
        Returns:
            train_df, test_df: training and testing datasets
        """
        print("Loading application data...")
        
        try:
            train_df = pd.read_csv(os.path.join(self.data_path, "application_train.csv"))
            test_df = pd.read_csv(os.path.join(self.data_path, "application_test.csv"))
            
            print(f" check train shape : {train_df.shape}")
            print(f" check test shape : {test_df.shape}")
            
            self.datasets['application_train'] = train_df
            self.datasets['application_test'] = test_df
            
            return train_df, test_df
            
        except FileNotFoundError as e:
            print(f"print FileNotFoundError: {e}")
            print("data/raw/application_train.csv or data/raw/application_test.csv not found!")
            return None, None
    
    def load_bureau_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load the bureau data
        
        Returns:
            bureau_df, bureau_balance_df: bureau data and bureau balance data
        """
        print("Loading bureau data...")
        
        try:
            bureau_df = pd.read_csv(os.path.join(self.data_path, "bureau.csv"))
            bureau_balance_df = pd.read_csv(os.path.join(self.data_path, "bureau_balance.csv"))
            
            print(f" check bureau shape : {bureau_df.shape}")
            print(f" check bureau_balance shape : {bureau_balance_df.shape}")
            
            self.datasets['bureau'] = bureau_df
            self.datasets['bureau_balance'] = bureau_balance_df
            
            return bureau_df, bureau_balance_df
            
        except FileNotFoundError as e:
            print(f"print FileNotFoundError: {e}")
            return None, None
    
    def load_previous_application_data(self) -> pd.DataFrame:
        """
        Load the previous application data
        
        Returns:
            previous_app_df: previous application data
        """
        print("Loading previous application data...")
        
        try:
            previous_app_df = pd.read_csv(os.path.join(self.data_path, "previous_application.csv"))
            
            print(f" check previous_app shape : {previous_app_df.shape}")
            
            self.datasets['previous_application'] = previous_app_df
            
            return previous_app_df
            
        except FileNotFoundError as e:
            print(f"print FileNotFoundError: {e}")
            return None
    
    def load_pos_cash_data(self) -> pd.DataFrame:
        """
        Load the POS and cash loan data
        
        Returns:
            pos_cash_df: POS and cash loan data
        """
        print("Loading POS and cash loan data...")
        
        try:
            pos_cash_df = pd.read_csv(os.path.join(self.data_path, "POS_CASH_balance.csv"))
            
            print(f" check pos_cash shape : {pos_cash_df.shape}")
            
            self.datasets['pos_cash'] = pos_cash_df
            
            return pos_cash_df
            
        except FileNotFoundError as e:
            print(f"print FileNotFoundError: {e}")
            return None
    
    def load_credit_card_data(self) -> pd.DataFrame:
        """
        Load the credit card data
        
        Returns:
            credit_card_df: credit card data
        """
        print("Loading credit card data...")
        
        try:
            credit_card_df = pd.read_csv(os.path.join(self.data_path, "credit_card_balance.csv"))
            
            print(f" check credit_card shape : {credit_card_df.shape}")
            
            self.datasets['credit_card'] = credit_card_df
            
            return credit_card_df
            
        except FileNotFoundError as e:
            print(f"print FileNotFoundError: {e}")
            return None
    
    def load_installments_data(self) -> pd.DataFrame:
        """
        Load the installments data
        
        Returns:
            installments_df: installments data
        """
        print("Loading installments data...")
        
        try:
            installments_df = pd.read_csv(os.path.join(self.data_path, "installments_payments.csv"))
            
            print(f" check installments shape : {installments_df.shape}")
            
            self.datasets['installments'] = installments_df
            
            return installments_df
            
        except FileNotFoundError as e:
            print(f"print FileNotFoundError: {e}")
            return None
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all data
        
        Returns:
            datasets: dictionary of all loaded datasets
        """
        print("Loading all data...")
        
        # Load the application data
        self.load_application_data()
        
        # Load other datasets
        self.load_bureau_data()
        self.load_previous_application_data()
        self.load_pos_cash_data()
        self.load_credit_card_data()
        self.load_installments_data()
        
        print(f"len of datasets : {len(self.datasets)} datasets loaded.")
        
        return self.datasets
    
    def get_data_info(self) -> None:
        """
        Display basic information about each dataset
        """
        if not self.datasets:
            print("No datasets loaded.")
            return
        
        print("\n=== Dataset Information ===")
        for name, df in self.datasets.items():
            if df is not None:
                print(f"\n{name}:")
                print(f"  Shape: {df.shape}")
                print(f"  Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                print(f"  Missing Values: {df.isnull().sum().sum()}")


def reduce_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Reduce memory usage of a DataFrame
    
    Args:
        df: Input DataFrame
        verbose: Whether to display detailed information

    Returns:
        Optimized DataFrame
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype

        # Skip object and string types
        if col_type == object or pd.api.types.is_string_dtype(df[col]):
            continue

        # Only process numeric types
        if pd.api.types.is_numeric_dtype(df[col]):
            try:
                c_min = df[col].min()
                c_max = df[col].max()

                # Check for NaN values
                if pd.isna(c_min) or pd.isna(c_max):
                    continue
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                elif 'float' in str(col_type):
                    # For float types, handle more conservatively
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
            except Exception as e:
                if verbose:
                    print(f"Skipping column {col}: {e}")
                continue
    
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    if verbose:
        print(f'Memory usage reduced from {start_mem:.2f} MB to {end_mem:.2f} MB '
              f'({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')

    return df