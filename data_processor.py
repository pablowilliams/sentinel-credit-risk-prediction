"""Data preprocessing pipeline for credit risk modelling"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


class DataProcessor:
    """Handles data loading, cleaning, and preprocessing"""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df = None
        self.encoders = {}
        self.scaler = StandardScaler()
        
    def load_data(self) -> pd.DataFrame:
        """Load and perform initial cleaning"""
        self.df = pd.read_csv(self.filepath)
        self.df = self._clean_data(self.df)
        return self.df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values and data type conversions"""
        emp_map = {'< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3,
                   '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7,
                   '8 years': 8, '9 years': 9, '10+ years': 10, 'n/a': np.nan}
        df['emp_length_num'] = df['emp_length'].map(emp_map)
        df['emp_length_num'] = df['emp_length_num'].fillna(df['emp_length_num'].median())
        df['default'] = (df['loan_status'] == 'Charged Off').astype(int)
        return df
    
    def encode_categoricals(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Encode categorical variables"""
        for col in columns:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.encoders[col].fit_transform(df[col])
            else:
                df[f'{col}_encoded'] = self.encoders[col].transform(df[col])
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """Prepare feature matrix and target vector"""
        feature_cols = ['loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti',
                       'delinq_2yrs', 'fico_range_low', 'inq_last_6mths', 'open_acc',
                       'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'mort_acc',
                       'pub_rec_bankruptcies', 'emp_length_num']
        X = df[feature_cols].values
        y = df['default'].values
        return X, y
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   test_size: float = 0.2, val_size: float = 0.1) -> tuple:
        """Split data into train, validation, and test sets"""
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y)
        val_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_adjusted, random_state=42, stratify=y_temp)
        return X_train, X_val, X_test, y_train, y_val, y_test
