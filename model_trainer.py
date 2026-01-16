"""Model training pipeline with ensemble methods"""

import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV


class ModelTrainer:
    """Trains and calibrates credit risk models"""
    
    def __init__(self):
        self.models = {}
        self.calibrated_models = {}
        
    def get_xgboost_model(self) -> XGBClassifier:
        """Configure XGBoost classifier"""
        return XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
            reg_lambda=1.0, scale_pos_weight=3.5, random_state=42,
            eval_metric='auc')
    
    def get_gradient_boosting_model(self) -> GradientBoostingClassifier:
        """Configure Gradient Boosting classifier"""
        return GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42)
    
    def get_random_forest_model(self) -> RandomForestClassifier:
        """Configure Random Forest classifier"""
        return RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=20,
            min_samples_leaf=10, class_weight='balanced',
            random_state=42, n_jobs=-1)
    
    def train_ensemble(self, X_train: np.ndarray, y_train: np.ndarray) -> VotingClassifier:
        """Train soft voting ensemble"""
        self.models['xgboost'] = self.get_xgboost_model()
        self.models['gradient_boosting'] = self.get_gradient_boosting_model()
        self.models['random_forest'] = self.get_random_forest_model()
        
        ensemble = VotingClassifier(
            estimators=[
                ('xgb', self.models['xgboost']),
                ('gb', self.models['gradient_boosting']),
                ('rf', self.models['random_forest'])
            ],
            voting='soft', weights=[0.4, 0.35, 0.25])
        ensemble.fit(X_train, y_train)
        return ensemble
    
    def calibrate_model(self, model, X_val: np.ndarray, y_val: np.ndarray):
        """Apply probability calibration"""
        calibrated = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
        calibrated.fit(X_val, y_val)
        return calibrated
