"""Model evaluation and performance metrics"""
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report, confusion_matrix

class Evaluator:
    def __init__(self):
        self.results = {}
        
    def compute_metrics(self, y_true, y_pred, y_prob):
        return {
            'auc_roc': roc_auc_score(y_true, y_prob),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'classification_report': classification_report(y_true, y_pred, output_dict=True)
        }
    
    def compute_expected_loss(self, y_prob, loan_amounts, lgd=0.45):
        return y_prob * lgd * loan_amounts
