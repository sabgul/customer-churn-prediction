"""
    Project for Machine Learning in Finance course @ SeoulTech
    project goal: Customer churn prediction

    author :Sabina Gulcikova (xgulci00@stud.fit.vutbr.cz)

    This file contains scripts for training and evaluating
    the performance of SVM on customer churn data.
"""
import numpy as np
from pgmpy.inference import VariableElimination
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

'External packages'
import argparse
import os
import pickle

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sn
from pgmpy.estimators import HillClimbSearch, BayesianEstimator, ParameterEstimator, BicScore, PC, K2Score, \
    MmhcEstimator, TreeSearch
from pgmpy.models import BayesianNetwork, BayesianModel
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc, classification_report, precision_recall_curve, average_precision_score,
)
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    args = argparse.ArgumentParser()
    args.add_argument('--csv-path-train', type=str, default='data/train_data.csv',
                      help='Path to csv file of train dataset.')
    args.add_argument('--csv-path-test', type=str, default='data/test_data.csv',
                      help='Path to csv file of test dataset.')
    args.add_argument('--net-structure-path', type=str, default='models/net_structure.pkl',
                      help='Path to file with trained structure of Bayesian net.')
    args.add_argument('--net-structure-and-params-path', type=str, default='models/net_structure_params.pkl',
                      help='Path to file with trained structure and parameters of Bayesian net.')
    args.add_argument('--batch-size', type=int, default=10,
                      help='Size of each batch for processing data.')

    return args.parse_args()


class SVMTrainer:
    def __init__(self, csv_train_path: str, csv_test_path: str, batch_size: int) -> None:
        self.csv_train_path = csv_train_path
        self.csv_test_path = csv_test_path
        self.train_df = pd.read_csv(self.csv_train_path)
        self.test_df = pd.read_csv(self.csv_test_path)
        self.target_variable = 'Churn'
        self.batch_size = batch_size

    def train_model(self):
        # Assuming 'churn' is your target variable
        x = self.train_df.drop(self.target_variable, axis=1)  # Features
        y = self.train_df[self.target_variable]  # Target variable

        # Split the dataset into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # Initialize SVM model
        # rbf, linear, poly
        svm_model = SVC(kernel='poly', C=1.0)

        # Train the model
        svm_model.fit(x_train, y_train)

        # Make predictions
        y_pred = svm_model.predict(x_test)
        return y_pred, y_test


class SVMEvaluator:
    def __init__(self, csv_train_path: str, csv_test_path: str) -> None:
        self.csv_train_path = csv_train_path
        self.csv_test_path = csv_test_path
        self.train_df = pd.read_csv(self.csv_train_path)
        self.test_df = pd.read_csv(self.csv_test_path)
        self.target_variable = 'Churn'

    def eval_model(self, y_pred, y_test):
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        print(f'Accuracy: {accuracy}')
        print(f'Classification Report:\n{report}')

        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()

        precision, recall, _ = precision_recall_curve(y_test, y_pred)
        average_precision = average_precision_score(y_test, y_pred)

        # Plot Precision-Recall curve
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                 label='Precision-Recall curve (AP = {:.2f})'.format(average_precision))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='upper right')
        plt.show()

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Plot confusion matrix
        plt.figure(figsize=(6, 4))
        sn.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()


if __name__ == '__main__':
    args = parse_args()

    trainer = SVMTrainer(args.csv_path_train, args.csv_path_test, args.batch_size)
    predictions, y_test = trainer.train_model()

    evaluator = SVMEvaluator(args.csv_path_train, args.csv_path_test)
    evaluator.eval_model(predictions, y_test)
