"""
    Project for Machine Learning in Finance course @ SeoulTech
    project goal: Customer churn prediction

    author :Sabina Gulcikova (xgulci00@stud.fit.vutbr.cz)

    This file contains scripts for benchmarking the value of
    ML model. In this case, evaluation of logistic regression is performed.
"""
'External packages'
import argparse
import pandas as pd
import numpy as np
from typing import Union, Tuple
import seaborn as sn
import matplotlib.pyplot as plt
from typing import Optional
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def parse_args() -> argparse.Namespace:
    args = argparse.ArgumentParser()
    args.add_argument('--csv-path-train', type=str, default='data/train_data.csv',
                      help='Path to csv file of train dataset.')
    args.add_argument('--csv-path-test', type=str, default='data/test_data.csv',
                      help='Path to csv file of test dataset.')

    return args.parse_args()


class BenchmarkTrainer:
    def __init__(self, csv_train_path: str, csv_test_path: str) -> None:
        self.csv_train_path = csv_train_path
        self.csv_test_path = csv_test_path
        self.train_df = pd.read_csv(self.csv_train_path)
        self.test_df = pd.read_csv(self.csv_test_path)

    def train_regression(self) -> Tuple[Union[pd.Series, np.ndarray], Union[pd.Series, np.ndarray]]:
        x_train = self.train_df.drop('Churn', axis=1)
        y_train = self.train_df['Churn']

        x_test = self.test_df.drop('Churn', axis=1)
        y_test = self.test_df['Churn']

        # Initialize the logistic regression model
        logistic_model = LogisticRegression(random_state=42, max_iter=1000)

        # Train the model
        logistic_model.fit(x_train, y_train)

        y_pred = logistic_model.predict(x_test)

        # Access coefficients and intercept
        coefficients = logistic_model.coef_
        intercept = logistic_model.intercept_

        print(f'-----------------------------')
        print(f'Coefficients:\n{coefficients}')
        print(f'-----------------------------')
        print(f'Intercept: {intercept}')
        print(f'-----------------------------')

        # ---------------------
        fpr, tpr, thresholds = roc_curve(y_test, logistic_model.predict_proba(x_test)[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.savefig('graphs/benchmark-roc.png')
        plt.show()

        return y_test, y_pred


class BenchmarkEvaluator:
    def __init__(self, y_test, y_pred) -> None:
        self.y_test = y_test
        self.y_pred = y_pred

    def evaluate_regression(self) -> None:
        # Evaluate accuracy
        accuracy = accuracy_score(self.y_test, self.y_pred)
        print(f'Accuracy: {accuracy:.2f}')
        print(f'-----------------------------')

        # Confusion matrix
        conf_matrix = confusion_matrix(self.y_test, self.y_pred)
        print(f'Confusion Matrix:\n{conf_matrix}')
        print(f'-----------------------------')

        # Classification report
        class_report = classification_report(self.y_test, self.y_pred)
        print(f'Classification Report:\n{class_report}')
        print(f'-----------------------------')

    # def visualize_results(self) -> None:
    #     fpr, tpr, thresholds = roc_curve(y_test, logistic_model.predict_proba(X_test)[:, 1])
    #     roc_auc = auc(fpr, tpr)


if __name__ == '__main__':
    args = parse_args()
    trainer = BenchmarkTrainer(args.csv_path_train, args.csv_path_test)
    y_test, y_pred = trainer.train_regression()

    evaluator = BenchmarkEvaluator(y_test, y_pred)
    evaluator.evaluate_regression()
