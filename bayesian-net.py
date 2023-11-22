"""
    Project for Machine Learning in Finance course @ SeoulTech
    project goal: Customer churn prediction

    author :Sabina Gulcikova (xgulci00@stud.fit.vutbr.cz)

    This file contains scripts for training and evaluating
    the performance of Bayesian network on customer churn data.
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


def parse_args() -> argparse.Namespace:
    args = argparse.ArgumentParser()
    args.add_argument('--csv-path-train', type=str, default='data/train_data.csv',
                      help='Path to csv file of train dataset.')
    args.add_argument('--csv-path-test', type=str, default='data/test_data.csv',
                      help='Path to csv file of test dataset.')

    return args.parse_args()


class BayesianTrainer:
    def __init__(self, csv_train_path: str, csv_test_path: str) -> None:
        self.csv_train_path = csv_train_path
        self.csv_test_path = csv_test_path
        self.train_df = pd.read_csv(self.csv_train_path)
        self.test_df = pd.read_csv(self.csv_test_path)


class BayesianEvaluator:
    def __init__(self, csv_train_path: str, csv_test_path: str) -> None:
        self.csv_train_path = csv_train_path
        self.csv_test_path = csv_test_path
        self.train_df = pd.read_csv(self.csv_train_path)
        self.test_df = pd.read_csv(self.csv_test_path)


if __name__ == '__main__':
    args = parse_args()
