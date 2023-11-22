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
import networkx as nx
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
from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.models import BayesianModel
from sklearn.model_selection import train_test_split


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
        self.target_variable = 'Churn'
        self.best_model = None

    def structure_train(self) -> None:
        # Perform Hill Climbing Search for Bayesian Network structure learning
        hc = HillClimbSearch(self.train_df)
        self.best_model = hc.estimate()

        # Print the edges of the learned structure
        print(self.best_model.edges())

    def plot_structure(self) -> None:
        # Create a directed graph from the edges of the learned structure
        G = nx.DiGraph(self.best_model.edges())

        # Draw the graph
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, font_weight='bold', node_color='skyblue', node_size=1200, arrowsize=20)

        # Show the plot
        plt.show()

    def check_for_cycles(self) -> None:
        if isinstance(self.best_model, BayesianModel):
            is_dag = nx.is_directed_acyclic_graph(self.best_model.to_directed())
            if is_dag:
                print("The learned structure is a Directed Acyclic Graph (DAG).")
            else:
                print("The learned structure contains cycles.")
        else:
            print("The learned structure is not a BayesianModel.")


class BayesianEvaluator:
    def __init__(self, csv_train_path: str, csv_test_path: str) -> None:
        self.csv_train_path = csv_train_path
        self.csv_test_path = csv_test_path
        self.train_df = pd.read_csv(self.csv_train_path)
        self.test_df = pd.read_csv(self.csv_test_path)


if __name__ == '__main__':
    args = parse_args()
    train = BayesianTrainer(args.csv_path_train, args.csv_path_test)

    train.structure_train()
    train.check_for_cycles()
    train.plot_structure()
