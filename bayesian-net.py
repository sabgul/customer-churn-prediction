"""
    Project for Machine Learning in Finance course @ SeoulTech
    project goal: Customer churn prediction

    author :Sabina Gulcikova (xgulci00@stud.fit.vutbr.cz)

    This file contains scripts for training and evaluating
    the performance of Bayesian network on customer churn data.
"""
'External packages'
import argparse
import os
import pickle

import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from tqdm import tqdm
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator, MmhcEstimator
from sklearn.metrics import (
    precision_recall_curve,
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc, classification_report, average_precision_score, matthews_corrcoef,
)
from pgmpy.inference import BeliefPropagation
from pgmpy.metrics import correlation_score, log_likelihood_score, structure_score, BayesianModelProbability


def parse_args() -> argparse.Namespace:
    args = argparse.ArgumentParser()
    args.add_argument('--csv-path-train', type=str, default='data/train_data_last_less.csv',
                      help='Path to csv file of train dataset.')
    args.add_argument('--csv-path-test', type=str, default='data/train_data_last_less.csv',
                      help='Path to csv file of test dataset.')
    args.add_argument('--net-structure-path', type=str, default='models/net_structure.pkl',
                      help='Path to file with trained structure of Bayesian net.')
    args.add_argument('--net-structure-and-params-path', type=str, default='models/net_structure_params.pkl',
                      help='Path to file with trained structure and parameters of Bayesian net.')
    args.add_argument('--batch-size', type=int, default=10,
                      help='Size of each batch for processing data.')

    return args.parse_args()


class BayesianTrainer:
    """
       Class containing helper functions for training the structure
       and parameters of Bayesian network on given dataset, verifying
       its correct structure, and for loading already existing structure.
    """
    def __init__(self, csv_train_path: str, csv_test_path: str, net_structure_path: str,
                 net_structure_and_params_path: str, batch_size: int) -> None:
        self.csv_train_path = csv_train_path
        self.csv_test_path = csv_test_path
        self.train_df = pd.read_csv(self.csv_train_path)
        self.test_df = pd.read_csv(self.csv_test_path)
        self.target_variable = 'Churn'
        self.net_structure_path = net_structure_path
        self.net_structure_and_params_path = net_structure_and_params_path
        self.batch_size = batch_size
        self.best_model = self.load_model_structure()
        self.state_names = {
            'tenure': [0, 1, 2, 3, 4],
            'TotalCharges': [0, 1, 2, 3, 4],
            'PaymentMethod': [0, 1, 2, 3],
            'MonthlyCharges': [0, 1, 2, 3, 4],
            'Contract': [0, 1, 2],
            'Churn': [0, 1]
        }

    def load_model_structure(self):
        if os.path.exists(self.net_structure_path):
            with open(self.net_structure_path, 'rb') as file:
                return pickle.load(file)
        else:
            self.structure_train()

    def structure_train(self) -> None:
        print(f'----------------------------')
        print("Starting structure learning...")
        print(f'----------------------------')

        mmhc_estimator = MmhcEstimator(self.train_df, state_names=self.state_names)
        learned_structure = mmhc_estimator.estimate(tabu_length=500, significance_level=0.02)

        print(f'----------------------------')
        print("Structure learning completed.")
        print(f'----------------------------')

        # Create a BayesianNetwork with the learned structure
        self.best_model = BayesianNetwork(learned_structure.edges())
        for variable in self.state_names:
            if variable != 'Churn':
                self.best_model.add_edge(variable, 'Churn')
                print(f"Edge {variable} -> Churn added.")

        for variable in self.state_names:
            if self.best_model.has_edge('Churn', variable):
                self.best_model.remove_edge('Churn', variable)
                print(f"Edge Churn-{variable} removed.")
            else:
                print(f"No edge Churn-{variable} found in the graph.")

        with open(self.net_structure_path, 'wb') as file:
            pickle.dump(self.best_model, file)

        print(self.best_model.edges())

    def plot_structure(self) -> None:
        if self.best_model is None:
            print(f'Err: No model structure found. Please train the model structure at first.')
            return

        net = nx.DiGraph(self.best_model.edges())

        pos = nx.spring_layout(net)
        plt.figure(figsize=(15, 10))
        nx.draw(net, pos, with_labels=True, font_weight='bold', node_color='skyblue', node_size=1500, arrowsize=12,
                font_size=7)

        plt.savefig('graphs/bayes_net.png')
        plt.show()

    def check_for_cycles(self) -> None:
        if self.best_model is None:
            print(f'Err: No model structure found. Please train the model structure at first.')
            return

        is_dag = nx.is_directed_acyclic_graph(self.best_model.to_directed())
        if is_dag:
            print("The learned structure is a Directed Acyclic Graph (DAG).")
        else:
            print("The learned structure contains cycles.")

    def load_parameters(self):
        with open(self.net_structure_and_params_path, 'rb') as file:
            return pickle.load(file)

    def train_parameters(self):
        model = self.best_model

        # Estimate CPDs
        # Divide the data into batches and train the model on each batch
        num_batches = len(self.train_df) // self.batch_size
        with tqdm(total=num_batches, desc="Training Progress") as pbar:
            for batch_start in range(0, len(self.train_df), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(self.train_df))
                batch_df = self.train_df.iloc[batch_start:batch_end]

                # Estimate CPDs for the current batch
                # other estimator=MaximumLikelihoodEstimator
                model.fit(batch_df, estimator=BayesianEstimator, state_names=self.state_names)

                pbar.update(1)

        with open(self.net_structure_and_params_path, 'wb') as file:
            pickle.dump(model, file)

        print(f' Model check: {model.check_model()}')
        return model

    def print_trained_structure(self, model):
        for cpd in model.get_cpds():
            print(cpd)

        for variable in model.nodes():
            cpd = model.get_cpds(variable)
            states = cpd.state_names[variable]
            print(f"Variable: {variable}, States: {states}")


class BayesianEvaluator:
    """
          Class containing helper functions for evaluating trained structure
          and performing inference on specified test data.
    """
    def __init__(self, csv_train_path: str, csv_test_path: str, model) -> None:
        self.csv_train_path = csv_train_path
        self.csv_test_path = csv_test_path
        self.train_df = pd.read_csv(self.csv_train_path)
        self.test_df = pd.read_csv(self.csv_test_path)
        self.model = model
        self.target_variable = 'Churn'

    def check_for_nan(self, list_to_check, name):
        nan_indices_true_labels = np.where(np.isnan(list_to_check))[0]
        if nan_indices_true_labels.size > 0:
            print(f"Warning: list {name} contain NaN values at indices: {nan_indices_true_labels}")

    def evaluate_predictions(self):
        inference = BeliefPropagation(self.model)
        predictions = []
        probabilities = []
        true_labels = self.test_df['Churn'].values
        for index, row in self.test_df.iterrows():
            evidence = {variable: row[variable] for variable in model.nodes() if variable != 'Churn'}
            prediction = inference.query(variables=[self.target_variable], evidence=evidence, joint=False)
            discrete_factor = list(prediction.values())[0]

            for i in prediction.values():
                print(f'---------------------')
                print(f'Prediction values:\n{i} \n for evidence: {evidence}')
                print(f'---------------------')

            probability_churn_1 = discrete_factor.values[1]
            predicted_label = 1 if probability_churn_1 > 0.5 else 0
            probabilities.append(probability_churn_1)
            predictions.append(predicted_label)
        self.check_for_nan(true_labels, 'true_labels')
        self.check_for_nan(probabilities, 'probabilities')

        # Assuming your target variable is 'target_variable'
        class_report = classification_report(true_labels, predictions)
        print(f'Classification Report:\n{class_report}')
        print(f'-----------------------------')

        cm = confusion_matrix(true_labels, predictions)
        print(f"Confusion Matrix: \n{cm}")
        print(f'-----------------------------')

        accuracy = accuracy_score(true_labels, predictions)
        print(f"Accuracy: {accuracy}")
        print(f'-----------------------------')

        precision = precision_score(true_labels, predictions)
        print("Precision:", precision)
        print(f'-----------------------------')

        recall = recall_score(true_labels, predictions)
        print("Recall:", recall)
        print(f'-----------------------------')

        f1 = f1_score(true_labels, predictions)
        print("F1 Score:", f1)
        print(f'-----------------------------')

        # Calculate Specificity
        specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
        print(f"Specificity: {specificity}")
        print(f'-----------------------------')

        # Calculate Matthews Correlation Coefficient (MCC)
        mcc = matthews_corrcoef(true_labels, predictions)
        print(f"MCC: {mcc}")
        print(f'-----------------------------')

        plt.figure(figsize=(6, 4))
        sn.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix for Bayesian net')
        plt.show()

        # Calculate AUC-PR
        fpr, tpr, _ = roc_curve(true_labels, probabilities)
        roc_auc = auc(fpr, tpr)
        print(f"ROC-AUC: {roc_auc}")
        print(f'-----------------------------')

        precision, recall, _ = precision_recall_curve(true_labels, probabilities)
        auc_pr = auc(recall, precision)
        print(f"AUC-PR: {auc_pr}")
        print(f'-----------------------------')

    def evaluate_model_quality(self):
        print(f'-------------------------------------------')
        print(f'--------Assessing model\'s quality----------')
        print(f'-------------------------------------------')

        score = correlation_score(model, self.test_df, test='chi_square', significance_level=0.01)
        print(f"Correlation Score: {score}")

        score = log_likelihood_score(model, self.test_df)
        print(f"Log-Likelihood Score: {score}")

        score = structure_score(model, self.test_df, scoring_method='bic')
        print(f"Structure Score: {score}")

        bml = BayesianModelProbability(model)

        # Log probability of each data point
        log_probabilities = bml.log_probability(self.test_df)
        print(f"Log Probabilities: {log_probabilities}")

        # Total log probability density under the model
        total_log_likelihood = bml.score(self.test_df)
        print(f"Total Log-Likelihood: {total_log_likelihood}")
        print(f'-------------------------------------------')

    # Function for determining what combination of states leads to
    # biggest probability of customer attrition
    def determine_highest_probability(self):
        inference = BeliefPropagation(model)

        # Create an empty dictionary to store the combination of states for each variable
        most_probable_states = {}

        # Iterate over each row in the training data
        for index, row in self.train_df.iterrows():
            # Convert the row to evidence dictionary
            evidence = {variable: row[variable] for variable in model.nodes() if variable != 'Churn'}

            # Perform inference to find the most probable state for 'Churn'
            prediction = inference.query(variables=['Churn'], evidence=evidence, joint=False)

            # Extract the most probable state for 'Churn'
            most_probable_state = list(prediction.values())[0].argmax()

            # Update the dictionary with the most probable state for 'Churn'
            most_probable_states['Churn'] = most_probable_state

        # Combine the results to form the final message
        result_message = ", ".join(f"{variable}={state}" for variable, state in most_probable_states.items())
        print(f"Most probable combination of states for churn: {result_message}")


if __name__ == '__main__':
    args = parse_args()
    train = BayesianTrainer(args.csv_path_train, args.csv_path_test, args.net_structure_path,
                            args.net_structure_and_params_path, args.batch_size)

    # train.structure_train()
    train.check_for_cycles()
    train.plot_structure()

    # model = train.train_parameters()
    model = train.load_parameters()
    train.print_trained_structure(model)

    evaluator = BayesianEvaluator(args.csv_path_train, args.csv_path_test, model)
    evaluator.evaluate_predictions()
    evaluator.evaluate_model_quality()
