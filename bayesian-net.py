"""
    Project for Machine Learning in Finance course @ SeoulTech
    project goal: Customer churn prediction

    author :Sabina Gulcikova (xgulci00@stud.fit.vutbr.cz)

    This file contains scripts for training and evaluating
    the performance of Bayesian network on customer churn data.
"""
import numpy as np
from pgmpy.inference import VariableElimination

'External packages'
import argparse
import os
import pickle

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from pgmpy.estimators import HillClimbSearch, BayesianEstimator, ParameterEstimator, BicScore
from pgmpy.models import BayesianNetwork, BayesianModel
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    precision_recall_curve,
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
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
    args.add_argument('--batch-size', type=int, default=1000,
                      help='Size of each batch for processing data.')

    return args.parse_args()


class BayesianTrainer:
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
        # Load the model if it exists, else set it to None
        self.best_model = self.load_model_structure()

    def load_model_structure(self):
        if os.path.exists(self.net_structure_path):
            with open(self.net_structure_path, 'rb') as file:
                return pickle.load(file)
        else:
            self.structure_train()

    def structure_train(self) -> None:
        # Perform Hill Climbing Search for Bayesian Network structure learning
        bic = BicScore(self.train_df)

        # Perform Hill Climbing Search for Bayesian Network structure learning
        hc = HillClimbSearch(self.train_df)

        # Track the BIC scores to monitor convergence
        bic_scores = []

        # Set your custom stopping criterion
        max_iterations = 10000
        epsilon = 1e-4

        for i in range(max_iterations):
            # Perform a single step of the search
            next_model = hc.estimate()

            # Get the BIC score of the new model
            bic_score = bic.score(next_model)
            bic_scores.append(bic_score)

            # Check for convergence based on the improvement in BIC score
            if i > 0 and abs(bic_scores[-1] - bic_scores[-2]) < epsilon:
                break

        # Print the learned structure
        print(next_model.edges())

        # hc = HillClimbSearch(self.train_df, max_iter=10000)
        # structure = hc.estimate()
        #
        # self.best_model = BayesianNetwork(structure.edges())
        # with open(self.net_structure_path, 'wb') as file:
        #     pickle.dump(self.best_model, file)
        #
        # # Print the edges of the learned structure
        # print(self.best_model.edges())

    def plot_structure(self) -> None:
        # Create a directed graph from the edges of the learned structure
        if self.best_model is None:
            print(f'Err: No model structure found. Please train the model structure at first.')
            return

        net = nx.DiGraph(self.best_model.edges())

        # Draw the graph
        pos = nx.spring_layout(net)
        plt.figure(figsize=(15, 10))
        nx.draw(net, pos, with_labels=True, font_weight='bold', node_color='skyblue', node_size=1500, arrowsize=12,
                font_size=7)

        plt.savefig('graphs/bayes_net.png')
        # Show the plot
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
                model.fit(batch_df, estimator=BayesianEstimator, prior_type='BDeu', equivalent_sample_size=10)

                # model.fit(df_train, estimator=MaximumLikelihoodEstimator)
                # infer = VariableElimination(model)

                # Update the progress bar
                pbar.update(1)

        with open(self.net_structure_and_params_path, 'wb') as file:
            pickle.dump(model, file)

        return model

    def make_predictions(self, model):
        print("Edges of the Bayesian model:", model.edges())

        # Print CPDs (conditional probability distributions) if available
        for cpd in model.get_cpds():
            print(cpd)

        for variable in model.nodes():
            cpd = model.get_cpds(variable)
            states = cpd.state_names[variable]
            print(f"Variable: {variable}, States: {states}")
        # test_data_for_prediction = self.test_df.copy()
        # # test_data_for_prediction.loc[:, 'Churn'] = None  # Assuming 'Churn' is the target variable
        # # test_data_for_prediction['Churn'] = np.nan
        # # test_data_for_prediction['Churn'] = np.nan  # or any placeholder value
        # # self.test_df.drop('Churn', axis=1, inplace=True)
        # # x_test = self.test_df.drop('Churn', axis=1, inplace=True)
        #
        # # self.test_df['Churn'] = np.nan
        # # Predict missing variables
        # # preds = model.predict(self.test_df)
        #
        # inference = VariableElimination(model)
        # # preds = inference.query(variables=[self.target_variable], evidence=x_test)
        # # preds = [model.predict(evidence=evidence_point) for evidence_point in x_test.to_dict(orient='records')]
        # if self.test_df is not None:
        #     if 'Churn' in self.test_df.columns:
        #         inference = VariableElimination(model)
        #
        #         # Drop 'Churn' column from x_test
        #         x_test_without_churn = self.test_df.drop('Churn', axis=1)
        #
        #         # Convert x_test_without_churn to a list of dictionaries
        #         evidence_list = x_test_without_churn.to_dict(orient='records')
        #
        #         # Make predictions for the 'Churn' variable for each row
        #         y_pred_churn_list = [inference.query(variables=['Churn'], evidence=evidence)['Churn'] for evidence in
        #                              evidence_list]
        #
        #         # Concatenate the predictions into a single DataFrame
        #         y_pred_churn_df = pd.concat(y_pred_churn_list, axis=1).transpose()
        #         return y_pred_churn_df
        #     else:
        #         print("Error: 'Churn' column not found in test dataset.")
        #         return None
        # else:
        #     print("Error: Test dataset is None.")
        #     return None


class BayesianEvaluator:
    def __init__(self, csv_train_path: str, csv_test_path: str, predictions, model) -> None:
        self.csv_train_path = csv_train_path
        self.csv_test_path = csv_test_path
        self.train_df = pd.read_csv(self.csv_train_path)
        self.test_df = pd.read_csv(self.csv_test_path)
        self.predictions = predictions
        self.model = model

    def evaluate_predictions(self):
        # 'Churn' is the target variable
        y_true = self.test_df['Churn']
        x_test = self.test_df.drop('Churn', axis=1)

        # Make predictions
        y_pred = self.model.predict(x_test)

        # Evaluate accuracy
        accuracy = accuracy_score(y_true, y_pred)
        print(f'Accuracy: {accuracy}')

        # Evaluate precision
        precision = precision_score(y_true, y_pred)
        print(f'Precision: {precision}')

        # Evaluate recall
        recall = recall_score(y_true, y_pred)
        print(f'Recall: {recall}')

        # Evaluate F1-score
        f1 = f1_score(y_true, y_pred)
        print(f'F1-score: {f1}')

        # Evaluate AUC-ROC
        roc_auc = roc_auc_score(y_true, y_pred)
        print(f'AUC-ROC: {roc_auc}')

        return y_true, y_pred, x_test

    def visualize_evaluation(self) -> None:
        y_true, y_pred, x_test = self.evaluate_predictions()

        # Confusion Matrix Visualization
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1"])
        disp.plot()
        plt.savefig('graphs/bayes_conf_matrix.png')
        plt.show()

        # Receiver Operating Characteristic (ROC) Curve
        fpr, tpr, _ = roc_curve(y_true, self.model.predict_proba(x_test)[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic Curve')
        plt.legend(loc='lower right')
        plt.savefig('graphs/bayes_roc.png')
        plt.show()

        # Precision-Recall Curve
        y_probs = model.predict_proba(x_test)[:, 1]

        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        plt.figure()
        plt.step(recall, precision, color='b', where='post')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.savefig('graphs/bayes_prec_recall_curve.png')
        plt.show()


if __name__ == '__main__':
    args = parse_args()
    train = BayesianTrainer(args.csv_path_train, args.csv_path_test, args.net_structure_path,
                            args.net_structure_and_params_path, args.batch_size)

    train.structure_train()
    train.check_for_cycles()
    train.plot_structure()

    # model = train.train_parameters()
    # model = train.load_parameters()
    # predictions = train.make_predictions(model)
    # train.make_predictions(model)

    # print(predictions)

    # evaluator = BayesianEvaluator(args.csv_path_train, args.csv_path_test, predictions, model)
    # evaluator.evaluate_predictions()
    # evaluator.visualize_evaluation()
