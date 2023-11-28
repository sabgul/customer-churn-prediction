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
from pgmpy.estimators import HillClimbSearch, BayesianEstimator, ParameterEstimator, BicScore, PC, K2Score, \
    MmhcEstimator, TreeSearch, ExhaustiveSearch, MaximumLikelihoodEstimator, ExpectationMaximization
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
    auc, classification_report,
)
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    args = argparse.ArgumentParser()
    args.add_argument('--csv-path-train', type=str, default='data/train_data_labels.csv',
                      help='Path to csv file of train dataset.')
    args.add_argument('--csv-path-test', type=str, default='data/train_data_labels.csv',
                      help='Path to csv file of test dataset.')
    args.add_argument('--net-structure-path', type=str, default='models/net_structure.pkl',
                      help='Path to file with trained structure of Bayesian net.')
    args.add_argument('--net-structure-and-params-path', type=str, default='models/net_structure_params.pkl',
                      help='Path to file with trained structure and parameters of Bayesian net.')
    args.add_argument('--batch-size', type=int, default=10,
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

    # TODO remove
    def get_dummy_struc(self):
        with open(self.net_structure_path, 'rb') as file:
            return pickle.load(file)

    def structure_train(self) -> None:
        print(f'----------------------------')
        print("Starting structure learning...")
        print(f'----------------------------')
        #
        # state_names = {
        #     'MonthlyCharges': [0, 1, 2],
        #     'TotalCharges': [0, 1, 2],
        #     'tenure': [0, 1, 2],
        #     'Churn': [0, 1],
        #     'Contract': [0, 1, 2],
        #     'PaymentMethod': [0, 1, 2, 3]
        #     # 'OnlineSecurity_No internet service': [0, 1],
        #     # 'TechSupport_No internet service': [0, 1],
        #     # 'Contract_Month-to-month': [0, 1],
        #     # 'Contract_Two year': [0, 1],
        #     # 'PaymentMethod_Credit card (automatic)': [0, 1],
        #     # 'PaymentMethod_Mailed check': [0, 1],
        #     # 'PaymentMethod_Electronic check': [0, 1],
        #     # 'PaymentMethod_Bank transfer (automatic)': [0, 1],
        #     # 'TechSupport_No': [0, 1],
        #     # 'Contract_One year': [0, 1],
        #     # 'TechSupport_Yes': [0, 1]
        # }

        # approach 1 HillClimbSearch
        # hc = HillClimbSearch(self.train_df, state_names=self.state_names)
        # learned_structure = hc.estimate()

        # approach 2 HillClimbSearch with k2
        # hc = HillClimbSearch(self.train_df)
        # learned_structure = hc.estimate(scoring_method=K2Score(self.train_df), tabu_length=150)

        # tree_estimator = TreeSearch(self.train_df)
        # learned_structure = tree_estimator.estimate()

        # tree_estimator = TreeSearch(self.train_df)
        # learned_structure = tree_estimator.estimate(estimator_type='tan', class_node='Churn')

        # pc_estimator = PC(self.train_df)
        # learned_structure = pc_estimator.estimate(ci_test='pearsonr', return_type='dag', show_progress=True)

        # pc_estimator = PC(self.train_df)
        # learned_structure = pc_estimator.estimate(ci_test='g_sq')

        # mmhc_estimator = MmhcEstimator(self.train_df, state_names=self.state_names)
        # learned_structure = mmhc_estimator.estimate(tabu_length=500)
        # learned_structure = mmhc_estimator.estimate(significance_level=0.05)
        # learned_structure = mmhc_estimator.estimate(scoring_method=BicScore(self.train_df))
        # learned_structure = mmhc_estimator.estimate()



        # approach 4 MMHC this doesnt show any training progress
        # mmhc_estimator = MmhcEstimator(self.train_df)
        # learned_structure = mmhc_estimator.estimate()

        # approach 5: this works worse than hc
        # tree_estimator = TreeSearch(self.train_df)
        # learned_structure = tree_estimator.estimate(estimator_type='chow-liu')

        # Use K2Score as the scoring method (you can change it based on your preference)
        scoring_method = K2Score(self.train_df)
        #
        # # Initialize ExhaustiveSearch with data, scoring method, and state_names
        es = ExhaustiveSearch(
            self.train_df,
            scoring_method=scoring_method,
            state_names=self.state_names,
            use_cache=True
        )
        #
        # # Perform exhaustive search to find the best model
        learned_structure = es.estimate()
        print(f'----------------------------')
        print("Structure learning completed.")
        print(f'----------------------------')

        # Create a BayesianNetwork with the learned structure
        self.best_model = BayesianNetwork(learned_structure.edges())

        # self.best_model = BayesianNetwork(structure.edges())

        with open(self.net_structure_path, 'wb') as file:
            pickle.dump(self.best_model, file)

        # Print the edges of the learned structure
        print(self.best_model.edges())

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
        # model.fit(df_train, estimator=MaximumLikelihoodEstimator)

        # Doing exact inference using Variable Elimination
        # infer = VariableElimination(model)
        model = self.best_model

        # Estimate CPDs
        # Divide the data into batches and train the model on each batch
        num_batches = len(self.train_df) // self.batch_size
        with tqdm(total=num_batches, desc="Training Progress") as pbar:
            for batch_start in range(0, len(self.train_df), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(self.train_df))
                batch_df = self.train_df.iloc[batch_start:batch_end]

                # Estimate CPDs for the current batch
                model.fit(batch_df, estimator=MaximumLikelihoodEstimator, state_names=self.state_names)

                # model.fit(df_train, estimator=MaximumLikelihoodEstimator)
                # infer = VariableElimination(model)

                # Update the progress bar
                pbar.update(1)

        with open(self.net_structure_and_params_path, 'wb') as file:
            pickle.dump(model, file)

        print(f' Model check: {model.check_model()}')
        return model

    def make_predictions(self, model):
        print("Edges of the Bayesian model:", model.edges())

        # Print CPDs (conditional probability distributions) if available
        for cpd in model.get_cpds():
            print(cpd)

        trained_states = {}

        for node in model.nodes():
            states = model.get_cpds(node).state_names[node]
            trained_states[node] = states

        for variable in model.nodes():
            cpd = model.get_cpds(variable)
            states = cpd.state_names[variable]
            print(f"Variable: {variable}, States: {states}")

        inference = VariableElimination(model)
        predictions = []

        for index, row in self.test_df.iterrows():
            evidence = {variable: row[variable] for variable in model.nodes() if variable != 'Churn'}
            prediction = inference.map_query(variables=['Churn'], evidence=evidence)
            predictions.append(prediction['Churn'])

        true_labels = self.test_df['Churn']

        # Confusion Matrix
        conf_matrix = confusion_matrix(true_labels, predictions)
        print('Confusion Matrix:\n', conf_matrix)

        # Accuracy
        accuracy = accuracy_score(true_labels, predictions)
        print('Accuracy:', accuracy)

        # ROC Curve
        fpr, tpr, thresholds = roc_curve(true_labels, predictions)
        roc_auc = auc(fpr, tpr)
        print('ROC AUC:', roc_auc)

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(true_labels, predictions)
        pr_auc = auc(recall, precision)
        print('Precision-Recall AUC:', pr_auc)

        # ROC Curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.show()

        # Precision-Recall Curve
        plt.figure()
        plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve (area = {:.2f})'.format(pr_auc))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.show()

        # filtered_test_df = self.test_df.copy()
        # # Map states in the test data to the closest matching states in the trained structure
        # for node, states in trained_states.items():
        #     filtered_test_df[node] = filtered_test_df[node].apply(
        #         lambda x: min(states, key=lambda state: abs(state - x)))

        # inference = VariableElimination(model)
        # #
        # predictions = []
        # for index, row in filtered_test_df.iterrows():
        #     # Specify evidence based on the values in the current row
        #     evidence = {'tenure': row['tenure'], 'Contract': row['Contract'],
        #                 'PaymentMethod': row['PaymentMethod'], 'MonthlyCharges': row['MonthlyCharges'],
        #                 'TotalCharges': row['TotalCharges']}
        #
        #     # Use the model for prediction
        #     prediction = inference.map_query(variables=[self.target_variable], evidence=evidence)
        #     predictions.append(prediction[self.target_variable])
        #
        # # Evaluate performance
        # true_labels = self.test_df[self.target_variable]
        # accuracy = accuracy_score(true_labels, predictions)
        # confusion_mat = confusion_matrix(true_labels, predictions)
        # classification_rep = classification_report(true_labels, predictions)
        #
        # print(f'Accuracy: {accuracy}')
        # print('Confusion Matrix:\n', confusion_mat)
        # print('Classification Report:\n', classification_rep)
        #
        # # ----------------------
        # predictions_proba = []
        # # Calculate predicted probabilities for positive class
        # for index, row in filtered_test_df.iterrows():
        #     evidence = {'tenure': row['tenure'], 'Contract': row['Contract'],
        #                 'PaymentMethod': row['PaymentMethod'], 'MonthlyCharges': row['MonthlyCharges'],
        #                 'TotalCharges': row['TotalCharges']}
        #     query_result = inference.map_query(variables=[self.target_variable], evidence=evidence, show_progress=False)
        #     prediction_proba = query_result.values[1]
        #     predictions_proba.append(prediction_proba)
        #
        # # Calculate ROC curve
        # fpr, tpr, _ = roc_curve(true_labels, predictions_proba)
        # roc_auc = auc(fpr, tpr)
        #
        # # Calculate precision-recall curve
        # precision, recall, _ = precision_recall_curve(true_labels, predictions_proba)
        #
        # # Plot ROC curve
        # plt.figure(figsize=(8, 8))
        # plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver Operating Characteristic (ROC) Curve')
        # plt.legend(loc="lower right")
        # plt.show()
        #
        # # Plot precision-recall curve
        # plt.figure(figsize=(8, 8))
        # plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.title('Precision-Recall Curve')
        # plt.legend(loc="lower left")
        # plt.show()


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

    # train.structure_train()
    train.check_for_cycles()
    train.plot_structure()

    # Remove after debug
    # mod = train.get_dummy_struc()
    # train.make_predictions(mod)
    # --

    # model = train.train_parameters()

    model = train.load_parameters()
    # predictions = train.make_predictions(model)
    train.make_predictions(model)

    # print(predictions)

    # evaluator = BayesianEvaluator(args.csv_path_train, args.csv_path_test, predictions, model)
    # evaluator.evaluate_predictions()
    # evaluator.visualize_evaluation()
