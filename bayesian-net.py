"""
    Project for Machine Learning in Finance course @ SeoulTech
    project goal: Customer churn prediction

    author :Sabina Gulcikova (xgulci00@stud.fit.vutbr.cz)

    This file contains scripts for training and evaluating
    the performance of Bayesian network on customer churn data.
"""
import numpy as np
from pgmpy.inference import VariableElimination, BeliefPropagation, Mplp, CausalInference
from pgmpy.metrics import correlation_score, log_likelihood_score, structure_score, BayesianModelProbability
from sklearn.preprocessing import label_binarize

'External packages'
import argparse
import os
import pickle

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from pgmpy.estimators import HillClimbSearch, BayesianEstimator, ParameterEstimator, BicScore, PC, K2Score, \
    MmhcEstimator, TreeSearch, ExhaustiveSearch, MaximumLikelihoodEstimator, ExpectationMaximization, IVEstimator, \
    BDeuScore, BDsScore, AICScore
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
    auc, classification_report, average_precision_score,
)
from tqdm import tqdm


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
        # approach 1 HillClimbSearch
        # k2score, bdeuscore, bdsscore, bicscore, aicscore


        # hc = HillClimbSearch(self.train_df, state_names=self.state_names, use_cache=True)
        # learned_structure = hc.estimate(scoring_method=BicScore(self.train_df))


        # learned_structure = hc.estimate(scoring_method=BDeuScore(self.train_df))
        # This one has nice pr curve (below)
        # BDsScore, K2Score, BicScore, BDeuScore


        # learned_structure = hc.estimate(scoring_method=K2Score(self.train_df))
        # approach 2 HillClimbSearch with k2
        # hc = HillClimbSearch(self.train_df, state_names=self.state_names)
        # learned_structure = hc.estimate(scoring_method=K2Score(self.train_df), tabu_length=150)

        # tree_estimator = TreeSearch(self.train_df)
        # learned_structure = tree_estimator.estimate()

        # tree_estimator = TreeSearch(self.train_df)
        # learned_structure = tree_estimator.estimate(estimator_type='tan', class_node='Churn')

        # pc_estimator = PC(self.train_df)
        # learned_structure = pc_estimator.estimate(ci_test='pearsonr', return_type='dag', show_progress=True)

        # pc_estimator = PC(self.train_df)
        # learned_structure = pc_estimator.estimate(ci_test='g_sq')

        mmhc_estimator = MmhcEstimator(self.train_df, state_names=self.state_names)
        learned_structure = mmhc_estimator.estimate(tabu_length=500, significance_level=0.02)

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
        # scoring_method = K2Score(self.train_df)
        #
        # # Initialize ExhaustiveSearch with data, scoring method, and state_names
        # es = ExhaustiveSearch(
        #     self.train_df,
        #     scoring_method=scoring_method,
        #     state_names=self.state_names,
        #     use_cache=True
        # )
        #
        # # Perform exhaustive search to find the best model
        # learned_structure = es.estimate()
        print(f'----------------------------')
        print("Structure learning completed.")
        print(f'----------------------------')

        # Create a BayesianNetwork with the learned structure
        self.best_model = BayesianNetwork(learned_structure.edges())

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
                model.fit(batch_df, estimator=MaximumLikelihoodEstimator, state_names=self.state_names)

                # model.fit(df_train, estimator=BayesianEstimator/MaximumLikelihoodEstimator)
                # infer = VariableElimination(model)

                # Update the progress bar
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

    def make_predictions(self, model):
        # Ensure that columns are in the same order as the training data
        ordered_columns = self.test_df.columns
        x_test = self.test_df[ordered_columns].copy()
        predicted_labels = []

        # Get the list of variables for which you want to compute the probability (e.g., 'Churn')

        # Initialize the inference object using Variable Elimination
        # inference = BeliefPropagation(model)
        inference = VariableElimination(model)
        # Make predictions
        probabilities = []
        #
        for index, row in x_test.iterrows():
            evidence = {variable: row[variable] for variable in model.nodes() if variable != 'Churn'}
            prediction = inference.query(variables=[self.target_variable], evidence=evidence, joint=False)
            discrete_factor = list(prediction.values())[0]
            probability_churn_1 = discrete_factor.values[1]
            probabilities.append(probability_churn_1)

            predicted_label = 1 if probability_churn_1 >= 0.5 else 0
            predicted_labels.append(predicted_label)

        # print(f'-----------')
        # print(probabilities)
        # print(f'-----------')
        # print(x_test[self.target_variable].values)
        # print(f'-----------')

        # fpr, tpr, thresholds = roc_curve(x_test[self.target_variable], probabilities)
        # roc_auc = auc(fpr, tpr)
        # print('ROC AUC:', roc_auc)
        #
        # # Visualize the ROC curve
        # plt.figure(figsize=(8, 8))
        # plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver Operating Characteristic (ROC) Curve for Bayesian net')
        # plt.legend(loc='lower right')
        # # plt.savefig('graphs/benchmark-roc.png')
        # plt.show()

        # Confusion Matrix
        conf_matrix = confusion_matrix(x_test[self.target_variable], predicted_labels)
        print('Confusion Matrix:\n', conf_matrix)

        # Accuracy
        accuracy = accuracy_score(x_test[self.target_variable], predicted_labels)
        print('Accuracy:', accuracy)
        #
        # # ROC Curve
        # fpr, tpr, thresholds = roc_curve(true_labels, predictions)
        # roc_auc = auc(fpr, tpr)
        # print('ROC AUC:', roc_auc)
        #
        # # Precision-Recall Curve
        # precision, recall, _ = precision_recall_curve(x_test[self.target_variable], probabilities)
        # average_precision = average_precision_score(x_test[self.target_variable], predicted_labels)
        # # Plot Precision-Recall Curve
        # plt.step(recall, precision, color='b', alpha=0.2, where='post')
        # plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.ylim([0.0, 1.05])
        # plt.xlim([0.0, 1.0])
        # plt.title('Precision-Recall Curve (AP = {0:.2f}), Bayesian net'.format(average_precision))
        # plt.show()

        #
        # print(f'-------------------------------------------')
        # print(f'--------Assessing model\'s quality---------')
        # print(f'-------------------------------------------')

        # # TODO move to function
        # score = correlation_score(model, self.test_df, test='chi_square', significance_level=0.01)
        # print(f"Correlation Score: {score}")
        #
        # score = log_likelihood_score(model, self.test_df)
        # print(f"Log-Likelihood Score: {score}")
        #
        # score = structure_score(model, self.test_df, scoring_method='bic')
        # print(f"Structure Score: {score}")
        #
        # bml = BayesianModelProbability(model)
        #
        # # Log probability of each data point
        # log_probabilities = bml.log_probability(self.test_df)
        # print(f"Log Probabilities: {log_probabilities}")
        #
        # # Total log probability density under the model
        # total_log_likelihood = bml.score(self.test_df)
        # print(f"Total Log-Likelihood: {total_log_likelihood}")
        #
        # print(f'-------------------------------------------')
        # print(f'-------------------------------------------')


class BayesianEvaluator:
    def __init__(self, csv_train_path: str, csv_test_path: str, model) -> None:
        self.csv_train_path = csv_train_path
        self.csv_test_path = csv_test_path
        self.train_df = pd.read_csv(self.csv_train_path)
        self.test_df = pd.read_csv(self.csv_test_path)
        self.model = model
        self.target_variable = 'Churn'

    def evaluate_predictions(self):
        inference = BeliefPropagation(self.model)
        predictions = []
        probabilities = []
        for index, row in self.test_df.iterrows():
            evidence = {variable: row[variable] for variable in model.nodes() if variable != 'Churn'}
            prediction = inference.query(variables=[self.target_variable], evidence=evidence, joint=False)
            discrete_factor = list(prediction.values())[0]
            probability_churn_1 = discrete_factor.values[1]
            # probabilities.append(probability_churn_1)

            predicted_label = 1 if probability_churn_1 >= 0.5 else 0
            probabilities.append(probability_churn_1)
            predictions.append(predicted_label)

        # Assuming your target variable is 'target_variable'
        true_labels = self.test_df['Churn'].values

        accuracy = accuracy_score(true_labels, predictions)
        print(f"Accuracy: {accuracy}")

        cm = confusion_matrix(true_labels, predictions)
        print(f"Confusion Matrix: \n{cm}")

        precision = precision_score(true_labels, predictions)
        print("Precision:", precision)

        recall = recall_score(true_labels, predictions)
        print("Recall:", recall)

        f1 = f1_score(true_labels, predictions)
        print("F1 Score:", f1)

        # fpr, tpr, _ = roc_curve(true_labels, probabilities)
        # auc_score = auc(fpr, tpr)
        #
        # plt.figure(figsize=(8, 8))
        # plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % auc_score)
        # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver Operating Characteristic (ROC) Curve')
        # plt.legend(loc="lower right")
        # plt.show()


        #
        # average_precision = average_precision_score(actual_churn, predicted_churn)
        # precision, recall, _ = precision_recall_curve(actual_churn, self.probabilities)
        #
        # plt.figure(figsize=(8, 8))
        # plt.step(recall, precision, color='b', alpha=0.2, where='post')
        # plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.title('Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
        # plt.show()


if __name__ == '__main__':
    args = parse_args()
    train = BayesianTrainer(args.csv_path_train, args.csv_path_test, args.net_structure_path,
                            args.net_structure_and_params_path, args.batch_size)

    train.structure_train()
    train.check_for_cycles()
    train.plot_structure()

    # Remove after debug
    # mod = train.get_dummy_struc()
    # train.make_predictions(mod)
    # --

    model = train.train_parameters()
    # model = train.load_parameters()
    train.print_trained_structure(model)

    # predictions, probabilities = train.make_predictions(model)
    # train.make_predictions(model)
    # train.dummyEval(model)
    # print(predictions)

    evaluator = BayesianEvaluator(args.csv_path_train, args.csv_path_test, model)
    evaluator.evaluate_predictions()
    # evaluator.visualize_evaluation()
