"""
    Project for Machine Learning in Finance course @ SeoulTech
    project goal: Customer churn prediction

    author :Sabina Gulcikova (xgulci00@stud.fit.vutbr.cz)

    This file contains helper scripts for preprocessing of datasets
    and exploratory data analysis.
"""

'External packages'
import argparse
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from scipy import stats


def parse_args() -> argparse.Namespace:
    args = argparse.ArgumentParser()
    args.add_argument('--csv-path', type=str, default='data/telco_customer_churn.csv',
                      help='Path to csv file of input dataset.')

    return args.parse_args()


'''
Class containing helper functions for obtaining basic information about the 
dataset. This information is further used for cleaning the data, 
replacement of missing values, encoding of categorical data into quantitative,
and other preparatory transformations.
'''
class DatasetPreparator:
    def __init__(self, csv_path: str) -> None:
        self.csv_path = csv_path
        self.df = pd.read_csv(self.csv_path)

    def get_dataset_characteristics(self) -> None:
        self.df.columns = map(str.lower, self.df.columns)
        print(f'Dimensions of the dataset: {self.df.shape}\n')
        print(f'Feature information:\n{self.df.info}\n')
        print(f'Descriptive statistics of the dataset:\n{self.df.describe().T}\n')
        print(f'Number of churned and nonchurned:\n{self.df["churn"].value_counts()}\n')

        # TODO iterate through the variables and get the information about the
        #     range and number of each values in each variable
        # TODO get unique values for each variable
        # TODO get domain of each variable

    def clean_dataset(self) -> None:
        # TODO remove missing values
        # TODO drop unnecessary values (id)
        # TODO rename values yes->1, no->0
        # TODO encode categorical to numeral/quantitative

        pass

# TODO exploratory data analysis class/separate file

if __name__ == '__main__':
    args = parse_args()
    dataset = DatasetPreparator(args.csv_path)
    dataset.get_dataset_characteristics()
