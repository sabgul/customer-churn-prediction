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


class DatasetPreparator:
    def __init__(self, csv_path: str) -> None:
        self.csv_path = csv_path
        self.data_pd = pd.read_csv(self.csv_path)


if __name__ == '__main__':
    args = parse_args()
    dataset = DatasetPreparator(args.csv_path)