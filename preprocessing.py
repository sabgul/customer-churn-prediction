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
from typing import Optional


def parse_args() -> argparse.Namespace:
    args = argparse.ArgumentParser()
    args.add_argument('--csv-path', type=str, default='data/telco_customer_churn.csv',
                      help='Path to csv file of input dataset.')

    return args.parse_args()


class DatasetPreparator:
    """
    Class containing helper functions for obtaining basic information about the
    dataset. This information is further used for cleaning the data,
    replacement of missing values, encoding of categorical data into quantitative,
    and other preparatory transformations.
    """
    def __init__(self, csv_path: str) -> None:
        self.csv_path = csv_path
        self.df = pd.read_csv(self.csv_path)

    def remove_missing_values(self) -> Optional[pd.DataFrame]:
        # Check for missing values
        missing_values = self.df.isnull().sum()

        # Display columns with missing values
        print(f'Columns with missing values:{missing_values[missing_values > 0]}\n')

        # Replace missing values if detected. In case of numerical values,
        # use the mean value, in case of categorical, use the most frequent value.
        self.df = (
            self.df.apply(
                lambda x: x.fillna(x.mean()) if pd.api.types.is_numeric_dtype(x) else x.fillna(x.mode().iloc[0]))
            if any(missing_values > 0)
            else self.df
        )

        return self.df

    # TODO handle outliers
    def handle_outliers(self, outliers_cols) -> Optional[pd.DataFrame]:
        # if outliers are present, process them
        return self.df

    # Use one hot encoding to avoid imposing ordinal relationships.
    # This is necessary for latter use of Bayesian networks.
    # TODO transform yes and no into 0 and 1
    def transform_categorical(self) -> Optional[pd.DataFrame]:
        self.df = pd.get_dummies(self.df, drop_first=True)
        print(f'Encoded dataset:\n{self.df}\n')
        return self.df

    def drop_attributes(self, attributes) -> Optional[pd.DataFrame]:
        # TODO drop unnecessary values
        # drop_values(list[]) -> identified by exploratory data analysis

        return self.df

    def split_dataset(self, processed_dataset) -> Optional[pd.DataFrame]:
        # TODO save datasets into data/train and data/test
        pass


class ExploratoryAnalyzer:
    """
    Class containing helper functions for performing exploratory data analysis.
    This analysis identifies which variables are important in the dataset, and which
    attributes have no added value.
    """
    def __init__(self, csv_path: str) -> None:
        self.csv_path = csv_path
        self.df = pd.read_csv(self.csv_path)

    def get_dataset_characteristics(self) -> None:
        self.df.columns = map(str.lower, self.df.columns)
        print(f'Dimensions of the dataset: {self.df.shape}\n')
        print(f'Feature information:\n{self.df.info}\n')

        # Get description of the dataset to determine most frequent values,
        # number of unique values for each parameter, and identify possible missing values
        data_description = self.df.describe(include='all').T
        print(f"Descriptive statistics of the dataset:\n{data_description}\n")

        # Get information about categorical and numerical attributes
        categorical_vars = []
        numerical_vars = []
        for column in self.df.columns:
            if self.df[column].dtype == 'object':
                categorical_vars.append(column)
            else:
                numerical_vars.append(column)
        print(f'Categorical variables:{categorical_vars}\n')
        print(f'Numerical variables:{numerical_vars}\n')

        # Get number of unique values for each column
        unique_counts = self.df.nunique()
        print(f'Number of unique values for each column:\n{unique_counts}\n')

        # Drop customer id attribute
        self.df = self.df.drop('customerid', axis=1)

    def identify_outliers(self) -> [str]:
        outliers_columns = []

        # Iterate through numerical columns
        # print(f"Unique values in monthly: {self.df['monthlycharges'].unique()}")

        print(f'Columns: {self.df.columns} and their types: {self.df.columns.dtype}\n')
        # print(f"monthlycharges dtype: {self.df['monthlycharges'].dtype}\n")
        # print(f"tenure dtype: {self.df['tenure'].dtype}\n")
        # print(f"senior dtype: {self.df['seniorcitizen'].dtype}\n")

        # self.df['monthlycharges'] = pd.to_numeric(self.df['monthlycharges'], errors='coerce')

        # TODO figure out why monthlycharges are skipped
        for column in self.df.select_dtypes(exclude='object').columns:
            print(f"Processing numeric column: {column}")
            # Skip columns with only values [0, 1]
            unique_values = self.df[column].unique()

            print(f"Unique values in {column}: {unique_values}")
            if len(unique_values) == 2 and 0 in unique_values and 1 in unique_values:
                print(f"Skipping outlier detection for {column} as it contains only values [0, 1]\n")
                continue

            sn.boxplot(x=self.df[column])
            plt.show()

            sn.scatterplot(x=self.df['tenure'], y=self.df['monthlycharges'])
            plt.show()

            # Calculate quartiles and IQR
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1

            # Identify outliers using IQR
            outliers = (self.df[column] < Q1 - 1.5 * IQR) | (self.df[column] > Q3 + 1.5 * IQR)

            # Print outliers
            if outliers.any():
                print(f"Outliers in {column}:\n{self.df[outliers]}\n")
                outliers_columns.append(column)
            else:
                print(f"No outliers detected in column {column}.\n")

            return outliers_columns

    def correlation_analysis(self) -> None:
        pass

if __name__ == '__main__':
    args = parse_args()
    analyzer = ExploratoryAnalyzer(args.csv_path)

    analyzer.get_dataset_characteristics()
    outliers_cols = analyzer.identify_outliers()
    # identify_unnecessary_attrs

    preparator = DatasetPreparator(args.csv_path)
    preparator.remove_missing_values()
    preparator.handle_outliers(outliers_cols)
    # preparator.drop_attributes()
    preparator.transform_categorical()
    # train_dataset, test_dataset = preparator.split_dataset()


# 1 exploratory analysis (Understand the distribution of variables.
# Identify outliers, missing values, and patterns in the data.
# Explore relationships between variables.
# Decide which features are relevant and whether any features need transformation.)

# 2 Handle missing values and address outliers

# 3 Feature engineering (create new features if needed, transform existing)

# 4 Drop Unnecessary Features

# 5 One hot encoding (make sure the resulting values are purely numerical)

# 6 Dataset splitting
