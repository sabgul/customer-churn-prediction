"""
    Project for Machine Learning in Finance course @ SeoulTech
    project goal: Customer churn prediction

    author :Sabina Gulcikova (xgulci00@stud.fit.vutbr.cz)

    This file contains helper scripts for preprocessing of datasets
    and exploratory data analysis.
"""

'External packages'
import argparse
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from typing import Optional
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


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

    def remove_missing_values(self, dataset=None) -> Optional[pd.DataFrame]:
        if dataset is None:
            dataset = self.df

        # Check for missing values
        missing_values = dataset.isnull().sum()

        # Display columns with missing values
        print(f'Columns with missing values: {missing_values[missing_values > 0]}\n')

        # Replace missing values if detected. In case of numerical values,
        # use the mean value, in case of categorical, use the most frequent value.
        dataset = (
            dataset.apply(
                lambda x: x.fillna(x.mean()) if pd.api.types.is_numeric_dtype(x) else x.fillna(x.mode().iloc[0]))
            if any(missing_values > 0)
            else dataset
        )

        return dataset

    # TODO handle outliers
    def handle_outliers(self, outliers_cols) -> Optional[pd.DataFrame]:
        # if outliers are present, process them
        return self.df

    # Use one hot encoding to avoid imposing ordinal relationships.
    # This is necessary for latter use of Bayesian networks.
    def one_hot_encoding_categorical(self) -> Optional[pd.DataFrame]:
        # print(f'COLUMNS WHICH ARE PRESENT: {self.df.columns}\n')
        self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')
        self.df['MonthlyCharges'] = pd.to_numeric(self.df['MonthlyCharges'], errors='coerce')

        categorical_cols = self.df.select_dtypes(include=['object']).columns
        more_than_two_unique_values = [col for col in categorical_cols if len(self.df[col].unique()) > 2]

        # Apply one-hot encoding only for selected categorical columns
        self.df = pd.get_dummies(self.df, columns=more_than_two_unique_values, drop_first=True)
        self.df = self.df.replace({True: 1, False: 0, 'Yes': 1, 'No': 0, 'Male': 0, 'Female': 1})

        return self.df

    # Use label encoding to transform categorical into numeric.
    # Used for identifying importance of variables with RandomForest.
    def label_encode_categorical(self) -> Optional[pd.DataFrame]:
        data = self.df

        data['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')
        data['MonthlyCharges'] = pd.to_numeric(self.df['MonthlyCharges'], errors='coerce')

        if 'customerID' in data.columns:
            data = data.drop('customerID', axis=1)
        categorical_cols = data.select_dtypes(include=['object']).columns
        more_than_two_unique_values = [col for col in categorical_cols if len(self.df[col].unique()) > 2]

        le = LabelEncoder()
        for column in more_than_two_unique_values:
            data[column] = le.fit_transform(data[column])
        data = data.replace({True: 1, False: 0, 'Yes': 1, 'No': 0, 'Male': 0, 'Female': 1})

        return data

    def drop_attributes(self, attributes) -> Optional[pd.DataFrame]:
        self.df.drop(columns=attributes, inplace=True)
        if 'customerID' in self.df.columns:
            self.df = self.df.drop('customerID', axis=1)
        return self.df

    def split_dataset(self, test_size=0.2, random_state=None):
        x = self.df.drop('Churn', axis=1)
        y = self.df['Churn']

        # Split the dataset
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

        # Combine features and target variable for both train and test sets
        train = pd.concat([x_train, y_train], axis=1)
        test = pd.concat([x_test, y_test], axis=1)

        # commented to avoid generating new distribution
        train.to_csv('data/train_data.csv', index=False)
        test.to_csv('data/test_data.csv', index=False)

        return train, test


class FeatureAnalyzer:
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
        print(f'-----------------------------')
        print(f'Feature information:\n{self.df.info}\n')

        # Get description of the dataset to determine most frequent values,
        # number of unique values for each parameter, and identify possible missing values
        data_description = self.df.describe(include='all').T
        print(f'-----------------------------')
        print(f"Descriptive statistics of the dataset:\n{data_description}\n")

        print(f'-----------------------------')
        print(f'Columns: {self.df.columns} and their types: {self.df.columns.dtype}\n')

        # Get information about categorical and numerical attributes
        categorical_vars = []
        numerical_vars = []
        for column in self.df.columns:
            if self.df[column].dtype == 'object':
                categorical_vars.append(column)
            else:
                numerical_vars.append(column)
        print(f'-----------------------------')
        print(f'Categorical variables:\n{categorical_vars}\n')
        print(f'Numerical variables:\n{numerical_vars}\n')

        self.df['totalcharges'] = pd.to_numeric(self.df['totalcharges'], errors='coerce')
        self.df['monthlycharges'] = pd.to_numeric(self.df['monthlycharges'], errors='coerce')

        # Get number of unique values for each column
        unique_counts = self.df.nunique()
        print(f'-----------------------------')
        print(f'Number of unique values for each column:\n{unique_counts}\n')
        print(f'-----------------------------')
        print(f"Who left more according to the gender?\n{self.df.groupby('gender')['churn'].value_counts().unstack()}")
        print(f'-----------------------------')
        print(f"How many years were they with company before leaving?\n{self.df.groupby(['gender', 'churn'])['tenure'].agg(['mean'])}")
        print(f'-----------------------------')
        print(f"Min, max and mean according to the dependent variable:\n{self.df.groupby('churn').agg({'tenure': ['min', 'mean', 'max'], 'monthlycharges': ['min', 'mean', 'max'], 'totalcharges': ['min', 'mean', 'max']})}")

    def identify_outliers(self) -> [str]:
        outliers_columns = []
        # num_cols = ['monthlycharges', 'tenure', 'totalcharges', 'seniorcitizen']

        self.df['totalcharges'] = pd.to_numeric(self.df['totalcharges'], errors='coerce')
        self.df['monthlycharges'] = pd.to_numeric(self.df['monthlycharges'], errors='coerce')

        print(f'----- OUTLIER IDENTIFICATION:')
        for column in self.df.select_dtypes(exclude='object').columns:
            print(f'-----------------------------')
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

        print(f'-----------------------------')
        return outliers_columns

    def correlation_analysis(self, dataset=None) -> None:
        if dataset is None:
            dataset = self.df
        correlation_matrix = dataset.corr()

        plt.figure(figsize=(12, 10))
        sn.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, annot_kws={"size": 7},)
        plt.title("Correlation Matrix")
        plt.show()
        pass

    def visualize_data(self) -> None:
        fig, axarr = plt.subplots(2, 2, figsize=(20, 12))
        sn.boxplot(y='gender', x='churn', hue='churn', data=self.df, ax=axarr[0][0])
        sn.boxplot(y='tenure', x='churn', hue='churn', data=self.df, ax=axarr[0][1])
        sn.boxplot(y='monthlycharges', x='churn', hue='churn', data=self.df, ax=axarr[1][0])
        sn.boxplot(y='totalcharges', x='churn', hue='churn', data=self.df, ax=axarr[1][1])
        plt.show()

        fig, axarr = plt.subplots(2, 2, figsize=(20, 12))
        sn.countplot(x='contract', hue='churn', data=self.df, ax=axarr[0][0])
        sn.countplot(x='gender', hue='churn', data=self.df, ax=axarr[0][1])
        sn.countplot(x='paymentmethod', hue='churn', data=self.df, ax=axarr[1][0])
        sn.countplot(x='internetservice', hue='churn', data=self.df, ax=axarr[1][1])
        plt.show()

    def tree_feature_importance_analysis(self, dataset=None) -> [str]:
        if dataset is None:
            dataset = self.df
        x = dataset.drop('Churn', axis=1)
        y = dataset['Churn']

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        rf_model = RandomForestClassifier()
        rf_model.fit(x_train, y_train)

        feature_importances = rf_model.feature_importances_
        feature_names = x.columns

        selected_features = [feature for feature, importance in zip(feature_names, feature_importances) if
                             importance < 0.025]
        print(f'-----------------------------')
        print(f'------ FEATURE IMPORTANCE ASSESSMENT:')
        for feature, importance in zip(feature_names, feature_importances):
            print(f"Feature {feature}: Importance {importance}")

        return selected_features


if __name__ == '__main__':
    args = parse_args()

    analyzer = FeatureAnalyzer(args.csv_path)
    preparator = DatasetPreparator(args.csv_path)

    analyzer.get_dataset_characteristics()
    outliers_cols = analyzer.identify_outliers()
    analyzer.visualize_data()

    # ----- Prepare dummy dataset for attribute importance analysis
    label_encoded_dataset = preparator.label_encode_categorical()
    label_encoded_dataset = preparator.remove_missing_values(label_encoded_dataset)
    analyzer.correlation_analysis(label_encoded_dataset)
    attrs_to_prune = analyzer.tree_feature_importance_analysis(label_encoded_dataset)

    # ----- Prepare actual dataset
    preparator.drop_attributes(attrs_to_prune)
    preparator.remove_missing_values()
    preparator.one_hot_encoding_categorical()
    train_data, test_data = preparator.split_dataset()