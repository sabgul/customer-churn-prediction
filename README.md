# Customer Churn Prediction

Predicts customer churn on the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) using three classifiers: Logistic Regression (baseline), Support Vector Machine, and a Bayesian Network. The Bayesian Network is the focus — it learns a probabilistic graphical model structure via the MMHC algorithm and performs inference using Belief Propagation.

## Structure

```
.
├── preprocessing.py       # EDA, feature engineering, train/test split generation
├── benchmark.py           # Logistic Regression baseline
├── svm.py                 # Support Vector Machine
├── bayesian-net.py        # Bayesian Network (structure learning + CPD estimation)
├── data/
│   ├── telco_customer_churn.csv     # Raw dataset
│   ├── train_data.csv / test_data.csv
│   └── train_data_last_less.csv / test_data_last_less.csv  # Feature-reduced splits
├── models/
│   ├── net_structure.pkl            # Learned Bayesian Network DAG
│   └── net_structure_params.pkl     # DAG + trained CPDs
└── graphs/                          # Saved plots (ROC curves, correlation matrix, etc.)
```

## Setup

```bash
pip install -r requirements.txt
```

## Running

Run in order:

**1. Preprocessing & EDA**
```bash
python preprocessing.py --csv-path data/telco_customer_churn.csv
```
Cleans data, performs feature importance analysis, discretizes continuous variables, and saves train/test splits.

**2. Baseline**
```bash
python benchmark.py --csv-path-train data/train_data_last_less.csv --csv-path-test data/test_data_last_less.csv
```

**3. SVM**
```bash
python svm.py --csv-path-train data/train_data_last_less.csv --csv-path-test data/test_data_last_less.csv
```

**4. Bayesian Network**
```bash
python bayesian-net.py \
  --csv-path-train data/train_data_last_less.csv \
  --csv-path-test data/test_data_last_less.csv \
  --net-structure-path models/net_structure.pkl \
  --net-structure-and-params-path models/net_structure_params.pkl
```
Loads the pre-trained structure and parameters from `models/`. To re-learn the structure from scratch, enable `structure_train()` in the script (slow — MMHC on ~5,600 samples).
