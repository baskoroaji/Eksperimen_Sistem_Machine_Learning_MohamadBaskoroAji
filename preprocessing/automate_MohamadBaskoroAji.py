import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import argparse


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def create_preprocessor(df: pd.DataFrame):
    cat_features = ["Product Name", "Category", "Location", "Platform"]
    num_features = ["Units Sold", "Price", "Discount", "Units Returned"]
    
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, num_features),
        ('cat', categorical_pipeline, cat_features)
    ])
    return preprocessor, num_features, cat_features

def run_preprocessor(input: str, output: str, test_size: float=0.2, random_state: int=42):
    os.makedirs(output, exist_ok=True)
    
    df = load_dataset(input)
    y = df["Revenue"]
    X = df.drop(columns=['Revenue', 'Date'])
    
    preprocessor, num_features, cat_features = create_preprocessor(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    features_name = preprocessor.get_feature_names_out()
    train_df = pd.DataFrame(X_train_proc, columns=features_name)
    train_df["Revenue"] = y_train.values
    train_path = os.path.join(output, "sales_train_preprocessed.csv")
    train_df.to_csv(train_path, index=False)

    test_df = pd.DataFrame(X_test_proc, columns=features_name)
    test_df["Revenue"] = y_test.values
    test_path = os.path.join(output, "sales_test_preprocessed.csv")
    test_df.to_csv(test_path, index=False)
    
    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description='Automate preprocessing')
        parser.add_argument('--input', type=str, required=True, help='Path to input CSV')
        parser.add_argument('--output', type=str, required=True, help='Directory to save preprocessed files')
        parser.add_argument('--test_size', type=float, default=0.2, help='Test set proportion')
        parser.add_argument('--random_state', type=int, default=42, help='Random state for split')
        args = parser.parse_args()

        run_preprocessor(
            input_path=args.input,
            output_dir=args.output,
            test_size=args.test_size,
            random_state=args.random_state
    )
