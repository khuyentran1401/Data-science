import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_data(file_path: str):
    return pd.read_csv(file_path)


def preprocess_data(df: pd.DataFrame):
    # Drop the customer_id column as it's not relevant for prediction
    df.drop("customer_id", axis=1, inplace=True)

    # Convert categorical variables to numerical using LabelEncoder
    cat_cols = ["gender", "marital_status", "education", "occupation"]
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df


def split_data(df: pd.DataFrame):
    # Split data into features and target
    X = df.drop("churn", axis=1)
    y = df["churn"]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame):
    model = LogisticRegression()
    model.fit(X_train, y_train)

    return model


def test_model(model: LogisticRegression, X_test: pd.DataFrame, y_test: pd.DataFrame):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Print results
    print("Accuracy:", accuracy)
    return accuracy


def save_model(model):
    joblib.dump(model, "churn_model.pkl")


def main():
    df = load_data(
        "https://gist.githubusercontent.com/khuyentran1401/146b40778a60293d15f261d06d27dc32/raw/12620a6108243731052c15b2cf024f4fa705cd6f/customer_churn_train.csv"
    )
    processed_df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(processed_df)
    model = train_model(X_train, y_train)
    save_model(model)
    return test_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
