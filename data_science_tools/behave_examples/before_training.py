import pandas as pd

import pandas as pd
import pandera as pa


def validate_model_input(df):
    schema = pa.DataFrameSchema(
        {
            "customer_id": pa.Column(pa.Int),
            "gender": pa.Column(pa.String, checks=pa.Check.isin(["Male", "Female"])),
            "age": pa.Column(pa.Int),
            "marital_status": pa.Column(
                pa.String, checks=pa.Check.isin(["Married", "Single"])
            ),
            "education": pa.Column(
                pa.String, checks=pa.Check.isin(["High School", "Graduate"])
            ),
            "occupation": pa.Column(
                pa.String,
                checks=pa.Check.isin(
                    ["Blue Collar", "White Collar", "Student", "Professional"]
                ),
            ),
            "income": pa.Column(pa.Int),
            "account_balance": pa.Column(pa.Int),
            "credit_score": pa.Column(pa.Int),
        }
    )
    return schema.validate(df)


data = {
    "customer_id": [1, 2, 3, 4, 5, 6, 7],
    "gender": ["Male", "Female", "Male", "Female", "Male", "Female", "Male"],
    "age": [45, 32, 22, 58, 39, 28, 48],
    "marital_status": ["Married", "Divorced", "Single", "Married", "Married", "Single", "Married"],
    "education": ["High School", "Graduate", "High School", "Graduate", "High School", "High School", "Graduate"],
    "occupation": ["Blue Collar", "White Collar", "Student", "White Collar", "Blue Collar", "Student", "White Collar"],
    "income": [50000, 70000, 20000, 90000, 60000, 25000, 80000],
    "account_balance": [1000, 4000, 300, 12000, 1500, 200, 8000],
    "credit_score": [600, 700, 500, 800, 550, 400, 750],
    "churn": ["No", "No", "Yes", "No", "Yes", "Yes", "No"],
}
validate_model_input(pd.DataFrame(data))
