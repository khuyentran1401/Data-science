import pandas as pd
from behave import given, then, when
from training import preprocess_data, split_data, test_model, train_model


@given("a original dataset without outliers exists")
def step_given_dataset(context):
    context.original_data = pd.read_csv(
        "https://gist.githubusercontent.com/khuyentran1401/146b40778a60293d15f261d06d27dc32/raw/12620a6108243731052c15b2cf024f4fa705cd6f/customer_churn_train.csv"
    )


@given("the performance when testing on the original dataset exists")
def step_given_performance_exists(context):
    data = context.original_data.copy()
    processed_data = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(processed_data)
    model = train_model(X_train, y_train)
    context.original_accuracy = test_model(model, X_test, y_test)


@given("a dataset with outliers exists")
def step_given_outliers(context):
    noisy_data = context.original_data.copy()
    noisy_data.loc[noisy_data["customer_id"] == 12, "income"] = 1_000_000_000
    noisy_data.loc[noisy_data["customer_id"] == 30, "income"] = 500_000_000
    context.noisy_data = noisy_data


@when("the model is trained on the dataset with outliers")
def step_when_trained(context):
    processed_data = preprocess_data(context.noisy_data)
    X_train, X_test, y_train, y_test = split_data(processed_data)
    model = train_model(X_train, y_train)
    context.noisy_accuracy = test_model(model, X_test, y_test)


@then("the model's performance should not be significantly affected")
def step_then_performance(context):
    # Evaluate the model's performance
    print(context.original_accuracy)
    print(context.noisy_accuracy)
    assert abs(context.original_accuracy - context.noisy_accuracy) < 1
