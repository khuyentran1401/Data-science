from data_engineering import data_engineer
from data_science import data_science
from prefect import flow


@flow
def main():
    data_engineer(test_data_ratio=0.2)
    data_science()


main()
