import hydra
import pandas as pd
from hydra.utils import to_absolute_path as abspath
from prefect import flow, task
from sqlalchemy import create_engine


@task(retries=3)
def read_data(connection, database: str):
    engine = create_engine(
        f"postgresql://{connection.user}:{connection.password}@{connection.host}/{connection.database}",
    )
    query = f"SELECT * FROM {database}"
    df = pd.read_sql(query, con=engine)
    return df


@task
def save_data(df: pd.DataFrame, save_path: str):
    df.to_csv(abspath(save_path))


@hydra.main(config_path="../config", config_name="get_data", version_base=None)
@flow
def get_data(config):
    for database, save_path in config.data.items():
        df = read_data(config.connection, database)
        save_data(df, save_path)


if __name__ == "__main__":
    get_data()
