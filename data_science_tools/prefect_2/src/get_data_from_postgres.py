import hydra
import pandas as pd
from hydra.utils import to_absolute_path as abspath
from prefect import flow, task
from sqlalchemy import create_engine


@task(retries=3)
def read_data(connection):
    engine = create_engine(
        f"postgresql://{connection.user}:{connection.password}@{connection.host}/{connection.database}",
    )
    query = """
    SELECT * FROM train 
    """
    df = pd.read_sql(query, con=engine)
    return df

@task
def save_data(df: pd.DataFrame, save_path: str):
    df.to_csv(abspath(save_path))


@hydra.main(config_path="../config", config_name="get_data", version_base=None)
@flow
def get_data(config):
    df = read_data(config.connection)
    save_data(df, config.data.raw)


if __name__ == "__main__":
    get_data()
