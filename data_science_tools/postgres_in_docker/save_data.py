import os
from re import L

import hydra
import pandas as pd
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig
from sqlalchemy import create_engine, engine


def download_data(url: str, table_name: str):
    os.system(f"wget {url} -O {table_name}")


@hydra.main(config_path=".", config_name="config", version_base=None)
def write_to_database(config: DictConfig):

    download_data(config.data_url, config.table_name)

    engine = create_engine(
        f"postgresql://{config.username}:{config.password}@{config.host}:{config.port}/{config.database}"
    )
    df = pd.read_csv(abspath(config.table_name))
    df.to_sql(name=config.table_name, con=engine, if_exists="replace")


if __name__ == "__main__":
    write_to_database()
