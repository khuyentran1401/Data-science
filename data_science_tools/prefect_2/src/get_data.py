import json
from collections import OrderedDict
from unicodedata import name

import hydra
from dotenv import dotenv_values
from hydra.utils import to_absolute_path as abspath
from prefect import flow, task
from rauth import OAuth2Service, OAuth2Session

# ---------------------------------------------------------------------------- #
#                                 Create tasks                                 #
# ---------------------------------------------------------------------------- #

@task
def get_env_vars():
    return dotenv_values(".env")

@task(retries=3)
def get_session(vars: OrderedDict):
    data = {"grant_type": "client_credentials"}

    service = OAuth2Service(
            client_id=vars["CLIENT_ID"],
            client_secret=vars["CLIENT_SECRET"],
            access_token_url=vars["TOKEN_URL"],
        )
        
    session = service.get_auth_session(data=data, decoder=json.loads)
    return session

@task 
def get_api(base_api: str, filters: list):


@task(retries=3)
def get_data(session: OAuth2Session, api: str):
    response = session.get(api)
    return response.json()

@task 
def save_data(data: dict, save_location: str):
    with open(abspath(save_location), 'w') as f:
        json.dump(data, f)

# ---------------------------------------------------------------------------- #
#                                 Create a flow                                #
# ---------------------------------------------------------------------------- #

@hydra.main(config_path="../config", config_name="process", version_base=None)
@flow 
def get_data_from_api(config):
    vars = get_env_vars()
    session = get_session(vars)
    data = get_data(session, config.api)
    save_data(data, config.data.raw)


if __name__ == "__main__":
    get_data_from_api()