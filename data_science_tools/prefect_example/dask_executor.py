from prefect import task, Flow, Parameter
import time 
from prefect.executors import LocalDaskExecutor
from prefect.run_configs import LocalRun


@task(log_stdout=True)
def process_data(name: str):
    time.sleep(10)
    print(f"Process data {name}!")


with Flow("Process Flow") as flow:
    name = Parameter('name', default=['Khuyen', 'Ben', 'Gavin'])
    process_data.map(name)


# flow.run_config = LocalRun()
flow.executor = LocalDaskExecutor()

flow.register('processing_flow')