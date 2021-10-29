from prefect import task, Flow, Parameter
import prefect 


@task(log_stdout=True)
def process_data(name: str):
    print(f"Process data {name}!")


with Flow("Process Flow") as flow:
    name = Parameter('name', default=['Khuyen', 'Ben', 'Gavin'])
    process_data(name)


flow.register('processing_flow')