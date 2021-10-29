from prefect import task, Flow, Parameter


@task(log_stdout=True)
def process_data(name: str):
    print(f"Process data {name}!")


with Flow("Process Flow") as flow:
    name = Parameter('name')
    process_data(name)


flow.run(name='data1') 
flow.run(name='data2') 

flow.register('processing_flow')