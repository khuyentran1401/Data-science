from prefect import Flow
from prefect.tasks.prefect import StartFlowRun

data_engineering_flow = StartFlowRun(
    flow_name="data-engineer", project_name='Iris Project', wait=True, parameters={'test_data_ratio': 0.3})
data_science_flow = StartFlowRun(
    flow_name="data-science", project_name='Iris Project', wait=True)

with Flow("main-flow") as flow:
    result = data_science_flow(upstream_tasks=[data_engineering_flow])

flow.run()
