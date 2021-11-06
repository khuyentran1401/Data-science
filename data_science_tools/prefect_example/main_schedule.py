from prefect import Flow 
from prefect.tasks.prefect import StartFlowRun
from datetime import timedelta, datetime
from prefect.schedules import IntervalSchedule

schedule = IntervalSchedule(
    start_date=datetime.utcnow() + timedelta(seconds=1),
    interval=timedelta(minutes=1),
)

data_engineering_flow = StartFlowRun(flow_name="data-engineer", project_name='Iris Project')
data_science_flow = StartFlowRun(flow_name="data-science", project_name='Iris Project')

with Flow("main-flow", schedule=schedule) as flow:
    data_science = data_science_flow(upstream_tasks=[data_engineering_flow])
    
# flow.register(project_name="Iris Project")
flow.run()