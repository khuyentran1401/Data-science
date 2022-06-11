from prefect.deployments import DeploymentSpec
from prefect.flow_runners import SubprocessFlowRunner

DeploymentSpec(
    name="data_engineer",
    flow_location="./data_engineering.py",
    parameters={"test_data_ratio": 0.2},
    flow_runner=SubprocessFlowRunner(),
)
