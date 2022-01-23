import mlfoundry as mlf
from data_engineering import data_engineer_flow
from data_science import data_science_flow

# Initialize a new MLFoundryRun
mlf_api = mlf.get_client()
mlf_run = mlf_api.create_run(project_name="Iris-project")

# Run flows
train_test_dict = data_engineer_flow(mlf_run)
data_science_flow(train_test_dict, mlf_run)
