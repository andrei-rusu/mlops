from azure.ai.ml import MLClient, command, Input
from azure.identity import DefaultAzureCredential

ml_client = MLClient.from_config(credential=DefaultAzureCredential())

# Define the command job
command_job = command(
    code="./src/model",
    command="python train.py --training_data ${{inputs.training_data}} --reg_rate ${{inputs.reg_rate}}",
    inputs={
        "training_data": Input(type="uri_folder", path="../production/data"),
        "reg_rate": 0.01
    },
    environment="AzureML-sklearn-1.0-ubuntu20.04-py38-cpu@latest",
    compute="local",
    experiment_name="diabetes-train",
    description="Job to train a diabetes model."
)

# Submit the command job
returned_job = ml_client.jobs.create_or_update(command_job)

# Print the job details
print(returned_job)