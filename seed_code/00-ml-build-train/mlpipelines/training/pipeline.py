import boto3
import logging
import sagemaker
from sagemaker import get_execution_role
from sagemaker.inputs import TrainingInput
from sagemaker.processing import ProcessingInput, ProcessingOutput
import sagemaker.session
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.tensorflow import TensorFlow
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
import traceback

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

def get_kms_key(
        region,
        account_id,
        kms_alias):
    try:
        kms_key = "arn:aws:kms:{}:{}:alias/{}".format(region, account_id, kms_alias)

        return kms_key
    except Exception as e:
        stacktrace = traceback.format_exc()
        LOGGER.error("{}".format(stacktrace))

        raise e

def get_session(
        region,
        default_bucket):
    try:
        boto_session = boto3.Session(region_name=region)

        sagemaker_client = boto_session.client("sagemaker")
        runtime_client = boto_session.client("sagemaker-runtime")

        return sagemaker.session.Session(
            boto_session=boto_session,
            sagemaker_client=sagemaker_client,
            sagemaker_runtime_client=runtime_client,
            default_bucket=default_bucket
        )
    except Exception as e:
        stacktrace = traceback.format_exc()
        LOGGER.error("{}".format(stacktrace))

        raise e

def get_pipeline(
    region,
    kms_account_id,
    kms_alias,
    bucket_name,
    inference_instance_type,
    model_package_group_name,
    processing_entrypoint,
    processing_framework_version,
    processing_instance_count,
    processing_instance_type,
    processing_input_files_path,
    processing_output_files_path,
    training_artifact_path,
    training_artifact_name,
    training_output_files_path,
    training_framework_version,
    training_python_version,
    training_instance_count,
    training_instance_type,
    training_hyperparameters={},
    role=None,
    pipeline_name="TrainingPipeline"):
    """Gets a SageMaker ML Pipeline instance working with on abalone data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    sagemaker_session = get_session(region, bucket_name)

    if role is None:
        role = get_execution_role()

    """
        Global parameters
    """

    kms_key = get_kms_key(region, kms_account_id, kms_alias)

    """
        Pipeline inputs
    """

    epochs = ParameterString(
        name="Epochs", default_value=""
    )

    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )

    model_package_group_name = ParameterString(
        name="ModelPackageGroupName", default_value=model_package_group_name
    )

    processing_inputs_param = ParameterString(
        name="ProcessingInput", default_value="s3://{}/{}".format(bucket_name, processing_input_files_path)
    )


    """
        Processing parameters
    """

    processing_inputs = [
        ProcessingInput(
            source=processing_inputs_param,
            destination="/opt/ml/processing/input"
        )
    ]

    processing_outputs = [
        ProcessingOutput(output_name="output",
                         source="/opt/ml/processing/output",
                         destination="s3://{}/{}".format(bucket_name, processing_output_files_path))]

    """
        Processing step
    """

    processor = SKLearnProcessor(
        framework_version=str(processing_framework_version),
        role=role,
        instance_count=processing_instance_count,
        instance_type=processing_instance_type,
        output_kms_key=kms_key,
        sagemaker_session=sagemaker_session
    )

    step_process = ProcessingStep(
        name="ProcessData",
        code=processing_entrypoint,
        processor=processor,
        inputs=processing_inputs,
        outputs=processing_outputs
    )

    """
        Training parameters
    """

    if epochs != "":
        training_hyperparameters["epochs"] = epochs

    source_dir = "s3://{}/{}/{}".format(
        bucket_name,
        training_artifact_path,
        training_artifact_name
    )

    training_input = TrainingInput(
        s3_data="s3://{}/{}/train".format(bucket_name, processing_output_files_path),
        content_type="text/csv"
    )

    test_input = TrainingInput(
        s3_data="s3://{}/{}/test".format(bucket_name, processing_output_files_path),
        content_type="text/csv"
    )

    output_path = "s3://{}/{}".format(
        bucket_name,
        training_output_files_path
    )

    """
        Training step
    """

    estimator = TensorFlow(
        entry_point="train.py",
        framework_version=str(training_framework_version),
        py_version=str(training_python_version),
        source_dir=source_dir,
        output_path=output_path,
        hyperparameters=training_hyperparameters,
        enable_sagemaker_metrics=True,
        metric_definitions=[
            {
                'Name': 'Test accuracy',
                'Regex': 'Test accuracy:.* ([0-9\\.]+)'
            }
        ],
        role=role,
        instance_count=training_instance_count,
        instance_type=training_instance_type,
        output_kms_key=kms_key,
        disable_profiler=True
    )

    step_train = TrainingStep(
        depends_on=[step_process],
        name="TrainModel",
        estimator=estimator,
        inputs={
            "train": training_input,
            "test": test_input
        }
    )

    step_register_model = RegisterModel(
        name="RegisterModel",
        estimator=estimator,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=[inference_instance_type],
        transform_instances=[inference_instance_type]
    )

    # pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            epochs,
            model_approval_status,
            model_package_group_name,
            processing_inputs_param
        ],
        steps=[
            step_process,
            step_train,
            step_register_model
        ],
        sagemaker_session=sagemaker_session
    )

    return pipeline
