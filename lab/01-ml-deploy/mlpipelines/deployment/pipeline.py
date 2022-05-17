"""
This sample is non-production-ready template
© 2021 Amazon Web Services, Inc. or its affiliates. All Rights Reserved.
This AWS Content is provided subject to the terms of the AWS Customer Agreement available at
http://aws.amazon.com/agreement or other written agreement between Customer and either
Amazon Web Services, Inc. or Amazon Web Services EMEA SARL or both.
"""

import boto3
from botocore.exceptions import ClientError
from datetime import datetime
import logging
from sagemaker import get_execution_role
import sagemaker.session
from sagemaker.tensorflow import TensorFlowModel
import traceback

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

sagemaker_client = boto3.client("sagemaker")

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

def describe_model_package(model_package_arn):
    try:
        model_package = sagemaker_client.describe_model_package(
            ModelPackageName=model_package_arn
        )

        LOGGER.info("{}".format(model_package))

        if len(model_package) == 0:
            error_message = ("No ModelPackage found for: {}".format(model_package_arn))
            LOGGER.error("{}".format(error_message))

            raise Exception(error_message)

        return model_package
    except ClientError as e:
        stacktrace = traceback.format_exc()
        error_message = e.response["Error"]["Message"]
        LOGGER.error("{}".format(stacktrace))

        raise Exception(error_message)

def get_approved_package(model_package_group):
    """Gets the latest approved model package for a model package group.

    Args:
        model_package_group: The model package group name.

    Returns:
        The SageMaker Model Package ARN.
    """
    try:
        # Get the latest approved model package
        response = sagemaker_client.list_model_packages(
            ModelPackageGroupName=model_package_group,
            ModelApprovalStatus="Approved",
            SortBy="CreationTime",
            SortOrder="Descending",
            MaxResults=1,
        )
        approved_packages = response["ModelPackageSummaryList"]

        # Return error if no packages found
        if len(approved_packages) == 0:
            error_message = ("No approved ModelPackage found for ModelPackageGroup: {}".format(model_package_group))
            LOGGER.error("{}".format(error_message))

            raise Exception(error_message)

        model_package = approved_packages[0]
        LOGGER.info("Identified the latest approved model package: {}".format(model_package))

        return model_package
    except ClientError as e:
        stacktrace = traceback.format_exc()
        error_message = e.response["Error"]["Message"]
        LOGGER.error("{}".format(stacktrace))

        raise Exception(error_message)

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

def get_deployed_model():
    try:
        response = sagemaker_client.list_models(
            SortBy="CreationTime",
            SortOrder="Descending",
            MaxResults=1
        )

        model_name = None

        if "Models" in response and len(response["Models"]) > 0:
            model_name = response["Models"][0]["ModelName"]

        return model_name
    except Exception as e:
        stacktrace = traceback.format_exc()
        LOGGER.error("{}".format(stacktrace))

        raise e

def deploy_model(session, model, model_package_group_name, env, inference_instance_count, inference_instance_type):
    try:
        LOGGER.info("Deploying endpoint {}".format(model_package_group_name + "-" + env))

        model.deploy(
            endpoint_name=model_package_group_name + "-" + env,
            initial_instance_count=inference_instance_count,
            instance_type=inference_instance_type,
            update_endpoint=True
        )
    except ClientError as e:
        stacktrace = traceback.format_exc()
        LOGGER.info("{}".format(stacktrace))

        model_name = get_deployed_model()

        update_model(session, model_name, model_package_group_name, env, inference_instance_count, inference_instance_type)

def update_model(session, model_name, model_package_group_name, env, inference_instance_count, inference_instance_type):
    try:
        LOGGER.info("Updating endpoint configuration {}".format(model_package_group_name + "-" + env))

        endpoint_config_name = session.create_endpoint_config(
            name="{}-{}-{}".format(model_package_group_name, env, datetime.today().strftime('%Y-%m-%d-%H-%M-%S')),
            model_name=model_name,
            initial_instance_count=inference_instance_count,
            instance_type=inference_instance_type
        )

        response = sagemaker_client.update_endpoint(
            EndpointName=model_package_group_name + "-" + env,
            EndpointConfigName=endpoint_config_name
        )

        LOGGER.info("Update endpoint {}-{}".format(model_package_group_name, env))
        LOGGER.info(response)

    except Exception as e:
        stacktrace = traceback.format_exc()
        LOGGER.info("{}".format(stacktrace))

        raise e

def get_pipeline(
    region,
    env,
    kms_account_id,
    kms_alias,
    bucket_artifacts,
    bucket_inference,
    inference_artifact_path,
    inference_artifact_name,
    inference_instance_count,
    inference_instance_type,
    model_package_group,
    training_framework_version,
    role=None,
    pipeline_name="DeployPipeline"):
    """Gets a SageMaker ML Pipeline instance working with on abalone data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    sagemaker_session = get_session(region, bucket_inference)

    if role is None:
        role = get_execution_role()

    """
        Global parameters
    """

    kms_key = get_kms_key(region, kms_account_id, kms_alias)

    inference_source_dir = "s3://{}/{}/{}".format(
        bucket_artifacts,
        inference_artifact_path,
        inference_artifact_name
    )

    """
       Get last approved model package
   """

    try:
        model_package_approved = get_approved_package(model_package_group)
        model_package = describe_model_package(model_package_approved["ModelPackageArn"])
    except Exception as e:
        stacktrace = traceback.format_exc()
        LOGGER.error("{}".format(stacktrace))

        raise e

    """
        Get Model from registry
    """

    LOGGER.info("Create SageMaker Model Object")

    model = TensorFlowModel(
        entry_point="inference.py",
        framework_version=str(training_framework_version),
        source_dir=inference_source_dir,
        model_data=model_package["InferenceSpecification"]["Containers"][0]["ModelDataUrl"],
        model_kms_key=kms_key,
        role=role,
        sagemaker_session=sagemaker_session
    )

    deploy_model(
        sagemaker_session,
        model,
        model_package_group,
        env,
        inference_instance_count,
        inference_instance_type)

    # pipeline instance
    pipeline = None

    return pipeline
