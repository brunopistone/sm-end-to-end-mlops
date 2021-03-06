# SageMaker End to End lab for Building, Training, and Deploying ML models

In this repository, we are stepping through an end to end implementation of Machine Learning (ML) models using Amazon SageMaker,
by deploying versioned models stored in the [Amazon SageMaker Model Registry](https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html) 
for real-time inference using [Amazon SageMaker Hosting Services](https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html).

This is a sample code repository for demonstrating how to organize your code for build and train your model, by starting from 
an implementation through notebooks for arriving to a code structure architecture for implementing ML pipeline using Amazon 
SageMaker Pipeline, and how to setup a repository for deploying ML models using CI/CD.

This repository is enriched by CloudFormation templates for setting up the ML environment, by creating the SageMaker Studio 
environment, Networking, and CI/CD for deploying ML models.

Everything can be tested by using the example notebooks for running training on SageMaker using the following frameworks:
* [TensorFlow](https://www.tensorflow.org/)
* [Huggingface - Amazon Bort](https://huggingface.co/amazon/bort)

## Machine Learning Task

In this example, we are performing a Sentiment Analysis task by creating a multiclass classification model using 
[Huggingface - Amazon Bort](https://huggingface.co/amazon/bort) and training on a dataset of Negative, Neutral, Positive tweets.

* Negative - 0
* Neutral - 1
* Positive - 2

## Content overview

In this repository, you will cover an end-to-end approach for building, training, deploying, and monitoring a ML model for 
fraud detection by using Amazon SageMaker.

1. Data Visualization using Amazon SageMaker Studio Notebooks
2. Prepare data for training using Amazon SageMaker Processing, by using [FrameworkProcessor](https://sagemaker.readthedocs.io/en/stable/api/training/processing.html) 
for installing custom python modules
3. Store feature using [Amazon SageMaker Feature Store](https://docs.aws.amazon.com/sagemaker/latest/dg/feature-store-getting-started.html)
4. Train, and version ML models using Amazon SageMaker Jobs and Amazon SageMaker Model Registry
5. Build a custom container using the [sagemaker-training-toolkit](https://github.com/aws/sagemaker-training-toolkit) for 
training ML models
6. Find hyperparameters combination using Amazon SageMaker Hyperparameter Optimization Jobs
7. Deploy Real-Time endpoints using Amazon SageMaker Hosting Services
8. Monitor the quality of your model using Amazon SageMaker Model Monitoring

## Environment Setup

Setup the ML environment by deploying the [CloudFormation](./infrastructure_templates) templates described as below:

1. [00-networking](./infrastructure_templates/00-networking/template.yml): This template is creating a networking resources,  
such as VPC, Private Subnets, Security Groups, for a secure environment for Amazon SageMaker Studio. The necessary variables 
used by SageMaker Studio are stored using [AWS Systems Manager](https://docs.aws.amazon.com/systems-manager/latest/userguide/what-is-systems-manager.html)
2. [01-sagemaker-studio](./infrastructure_templates/01-sagemaker-studio-environment/template.yml): This template is creating 
the SageMaker Studio environment, with the necessary execution role used during the experimentation and the execution of the 
SageMaker Jobs. *Parameters*:
   1. *SageMakerDomainName*: Name to assign to the Amazon SageMaker Studio Domain. *Mandatory* 
   2. *SecurityGroupId*: Provide a Security Group for studio if you want to use your own networking setup, otherwise the parameter
   is read by using AWS SSM after the deployment of the template [00-networking](./infrastructure_templates/00-networking/template.yml). *Optional*
   3. *SubnetId*: Provide a Subnet (Public or Private) for studio if you want to use your own networking setup, otherwise the parameter
   is read by using AWS SSM after the deployment of the template [00-networking](./infrastructure_templates/00-networking/template). *Optional*
   4. *VpcId*: Provide a Vpc ID for studio if you want to use your own networking setup, otherwise the parameter is read by 
   using AWS SSM after the deployment of the template [00-networking](./infrastructure_templates/00-networking/template). *Optional*
3. [02-ci-cd](./infrastructure_templates/02-ci-cd/template.yml): This template is creating the CI/CD pipelines using 
[AWS CodeBuild](https://docs.aws.amazon.com/codebuild/latest/userguide/welcome.html) and [AWS CodePipeline](https://docs.aws.amazon.com/codepipeline/latest/userguide/welcome.html).
It creates two CI/CD pipelines, linked to two AWS CodeCommit repositories, one for training and one for deployment, that can 
be triggered through pushes on the main branch or with the automation part deployed in the next stack. *Parameters*:
   1. *AccountIdDev*: AWS Account ID where Data Scientists will develop ML code artifacts. If not specified, the AWS current account 
   will be used. *Optional*
   2. *PipelineSuffix*: Suffix to use for creating the CI/CD pipelines. *Mandatory*
   3. *RepositoryTrainingName*: Name for the repository where the build and train code will be stored. *Mandatory*
   4. *RepositoryDeploymentName*:  Name for the repository where the deployment code will be stored. *Mandatory*
   5. *S3BucketArtifacts*: Name of the Amazon S3 Bucket that will be created in the next stack used for storing code and model artifacts. *Mandatory*
4. [03-ml-environment](./infrastructure_templates/03-ml-environment/template.yml): This template is creating the necessary resources for the 
ML workflow, such as Amazon S3 bucket for storing code and model artifacts, [Amazon SageMaker Model Registry](https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html) 
for versioning trained ML models, and [Amazon EventBridge Rule](https://docs.aws.amazon.com/eventbridge/latest/userguide/eb-rules.html) 
for monitoring updates in the SageMaker Model Registry and start the CI/CD pipeline for deploying ML models in the production environments.
*Parameters*:
   1. AccountIdTooling: AWS Account ID where CI/CD is deployed. Can be empty if you have deployed the complete stack. In this 
   case, the parameter is read through SSM. *Optional*
   2. CodePipelineDeploymentArn: ARN of the CodePipeline used for deploying ML models. Can be empty if you have deployed the complete stack. In this 
   case, the parameter is read through SSM. *Optional*
   3. KMSAlias: Alias to use for the KMS Key for encrypting data. *Optional*
   4. ModelPackageGroupDescription: Description of the Amazon SageMaker Model Package Group. This value can be empty. *Optional*
   5. ModelPackageGroupName: Name of the Amazon SageMaker Model Package Group where ML models will be stored. *Mandatory*
   6. S3BucketName: Name for the S3 bucket where ML model artifacts, code artifacts, and data will be stored. *Mandatory*


## Labs

### Build and Train ML models

The code structure defined for the [Build and Train ML models](seed_code/00-model-build-train) is the following:

* [algorithms](seed_code/00-ml-build-train/algorithms): The code used by the ML pipelines for processing and training ML models is stored in this folder
  * [algorithms/processing](seed_code/00-ml-build-train/algorithms/processing): This folder contains the python code for performing processing of data
  using Amazon SageMaker Processing Jobs
  * [algorithms/training](seed_code/00-ml-build-train/algorithms/training): This folder contains the python code for training a custom ML model 
  using Amazon SageMaker Training Jobs
* [mlpipelines](seed_code/00-ml-build-train/mlpipelines): This folder contains some utilities scripts created in the official AWS example 
[Amazon SageMaker secure MLOps](https://github.com/aws-samples/amazon-sagemaker-secure-mlops) and it contains the definition for the 
Amazon SageMaker Pipeline used for training
  * [mlpipelines](seed_code/00-ml-build-train/mlpipelines/training): This folder contains the python code for the ML pipelines used for training
* [notebooks](seed_code/00-model-build-train/notebooks): This folder contains the lab notebooks to use for this workshop:
  * [notebooks/00-Data-Visualization](seed_code/00-ml-build-train/notebooks/00-Data-Visualization.ipynb): Explore the input data and test the processing scripts 
  in the notebook
  * [notebooks/01-Prepare-Data-ML-Framework-Container](seed_code/00-ml-build-train/notebooks/01-Prepare-Data-ML-Framework-Container.ipynb): Define a Python Script and create jobs for data processing using 
  Amazon SageMaker Processing
  * [notebooks/02-Prepare-Data-ML-Custom-Script-Container](seed_code/00-ml-build-train/notebooks/02-Prepare-Data-ML-Custom-Script-Container.ipynb): Create jobs for data processing using 
  Amazon SageMaker Processing with a Custom Container by providing a Custom Script
  * [notebooks/03-Prepare-Data-ML-Custom-Container](seed_code/00-ml-build-train/notebooks/03-Prepare-Data-ML-Custom-Container.ipynb): Create jobs for data processing using 
  Amazon SageMaker Processing with by using a Custom Container (*BYOC*)
  * [notebooks/04-Store-Features](seed_code/00-ml-build-train/notebooks/04-Store-Features.ipynb): Store features from prepared data using 
  Amazon SageMaker Feature Store
  * [notebooks/05-Training-Build-Model-Framework-Container](seed_code/00-ml-build-train/notebooks/05-Train-Build-Model-Framework-Container.ipynb): Train a ML model using SageMaker Training, Register 
  the trained model version by using Amazon SageMaker Model Registry.
  * [notebooks/06-Train-Build-Model-Custom-Script-Containerl](seed_code/00-ml-build-train/notebooks/06-Train-Build-Model-Custom-Script-Container.ipynb): 
  Train a ML model using SageMaker Training with a *Custom Container* by providing *Custom Scripts*, and Register the trained model version by using Amazon SageMaker Model Registry.
  * [notebooks/07-Train-Build-Model-Custom-Container](seed_code/00-ml-build-train/notebooks/07-Train-Build-Model-Custom-Container.ipynb): 
  Train a ML model using SageMaker Training by using a Custom Container (*BYOC*), and Register the trained model version by using Amazon SageMaker Model Registry.
  * [notebooks/08-Hyperparameter-Optimization](seed_code/00-ml-build-train/notebooks/08-Hyperparameter-Optimization.ipynb): Identify 
  the best configuration set of hyperparameters for your ML algorithm by using Amazon SageMaker Hyperparameter Optimization 
  * [notebooks/09-SageMaker-Pipeline-Training](seed_code/00-ml-build-train/notebooks/09-SageMaker-Pipeline-Training.ipynb): Define 
  the workflow steps and test the entire end to end using Amazon SageMaker Pipeline

### Deploy ML models

The code structure defined for the [Deploy ML models](seed_code/01-ml-deploy) is the following:

* [algorithms](seed_code/01-ml-deploy/algorithms): The code used by Amazon SageMaker Endpoint for performing inference is stored in this folder
  * [algorithms/inference](seed_code/01-ml-deploy/algorithms/inference): This folder contains the python code for performing inference 
  using Amazon SageMaker Endpoints
* [mlpipelines](seed_code/01-ml-deploy/mlpipelines): This folder contains some utilities scripts created in the official AWS example
[Amazon SageMaker secure MLOps](https://github.com/aws-samples/amazon-sagemaker-secure-mlops) and it contains the scripts to run 
through CI/CD for creating or updating Amazon SageMaker Endpoints
* [notebooks](seed_code/01-ml-deploy/notebooks): This folder contains the lab notebooks to use for this workshop:
  * [notebooks/00-SageMaker-Endpoint](seed_code/01-ml-deploy/notebooks/00-Deploy-SageMaker-Endpoint.ipynb): This notebook shows you how to 
  deploy a trained ML model taken from the Amazon SageMaker Model Registry
  * [notebooks/01-Model-Monitor](seed_code/01-ml-deploy/notebooks/01-Model-Monitor.ipynb): This notebook shows you how to 
  create Amazon SageMaker Model Monitor jobs for monitoring model quality on the deployed endpoint
  * [notebooks/01-Pipeline-Deployment](seed_code/01-ml-deploy/notebooks/02-Pipeline-Deployment.ipynb): Define 
  the workflow steps and test the entire end to end using the script for CI/CD deployment

## CI/CD

The [CloudFormation templates](./infrastructure_templates) provided are creating a fully worked ML environment with CI/CD pipelines for automating the training 
of ML models and the deployment of real-time endpoints.

By starting from the AWS CodeCommit repositories created, you can customize the execution of CI/CD pipelines by editing the configurations 
for the repositories *ml-build-train* and *ml-deploy*.

![AWS CodeCommit Repositories](./docs/code-repos.png "AWS CodeCommit Repositories")

Examples:

```
training:
    pipeline_name: MLOpsTrainPipeline
    region: eu-west-1
    role: # Execution role name for SageMaker
    kms_account_id: # AWS Account ID where the KMS key was created
    kms_alias: ml-kms
    bucket_name: # Amazon S3 Bucket where ML models will be stored
    inference_instance_type: ml.m5.xlarge
    model_package_group_name: ml-end-to-end-group
    processing_artifact_path: artifact/processing
    processing_artifact_name: sourcedir.tar.gz
    processing_framework_version: 0.23-1
    processing_instance_count: 1
    processing_instance_type: ml.t3.large
    processing_input_files_path: data/input
    processing_output_files_path: data/output
    training_artifact_path: artifact/training
    training_artifact_name: sourcedir.tar.gz
    training_output_files_path: models
    training_framework_version: 2.4
    training_python_version: py37
    training_instance_count: 1
    training_instance_type: ml.p2.xlarge
    training_hyperparameters:
        epochs: 5
        learning_rate: 1.45e-4
        batch_size: 100
```

```
deployment:
    pipeline_name: MLOpsDeploymentPipeline
    region: eu-west-1
    kms_account_id: # AWS Account ID where the KMS key was created
    kms_alias: ml-kms
    env: dev
    role:
    bucket_artifacts: # Amazon S3 Bucket where ML model artifacts are stored
    bucket_inference: # Amazon S3 Bucket for ML inference
    inference_artifact_path: artifact/inference
    inference_artifact_name: sourcedir.tar.gz
    inference_instance_count: 1
    inference_instance_type: ml.m5.xlarge
    model_package_group: ml-end-to-end-group
    monitoring_output_path: data/monitoring/captured
    training_framework_version: 2.4
```

The pushes in the two AWS CodeCommit repositories will automate the execution of the AWS CodePipeline for executing the training 
using the [Amazon SageMaker Pipeline](./seed_code/00-ml-build-train/mlpipelines/training/pipeline.py) and the [deployment](./seed_code/01-ml-deploy/mlpipelines/deployment/pipeline.py)
of a real-time endpoint.

![AWS CodePipelines](./docs/code-pipelines.png "AWS CodePipelines")

For automating the deployment of a ML model, an Amazon EventBridge Rule for monitoring updates in the Amazon SageMaker Model Package group is created, 
by targeting the AWS CodePipeline for deployment.

![AWS EventBridge](./docs/event-rule.png "AWS EventBridge")