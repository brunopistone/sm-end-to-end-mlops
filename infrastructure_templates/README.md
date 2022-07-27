## Infrastructure folder

The cloudformation scripts in this folder allows you to create the Amazon SageMaker Studio domain, Amazon Sagemaker User 
Profile and the Simulated edge environment

### Deployment order

1. 00-networking
2. 01-sagemaker-studio-environment
3. 02-ci-cd
4. 03-ml-environment

### Script buildspec.sh

Parameters:
* STACK_NAME: Mandatory - name of the stack we want to deploy
* S3_BUCKET_NAME: Mandatory - bucket name used for the deployment

```
./buildspec.sh <STACK_NAME> <S3_BUCKET_NAME>
```
 
### Examples:

#### Networking

```
./buildspec.sh 00-networking test-bucket
```

#### SageMaker Studio Environment

```
./buildspec.sh 01-sagemaker-studio-environment test-bucket
```

#### CI/CD

```
./buildspec.sh 02-ci-cd test-bucket
```

#### ML Environment

```
./buildspec.sh 03-ml-environment test-bucket
```