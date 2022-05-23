## buildspec.sh

This script allows you to create the source artifact to use for SageMaker Training Jobs or SageMaker Processing Jobs

Parameters:
* ALGORITHM_NAME: Mandatory - Name of the algorithm you want to package
* S3_BUCKET_NAME: Optional - S3 bucket name where the package will be uploaded in the path s3://<S3_BUCKET_NAME>/artifact/<ALGORITHM_NAME>

```
.buildspec.sh <ALGORITHM_NAME> <ACCOUNT_ID> <S3_BUCKET_NAME> <KMS_ALIAS>
```

Example:

```
./buildspec.sh training isengard-bpistone-bert-profanity-dev

./buildspec.sh inference isengard-bpistone-bert-profanity-dev
```

## build_image.sh

This script allows you to create a custom docker image and push on ECR

Parameters:
* ALGORITHM_NAME: Mandatory - Name of the algorithm you want to package
* REGISTRY_NAME: Mandatory - Name of the ECR repository you want to use for pushing the image
* IMAGE_TAG: Mandatory - Tag to apply to the ECR image
* DOCKER_FILE: Mandatory - Dockerfile to build
* AWS_PROFILE: Optional - AWS profile to use for pushing the image on ECR

```
./build_image.sh <ALGORITHM_NAME> <REGISTRY_NAME> <IMAGE_TAG> <DOCKER_FILE>
```

Example:

```
./build_image.sh processing bert-pre-processing latest Dockerfile
```