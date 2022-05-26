#!/bin/sh

# The name of our algorithm
repo=$1
registry_name=$2
image_tag=$3
docker_file=$4
aws_profile_name=$5

echo "[INFO]: registry_name=${registry_name}"
echo "[INFO]: image_tag=${image_tag}"
echo "[INFO]: docker_file=${docker_file}"
echo "[INFO]: aws_profile_name=${aws_profile_name}"

cd $repo

chmod +x src/*

FINAL_JSON="["

index=1;
target_index=6

if [ -z ${aws_profile_name} ]
then
  echo "[INFO]: AWS_PROFILE_NAME not passed"
  target_index=$((target_index - 1));
else
  export AWS_DEFAULT_PROFILE=${aws_profile_name} || (echo "[INFO]: AWS_PROFILE_NAME not passed" && unset aws_profile_name && target_index=$((target_index - 1)));
fi

j=$#;
while [ $index -le $j ]
do
    if [ "${index}" -ge "${target_index}" ]
    then
      IFS='= ' read -r -a array <<< "$1"

      FINAL_JSON="$FINAL_JSON{\"Key\": \"${array[0]}\",\"Value\": \"${array[1]}\"}, "
    fi

    index=$((index + 1));
    shift 1;
done

FINAL_JSON=${FINAL_JSON%?}
FINAL_JSON=${FINAL_JSON%?}
FINAL_JSON="$FINAL_JSON]"

echo "[INFO]: TAGS=${FINAL_JSON}"

account=$(aws sts get-caller-identity --query Account --output text)

# Get the region defined in the current configuration (default to us-west-2 if none defined)
region=$(aws configure get region)

echo "[INFO]: Region ${region}"

fullname="${account}.dkr.ecr.${region}.amazonaws.com/${registry_name}:${image_tag}"

echo "[INFO]: Image name: ${fullname}"

# If the repository doesn't exist in ECR, create it.

aws ecr describe-repositories --repository-names "${registry_name}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    if [ "$FINAL_JSON" == "[]" ] || [ "$FINAL_JSON" == "]" ] || [ "$FINAL_JSON" == "[" ];
    then
        aws ecr create-repository --repository-name "${registry_name}" > /dev/null
    else
        echo "[INFO]: TAGS:"
        echo "${FINAL_JSON}"
        aws ecr create-repository --repository-name "${registry_name}" --tags "$FINAL_JSON" > /dev/null
    fi
fi

# Get the login command from ECR and execute it directly
password=$(aws ecr --region ${region} get-login-password)

docker login -u AWS -p ${password} "${account}.dkr.ecr.${region}.amazonaws.com"

# Build the docker image locally with the image name and then push it to ECR
# with the full name.

# On a SageMaker Notebook Instance, the docker daemon may need to be restarted in order
# to detect your network configuration correctly.  (This is a known issue.)
if [ -d "/home/ec2-user/SageMaker" ]; then
  sudo service docker restart
fi

docker build -t ${fullname} -f ${docker_file} .

docker push ${fullname}