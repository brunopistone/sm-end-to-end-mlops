#!/bin/sh

STACK_NAME=$1
S3_BUCKET_NAME=$2

echo "[INFO] Running deploy.sh"
echo "[INFO] Stack Name: ${STACK_NAME}"
echo "[INFO] S3 Bucket: ${S3_BUCKET_NAME}"

if [ -z ${STACK_NAME} ] ;
then
  echo "STACK_NAME not passed"
  exit 1
fi

if [ -z ${S3_BUCKET_NAME} ] ;
then
  echo "S3_BUCKET_NAME not passed"
  exit 1
fi

cd ${STACK_NAME} || exit

while [[ ${STACK_NAME} =~ ^[0-9] ]] || [[ ${STACK_NAME} =~ ^- ]]; do
  STACK_NAME="${STACK_NAME:1}"
done

template_file=template.yml
template_name="${template_file%.*}"
output_file_name="${template_name}_packaged.yml"

echo "[INFO] File - ${template_file}"
echo "[INFO] Basename - ${template_name}"

aws cloudformation package --template-file $template_file --output-template-file $output_file_name --s3-bucket $S3_BUCKET_NAME

aws cloudformation deploy \
   --template-file $output_file_name \
   --stack-name "${STACK_NAME}-${template_name}" \
   --capabilities "CAPABILITY_NAMED_IAM" \
   --parameter-overrides file://params/config.json \
   --no-fail-on-empty-changeset

rm -rf "${output_file_name}"

echo "[INFO] Deployed Infrastructure..."