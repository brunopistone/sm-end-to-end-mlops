from datetime import datetime
from sagemaker.processing import Processor, ProcessingInput, ProcessingOutput

def get_model_monitor_container_uri(region):
    container_uri_format = '{0}.dkr.ecr.{1}.amazonaws.com/sagemaker-model-monitor-groundtruth-merger'

    regions_to_accounts = {
        'eu-north-1': '895015795356',
        'me-south-1': '607024016150',
        'ap-south-1': '126357580389',
        'eu-west-3': '680080141114',
        'us-east-2': '777275614652',
        'eu-west-1': '468650794304',
        'eu-central-1': '048819808253',
        'sa-east-1': '539772159869',
        'ap-east-1': '001633400207',
        'us-east-1': '156813124566',
        'ap-northeast-2': '709848358524',
        'eu-west-2': '749857270468',
        'ap-northeast-1': '574779866223',
        'us-west-2': '159807026194',
        'us-west-1': '890145073186',
        'ap-southeast-1': '245545462676',
        'ap-southeast-2': '563025443158',
        'ca-central-1': '536280801234'
    }

    container_uri = container_uri_format.format(regions_to_accounts[region], region)
    return container_uri

def run_merge_job_processor(
        region,
        instance_type,
        role,
        bucket_name,
        groundtruth_path,
        endpoint_input,
        merge_path,
        instance_count=1,
        kms_key=None):

    groundtruth_input_1 = ProcessingInput(input_name="groundtruth_input_1",
                              source="s3://{}/{}".format(bucket_name, groundtruth_path),
                              destination=f"/opt/ml/processing/groundtruth/{datetime.utcnow():%Y/%m/%d/%H/%M}",
                              s3_data_type="S3Prefix",
                              s3_input_mode="File")

    "/".join(endpoint_input.split("/")[3:])

    endpoint_input_1 = ProcessingInput(input_name="endpoint_input_1",
                                  source="s3://{}/{}".format(bucket_name, endpoint_input),
                                  destination="/opt/ml/processing/input_data/{}".format("/".join(endpoint_input.split("/")[3:])),
                                  s3_data_type="S3Prefix",
                                  s3_input_mode="File")

    output = ProcessingOutput(output_name="result",
                               source="/opt/ml/processing/output",
                               destination=f"s3://{bucket_name}/{merge_path}")

    inputs = [
        groundtruth_input_1,
        endpoint_input_1
    ]

    outputs = [
        output
    ]

    env = {
        'dataset_source': '/opt/ml/processing/input_data',
        'ground_truth_source': '/opt/ml/processing/groundtruth',
        'output_path': '/opt/ml/processing/output',
    }

    processor = Processor(image_uri=get_model_monitor_container_uri(region),
                          instance_count=instance_count,
                          instance_type=instance_type,
                          role=role,
                          env=env,
                          output_kms_key=None if kms_key == None else kms_key)

    return processor.run(
        inputs=inputs,
        outputs=outputs,
        wait=True
    )
