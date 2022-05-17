FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.4-cpu-py37

# Do not change this argument
ENV PATH="/opt/ml/code:${PATH}"

# this environment variable is used by the SageMaker container to determine our user code directory.
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

# Set up the program in the image
COPY src/requirements.txt ${SAGEMAKER_SUBMIT_DIRECTORY}

RUN pip3 install -r ${SAGEMAKER_SUBMIT_DIRECTORY}/requirements.txt