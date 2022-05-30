FROM tensorflow/tensorflow:2.3.1

RUN pip3 install sagemaker-training

# Set up the program in the image
COPY src /opt/ml/code

RUN pip3 install -r ${APP_HOME}/requirements_gpu.txt

ENV SAGEMAKER_PROGRAM train.py