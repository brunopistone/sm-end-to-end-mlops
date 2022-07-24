FROM tensorflow/tensorflow:2.5.0-jupyter

ARG APP_HOME=/opt/ml/code

# Setup the environment path
ENV PATH="${APP_HOME}:${PATH}"

ARG PYTHON=python3
ARG PYTHON_PIP=python3-pip
ARG PIP=pip3

RUN ${PIP} --no-cache-dir install --upgrade pip

COPY src ${APP_HOME}

RUN ${PIP} install -r ${APP_HOME}/requirements.txt

## SageMaker is running images as docker run train. In order to provide hyperparameters, we are renaming main
## python file to train
RUN mv ${APP_HOME}/serve.py ${APP_HOME}/serve

WORKDIR ${APP_HOME}