FROM python:3.7-slim-buster

ARG PYTHON=python3
ARG PYTHON_PIP=python3-pip
ARG PIP=pip3

# Upgrade pip and setuptools
RUN ${PIP} install --upgrade pip
RUN ${PIP} install --upgrade setuptools

RUN ${PIP} install \
    boto3 \
    pandas \
    scikit-learn

ENV PYTHONUNBUFFERED=TRUE

ENTRYPOINT ["python3"]