FROM python:3.7-slim-buster

# Upgrade pip and setuptools
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools

RUN pip3 install \
    boto3 \
    emoji \
    numpy \
    pandas \
    scikit-learn

ENTRYPOINT ["python3"]