FROM tensorflow/tensorflow:2.4.1

RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools

# Installing Framework Container modules
#ADD dist/custom_framework_training-1.0.0.tar.gz /custom_framework_training-1.0.0.tar.gz

# Installing code modules
ADD src /src
## Installing these modules
#fsspec
#h5py
#nltk
#params-flow==0.8.2
#sentencepiece==0.1.91
#transformers==3.5.0
RUN pip3 install -r /src/requirements.txt
#RUN pip3 install /custom_framework_training-1.0.0.tar.gz

RUN rm -rf /src
#RUN rm -rf /custom_framework_training-1.0.0.tar.gz

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib" \
    PYTHONIOENCODING=UTF-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

#ENV SAGEMAKER_TRAINING_MODULE custom_framework_training.training:main

RUN pip3 install sagemaker-training