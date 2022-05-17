FROM tensorflow/tensorflow:2.3.1

# Do not change this argument
ARG APP_HOME=/opt/program

# Setup the environment path
ENV PATH="${APP_HOME}:${PATH}"

# Upgrade pip and setuptools
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools

# Set up the program in the image
COPY src ${APP_HOME}

RUN pip3 install -r ${APP_HOME}/requirements_gpu.txt

# Define the working directory
WORKDIR ${APP_HOME}

ENTRYPOINT ["python", "train.py"]