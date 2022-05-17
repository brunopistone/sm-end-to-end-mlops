import argparse
import boto3
import csv
import logging
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import pathlib
import re
import traceback

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

client = boto3.client("s3")
resource = boto3.resource("s3")

BASE_PATH = os.path.join("/", "opt", "ml")
PROCESSING_PATH = os.path.join(BASE_PATH, "processing")
PROCESSING_PATH_INPUT = os.path.join(PROCESSING_PATH, "input")
PROCESSING_PATH_OUTPUT = os.path.join(PROCESSING_PATH, "output")

def clean_text(text):
    text = text.lower()
    text = re.sub("\[.*?\]", "", text)
    text = re.sub("https?://\S+|www\.\S+", "", text)
    text = re.sub("\n", "", text)
    text = " ".join(filter(lambda x:x[0]!="@", text.split()))
    return text

def download_dir(client, resource, source_prefix, local="/tmp", bucket="your_bucket"):
    paginator = client.get_paginator("list_objects")

    for result in paginator.paginate(Bucket=bucket, Delimiter="/", Prefix=source_prefix):
        if result.get("CommonPrefixes") is not None:
            for subdir in result.get("CommonPrefixes"):
                download_dir(client, resource, subdir.get("Prefix"), local, bucket)

        for file in result.get("Contents", []):
            LOGGER.info("File {}".format(file.get("Key").replace(source_prefix, "")))
            dest_pathname = os.path.join(local, file.get("Key").replace(source_prefix, ""))

            if not os.path.exists(os.path.dirname(dest_pathname)):
                LOGGER.info("Creating os.path.dirname(dest_pathname)")
                os.makedirs(os.path.dirname(dest_pathname))

            if not file.get("Key").endswith("/"):
                LOGGER.info("Download {} in {}".format(file.get("Key"), dest_pathname))
                resource.meta.client.download_file(bucket, file.get("Key"), dest_pathname)

def download_data(args, dir):

    pathlib.Path(dir).mkdir(parents=True, exist_ok=True)

    bucket_input = args.input_data.split("/")[2]
    input_data = "/".join(args.input_data.split("/")[3:])

    if pathlib.Path(input_data).suffix != "":
        input_data = "/".join(args.input_data.split("/")[3:-1])

    boto3.setup_default_session()

    LOGGER.info("Downloading s3://{}/{}/".format(bucket_input, input_data))
    download_dir(client, resource, "{}/".format(input_data), dir, bucket=bucket_input)

    downloaded_files = [f for f in listdir(dir) if isfile(join(dir, f))]

    LOGGER.info("Downloaded files: {}".format(downloaded_files))

def extract_data(file_path, percentage=100):
    try:
        files = [f for f in listdir(file_path) if isfile(join(file_path, f)) and f.endswith(".csv")]
        LOGGER.info("{}".format(files))

        frames = []

        for file in files:
            df = pd.read_csv(
                os.path.join(file_path, file),
                sep=",",
                quotechar='"',
                quoting=csv.QUOTE_ALL,
                escapechar='\\',
                encoding='utf-8',
                error_bad_lines=False
            )

            df = df.head(int(len(df) * (percentage / 100)))

            frames.append(df)

        df = pd.concat(frames)

        return df
    except Exception as e:
        stacktrace = traceback.format_exc()
        LOGGER.error("{}".format(stacktrace))

        raise e

def load_data(df, file_path, file_name):
    try:
        path = os.path.join(file_path, file_name + ".csv")

        LOGGER.info("Saving file in {}".format(path))

        df.to_csv(
            path,
            index=False,
            header=True,
            quoting=csv.QUOTE_ALL,
            encoding="utf-8",
            escapechar="\\",
            sep=","
        )
    except Exception as e:
        stacktrace = traceback.format_exc()
        LOGGER.error("{}".format(stacktrace))

        raise e

def transform_data(df):
    try:
        df = df[["text", "Sentiment"]]

        LOGGER.info("Original count: {}".format(len(df.index)))

        df = df[df["text"].notna()]
        df = df[df["Sentiment"].notna()]

        LOGGER.info("Current count: {}".format(len(df.index)))

        df["text"] = df["text"].apply(lambda x: clean_text(x))
        df["Sentiment"] = df["Sentiment"].map({"Negative": 0, "Neutral": 1, "Positive": 2})

        return df
    except Exception as e:
        stacktrace = traceback.format_exc()
        LOGGER.error("{}".format(stacktrace))

        raise e

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    args = parser.parse_args()

    LOGGER.info("Arguments: {}".format(args))

    download_data(args, PROCESSING_PATH_INPUT)

    df = extract_data(PROCESSING_PATH_INPUT, 100)

    df = transform_data(df)

    load_data(df, PROCESSING_PATH_OUTPUT, "processed_data")
