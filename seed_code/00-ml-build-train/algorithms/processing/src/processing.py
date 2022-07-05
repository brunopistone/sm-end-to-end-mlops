import argparse
import boto3
import csv
import datetime
import emoji
import logging
import numpy as  np
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import time
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

    text = text.lstrip()
    text = text.rstrip()

    text = re.sub("\[.*?\]", "", text)
    text = re.sub("https?://\S+|www\.\S+", "", text)
    text = re.sub("\n", "", text)
    text = " ".join(filter(lambda x:x[0]!="@", text.split()))

    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U0001F1F2-\U0001F1F4"  # Macau flag
                               u"\U0001F1E6-\U0001F1FF"  # flags
                               u"\U0001F600-\U0001F64F"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U0001F1F2"
                               u"\U0001F1F4"
                               u"\U0001F620"
                               u"\u200d"
                               u"\u2640-\u2642"
                               "]+", flags=re.UNICODE)

    text = emoji_pattern.sub(r'', text)

    text = emoji.replace_emoji(text, "")

    text = text.replace("u'", "'")

    text = text.encode("ascii", "ignore")
    text = text.decode()

    word_list = text.split(' ')

    for word in word_list:
        if isinstance(word, bytes):
            word = word.decode("utf-8")

    text = " ".join(word_list)

    if not any(c.isalpha() for c in text):
        return ""
    else:
        return text

def convert_date(date):
    date = time.mktime(datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S").timetuple())

    if isinstance(date, float):
        return date
    else:
        return ""

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
        if not os.path.exists(file_path):
            os.makedirs(file_path)

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
        df = df[["user_name", "date", "text", "Sentiment"]]

        LOGGER.info("Original count: {}".format(len(df.index)))

        df = df.dropna()

        df["user_name"] = df["user_name"].apply(lambda x: clean_text(x))
        df["text"] = df["text"].apply(lambda x: clean_text(x))

        df['user_name'] = df['user_name'].map(lambda x: x.strip())
        df['user_name'] = df['user_name'].replace('', np.nan)
        df['user_name'] = df['user_name'].replace(' ', np.nan)

        df['date'] = df['date'].map(lambda x: x.strip())
        df['date'] = df['date'].replace('', np.nan)
        df['date'] = df['date'].replace(' ', np.nan)
        df["date"] = df["date"].apply(lambda x: convert_date(x))

        df['text'] = df['text'].map(lambda x: x.strip())
        df['text'] = df['text'].replace('', np.nan)
        df['text'] = df['text'].replace(' ', np.nan)

        df['Sentiment'] = df['Sentiment'].map(lambda x: x.strip())
        df['Sentiment'] = df['Sentiment'].replace('', np.nan)
        df['Sentiment'] = df['Sentiment'].replace(' ', np.nan)

        df["Sentiment"] = df["Sentiment"].map({"Negative": 0, "Neutral": 1, "Positive": 2})

        df = df.dropna()

        LOGGER.info("Current count: {}".format(len(df.index)))

        return df
    except Exception as e:
        stacktrace = traceback.format_exc()
        LOGGER.error("{}".format(stacktrace))

        raise e

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    LOGGER.info("Arguments: {}".format(args))

    df = extract_data(PROCESSING_PATH_INPUT, 100)

    df = transform_data(df)

    data_train, data_test = train_test_split(df, test_size=0.2)

    load_data(data_train, os.path.join(PROCESSING_PATH_OUTPUT, "train"), "train")
    load_data(data_test, os.path.join(PROCESSING_PATH_OUTPUT, "test"), "test")
