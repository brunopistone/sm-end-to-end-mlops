import csv
import html
import json
import logging
import numpy as np
import os
import pandas as pd
import pickle
import re
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_text(text):
    try:
        text = html.unescape(text)
        text = text.replace('"', "")
        text = text.replace('“', "")
        text = text.replace('|', "")
        text = text.replace('~', "")
        text = re.sub('<[^<]+?>', '', text)
        text = re.sub(' +', ' ', text)

        return text
    except Exception as e:
        stacktrace = traceback.format_exc()
        logger.error("{}".format(stacktrace))

        raise e

def define_tokenizer_configs(train_set):
    tmp_tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tmp_tokenizer.fit_on_texts(train_set)

    total_avg = sum(map(len, train_set)) / len(train_set)
    max_length = max([len(x) for x in train_set])

    print('Num words = {0}'.format(len(tmp_tokenizer.word_index)))

    print("Average length: {}".format(total_avg))

    print("Max length: {}".format(max_length))

    return total_avg, max_length

def get_percentage(df, percentage=50):
    df = df.head(int(len(df) * (int(percentage) / 100)))

    return df

def is_json(json_obj):
    try:
        parsed = json.loads(json_obj)
        return True
    except ValueError as e:
        logger.error("{} not a JSON".format(json_obj))
        print("{} not a JSON".format(json_obj))

        return False

def load_bert_model_tf(bert_layer, model_dir, model_name):
    try:
        modelName = os.path.join(model_dir, model_name)

        json_file = open(modelName + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        clf = tf.keras.models.model_from_json(loaded_model_json, custom_objects={"BertModelLayer": bert_layer})
        clf.load_weights(modelName + ".h5")

        logger.info("Loaded {} from disk".format(modelName))

        return clf
    except Exception as e:
        stacktrace = traceback.format_exc()
        logger.error("{}".format(stacktrace))

        raise e

def load_checkpoint_tf(model, model_path, model_name):
    try:
        checkpointName = os.path.join(model_path, model_name + ".ckpt")

        logger.info("Try reading checkpoints from {}".format(checkpointName))

        model.load_weights(checkpointName)
    except Exception as e:
        stacktrace = traceback.format_exc()
        logger.error("{}".format(stacktrace))

        raise e

def load_data_csv(file_path, file_name, sep=",", quotechar='"', percentage=100):
    try:
        logger.info("Try reading {}".format(os.path.join(file_path, file_name)))

        df = pd.read_csv(
            os.path.join(file_path, file_name),
            sep=sep,
            quotechar=quotechar,
            quoting=csv.QUOTE_ALL,
            escapechar='\\',
            encoding='utf-8',
            error_bad_lines=False
        )

        df = get_percentage(df, percentage)

        return df
    except Exception as e:
        stacktrace = traceback.format_exc()
        logger.error("{}".format(stacktrace))

        raise e

def load_model_tf(model_dir, model_name):
    modelName = os.path.join(model_dir, model_name)

    try:
        json_file = open(modelName + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        clf = tf.keras.models.model_from_json(loaded_model_json)
        clf.load_weights(modelName + ".h5")

        logger.info("Loaded {} from disk".format(modelName))

        return clf
    except ValueError as e:
        json_file = open(modelName + '.json', 'r')
        loaded_model_json = json.load(json_file)

        clf = tf.keras.models.model_from_json(loaded_model_json)
        clf.load_weights(modelName + ".h5")

        logger.info("Loaded {} from disk".format(modelName))

        return clf
    except Exception as e:
        stacktrace = traceback.format_exc()
        logger.error("{}".format(stacktrace))

        raise e

def load_pickle(file_path, file_name):
    try:
        path = "{}.pkl".format(os.path.join(file_path, file_name))

        logger.info("Reading files from {}".format(path))

        with open(path, 'rb') as pickle_file:
            data = pickle.load(pickle_file)
        pickle_file.close()

        return data
    except Exception as e:
        stacktrace = traceback.format_exc()
        logger.error("{}".format(stacktrace))

        raise e

def one_hot_encoding(labels):
    y = OneHotEncoder().fit_transform(labels).toarray()

    return y

def save_model_tf(model, model_path, model_name):
    try:
        modelName = os.path.join(model_path, model_name)

        logger.info("Try to save {}".format(modelName))

        model_json = model.to_json()

        with open(modelName + ".json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()

        model.save_weights(modelName + ".h5")

        logger.info("Saved {} to disk".format(modelName))
    except Exception as e:
        stacktrace = traceback.format_exc()
        logger.error("{}".format(stacktrace))

        raise e

def save_to_pickle(data, file_path, file_name):
    try:
        path = os.path.join(file_path, file_name + ".pkl")

        logger.info("Try to save {}".format(path))

        if not os.path.exists(file_path):
            os.mkdir(file_path)

        with open(path, 'wb') as data_pkl:
            pickle.dump(data, data_pkl)

        data_pkl.close()
    except Exception as e:
        stacktrace = traceback.format_exc()
        logger.error("{}".format(stacktrace))

        raise e

def split_data(data, train_percentage=0.8):
    try:
        train, test = train_test_split(data, test_size=1.0 - train_percentage)

        return train, test
    except Exception as e:
        stacktrace = traceback.format_exc()
        logger.error("{}".format(stacktrace))

        raise e

def split_data_balanced(data, labels, train_percentage=0.8):
    X_train, y_train, X_test, y_test = None, None, None, None

    sss = StratifiedShuffleSplit(train_size=train_percentage, n_splits=1,
                                 test_size=1.0 - train_percentage, random_state=0)

    for train_index, test_index in sss.split(data, labels):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

    return X_train, y_train, X_test, y_test

def tokenize_sequence(tokenizer, max_seq_length, data):
    try:
        input_ids = []
        input_masks = []
        input_segments = []

        bert_input = tokenizer.encode_plus(
            data,
            add_special_tokens=True,
            max_length=max_seq_length,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=True
        )

        input_ids.append(bert_input['input_ids'])
        input_masks.append(bert_input['attention_mask'])
        input_segments.append(bert_input['token_type_ids'])

        return bert_input['input_ids'], bert_input['attention_mask'], bert_input['token_type_ids']
    except Exception as e:
        stacktrace = traceback.format_exc()

        logger.error("{}".format(stacktrace))
        print("{}".format(stacktrace))

        raise e

def tokenize_sequences(tokenizer, max_seq_length, data, labels):
    try:
        input_ids = []
        input_masks = []
        input_segments = []

        for sentence in data:
            bert_input = tokenizer.encode_plus(
                sentence,
                add_special_tokens=True,
                max_length=max_seq_length,
                truncation=True,
                padding='max_length',
                return_attention_mask=True,
                return_token_type_ids=True
            )

            input_ids.append(bert_input['input_ids'])
            input_masks.append(bert_input['attention_mask'])
            input_segments.append(bert_input['token_type_ids'])

        return input_ids, input_masks, input_segments
    except Exception as e:
        stacktrace = traceback.format_exc()

        logger.error("{}".format(stacktrace))
        print("{}".format(stacktrace))

        raise e
