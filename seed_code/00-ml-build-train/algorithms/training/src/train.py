from argparse import ArgumentParser
from datetime import datetime
import csv
import glob
import logging
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
import traceback
from transformers import AutoConfig, AutoTokenizer, TFAutoModel

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

DATETIME_NOW = datetime.now().strftime("%Y-%m-%d%H:%M:%S")
MODEL_NAME = "model"
BASE_PATH = os.path.join("/", "opt", "ml")
INPUT_PATH_TRAIN = os.path.join(BASE_PATH, "input", "data", "train")
INPUT_PATH_TEST = os.path.join(BASE_PATH, "input", "data", "test")
INPUT_MODELS_PATH = os.path.join(BASE_PATH, "input", "data", "models")
CHECKPOINT_PATH = os.path.join(BASE_PATH, "checkpoints", DATETIME_NOW)
MODEL_PATH = os.path.join(BASE_PATH, "model")
OUTPUT_PATH = os.path.join(BASE_PATH, "output")
OUTPUT_PATH_DATA = os.path.join(BASE_PATH, "output", "data")
PARAM_FILE = os.path.join(BASE_PATH, "input", "config", "hyperparameters.json")

"""
    Extract info from the dataset
"""
def __define_tokenizer_configs(train_set):
    tmp_tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tmp_tokenizer.fit_on_texts(train_set)

    total_avg = sum(map(len, train_set)) / len(train_set)
    max_length = max([len(x) for x in train_set])

    print('Num words = {0}'.format(len(tmp_tokenizer.word_index)))

    print("Average length: {}".format(total_avg))

    print("Max length: {}".format(max_length))

    return total_avg, max_length

"""
    Read input data
"""
def __read_data(files_path, args):
    try:
        LOGGER.info("Reading dataset from source...")

        all_files = glob.glob(os.path.join(files_path, "*.csv"))

        datasets = []

        for filename in all_files:
            data = pd.read_csv(
                filename,
                sep=',',
                quotechar='"',
                quoting=csv.QUOTE_ALL,
                escapechar='\\',
                encoding='utf-8',
                error_bad_lines=False
            )

            datasets.append(data)

        data = pd.concat(datasets, axis=0, ignore_index=True)

        data = data[["text", "Sentiment"]]

        data = data.dropna()

        data = data.head(int(len(data) * (int(args.dataset_percentage) / 100)))

        data.Sentiment = data.Sentiment.astype(str)

        X, y = data.text, data.Sentiment

        y = [[label] for label in y]

        y = OneHotEncoder().fit_transform(y).toarray()

        return X, y
    except Exception as e:
        stacktrace = traceback.format_exc()
        LOGGER.error("{}".format(stacktrace))

        raise e

"""
    Compile tensorflow model
"""
def compile_model(transformers_model, configs, shape, learning_rate=3e-5, fine_tuning=False):
    try:
        input_ids_in = Input(shape=(int(shape),), name='input_token', dtype='int32')
        input_masks_in = Input(shape=(int(shape),), name='masked_token', dtype='int32')

        encoder, pooler = transformers_model(input_ids_in, attention_mask=input_masks_in)
        X = Dense(117, activation="relu")(pooler)
        X = Dropout(configs.hidden_dropout_prob)(X)
        X = Dense(
            3,
            kernel_initializer=TruncatedNormal(stddev=configs.initializer_range),
            activation="softmax"
        )(X)

        model = Model(
            inputs=[input_ids_in, input_masks_in],
            outputs=[X]
        )

        if fine_tuning:
            for layer in model.layers[:3]:
                layer.trainable = False

        optimizer = Adam(learning_rate=learning_rate)
        loss_obj = CategoricalCrossentropy(from_logits=True)

        model.compile(optimizer=optimizer, loss=loss_obj, metrics=['accuracy'])

        LOGGER.info("{}".format(model.summary()))

        return model
    except Exception as e:
        stacktrace = traceback.format_exc()
        LOGGER.error("{}".format(stacktrace))

        raise e

"""
    Get Huggingface model
"""
def retrieve_transformes(model_name, output_hidden_states=False):
    try:
        configs = AutoConfig.from_pretrained(model_name)
        configs.output_hidden_states = output_hidden_states

        model = TFAutoModel.from_pretrained(model_name, config=configs)

        model = model.bert

        return model, configs
    except Exception as e:
        stacktrace = traceback.format_exc()
        LOGGER.error("{}".format(stacktrace))

        raise e

"""
    Model fit
"""
def train_model(model, checkpoint_path, model_name, X_train, y_train, checkpoints=False, epochs=10, batch_size=100):
    callbacks = []

    if checkpoints:
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)

        checkpointName = os.path.join(checkpoint_path, model_name + ".ckpt")

        # Create a callback that saves the src's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpointName,
                                                         save_weights_only=True,
                                                         verbose=1)

        callbacks.append(cp_callback)

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=int(epochs),
        batch_size=batch_size,
        callbacks=callbacks
    )

    return history

"""
    Get Huggingface tokenizer
"""
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

        return np.asarray(input_ids, dtype='int32'), np.asarray(input_masks, dtype='int32'), np.asarray(
            input_segments, dtype='int32'), np.asarray(labels)
    except Exception as e:
        stacktrace = traceback.format_exc()
        LOGGER.error("{}".format(stacktrace))

        raise e

if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--epochs', type=str, default=5)
    parser.add_argument('--learning_rate', type=float, default=1.45e-4)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--dataset_percentage', type=str, default=100)
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))

    args = parser.parse_args()

    transformers_model, transformers_configs = retrieve_transformes("amazon/bort")
    tokenizer = AutoTokenizer.from_pretrained("amazon/bort")

    '''
    Prepare data
    '''
    X_train, y_train = __read_data(INPUT_PATH_TRAIN, args)
    X_test, y_test = __read_data(INPUT_PATH_TEST, args)

    total_avg, max_length = __define_tokenizer_configs(X_train.tolist())

    train_input_ids, train_input_masks, train_input_segments, train_labels = tokenize_sequences(tokenizer, int(max_length), X_train, y_train)

    '''
    Train model
    '''

    model = compile_model(transformers_model, transformers_configs, int(max_length), args.learning_rate)

    hystory = train_model(model, CHECKPOINT_PATH, MODEL_NAME, [train_input_ids, train_input_masks], train_labels, False, args.epochs, args.batch_size)

    '''
    Tensorflow saving
    '''
    model.save(os.path.join(MODEL_PATH, "saved_model", "1"), save_format='tf')

    '''
    Pickle saving tokenizer
    '''

    path = os.path.join(MODEL_PATH, "tokenizer.pkl")

    LOGGER.info("Try to save {}".format(path))

    if not os.path.exists(MODEL_PATH):
        os.mkdir(MODEL_PATH)

    with open(path, 'wb') as data_pkl:
        pickle.dump(tokenizer, data_pkl)

    """
    Evaluate model on test data
    """
    test_input_ids, test_input_masks, test_input_segments, test_labels = tokenize_sequences(tokenizer, int(max_length), X_test, y_test)

    results = model.evaluate([test_input_ids, test_input_masks], test_labels, batch_size=args.batch_size)

    if len(results) > 0:
        LOGGER.info("Test loss: {}".format(results[0]))

    if len(results) > 1:
        LOGGER.info("Test accuracy: {}".format(results[1]))
