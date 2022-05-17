import appconfig
import logging
import os
import traceback
from BertWrapper import BertModel
from utils import constants, utils

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

def __read_data():
    try:
        LOGGER.info("Reading dataset from source...")

        data = utils.load_data_csv(
            file_path=constants.INPUT_PATH,
            file_name=appconfig.args.input_file,
            percentage=appconfig.args.dataset_percentage)

        data = data.dropna()

        data.Sentiment = data.Sentiment.astype(str)

        data_train, data_test = utils.split_data(data)

        X_train, y_train, X_test, y_test = data_train.text, data_train.Sentiment, data_test.text, data_test.Sentiment

        y_train = [[label] for label in y_train]
        y_test = [[label] for label in y_test]

        y_train = utils.one_hot_encoding(y_train)
        y_test = utils.one_hot_encoding(y_test)

        return X_train, y_train, X_test, y_test
    except Exception as e:
        stacktrace = traceback.format_exc()
        LOGGER.error("{}".format(stacktrace))

        raise e

if __name__ == '__main__':

    bert_model = BertModel("amazon/bort")

    bert_model.retrieve_model()
    tokenizer = bert_model.retrieve_tokenizer()

    '''
    Prepare data
    '''
    X_train, y_train, X_test, y_test = __read_data()

    total_avg, max_length = utils.define_tokenizer_configs(X_train.tolist())

    train_input_ids, train_input_masks, train_input_segments, train_labels = bert_model.tokenize_sequences(int(max_length), X_train, y_train)

    '''
    Train model
    '''

    bert_model.build(int(max_length))

    bert_model.fit(constants.CHECKPOINT_PATH, constants.MODEL_NAME, [train_input_ids, train_input_masks], train_labels, False, False, appconfig.args.epochs)

    '''
    Tensorflow saving
    '''
    bert_model.get_model().save(os.path.join(constants.MODEL_PATH, "saved_model", "1"), save_format='tf')

    '''
    Keras saving
    '''
    # bert_model.get_model().save(os.path.join(constants.MODEL_PATH, constants.MODEL_NAME + ".h5"), save_format='h5')

    '''
    Custom saving
    '''
    # utils.save_model_tf(model, constants.MODEL_PATH, constants.MODEL_NAME)

    '''
    Pickle saving
    '''
    # utils.save_to_pickle(rnn_model.get_tokenizer(), constants.MODEL_PATH, "bort_tokenizer")

    '''
    Pickle saving tokenizer
    '''
    utils.save_to_pickle(tokenizer, constants.MODEL_PATH, "tokenizer")

    utils.define_tokenizer_configs(X_test.tolist())

    test_input_ids, test_input_masks, test_input_segments, test_labels = bert_model.tokenize_sequences(int(max_length), X_test, y_test)

    results = bert_model.get_model().evaluate([test_input_ids, test_input_masks], test_labels, batch_size=100)

    if len(results) > 0:
        LOGGER.info("Test loss: {}".format(results[0]))

    if len(results) > 1:
        LOGGER.info("Test accuracy: {}".format(results[1]))
