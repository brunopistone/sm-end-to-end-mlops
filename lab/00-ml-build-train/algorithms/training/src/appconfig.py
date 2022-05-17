from argparse import ArgumentParser
import logging
import os
import traceback

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

"""
Put Sagemaker Hyperparameters HERE
"""
def initialize_parameters(parser):
    parser.add_argument('--epochs', type=str, default=5)
    parser.add_argument('--dataset_percentage', type=str, default=100)
    parser.add_argument('--input_file', type=str, default=None)
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))

    # '''
    #     For SKLearn framework
    # '''
    # parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    '''
        For the other framework
    '''
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))

    return parser.parse_args()


def retrieve_environment_variables():
    envs = {}

    envs["ROOT_DIR"] = os.path.dirname(os.path.realpath(__file__))

    return envs


def retrieve_parameters():
    parser = ArgumentParser()

    try:
        args = initialize_parameters(parser)
    except KeyError as e:
        stacktrace = traceback.format_exc()
        LOGGER.error("{}".format(stacktrace))
        # this happens when running form a notebook, not from the command line
        args = {}

    envs = retrieve_environment_variables()

    return args, envs

args, envs = retrieve_parameters()
