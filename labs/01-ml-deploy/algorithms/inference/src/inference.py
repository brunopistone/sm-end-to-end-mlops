import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "boto3"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "params-flow==0.8.2"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "sentencepiece==0.1.91"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow==2.4.1"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers==3.5.0"])

import boto3
import json
import logging
import numpy as np
from services import TranslateService
import traceback
from utils import constants, utils

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

tokenizer = utils.load_pickle(constants.MODEL_PATH, "tokenizer")
translate_service = TranslateService()

comprehend_client = boto3.client("comprehend")

MODEL_SHAPE = 144

def __detect_language(body):
    try:
        results = comprehend_client.detect_dominant_language(Text=body)

        max_result = max(results["Languages"], key=lambda x: x['Score'])

        return max_result["LanguageCode"]
    except Exception as e:
        stacktrace = traceback.format_exc()
        LOGGER.error("{}".format(stacktrace))

        raise e

def __encode(tokenizer, shape, body):
    input_ids, input_masks, input_segments = utils.tokenize_sequence(tokenizer, shape, body)

    return input_ids, input_masks, input_segments

def input_handler(data, context):
    try:
        data_str = data.read().decode("utf-8")
        LOGGER.info("data_str: {}".format(data_str))
        LOGGER.info("type data_str: {}".format(type(data_str)))

        jsonlines = data_str.split("\n")
        LOGGER.info("jsonlines: {}".format(jsonlines))
        LOGGER.info("type jsonlines: {}".format(type(jsonlines)))

        transformed_instances = []

        for jsonline in jsonlines:
            LOGGER.info("jsonline: {}".format(jsonline))
            LOGGER.info("type jsonline: {}".format(type(jsonline)))

            review_body = json.loads(jsonline)["features"][0]

            start_lan = __detect_language(review_body)

            if start_lan != "en":
                review_body = translate_service.translate_string(review_body, start_lan, "en")

                LOGGER.info("Translated sentence: {}".format(data))
            else:
                LOGGER.info("Detected en language")

            review_body = utils.clean_text(review_body)

            LOGGER.info("""review_body: {}""".format(review_body))

            input_ids, input_masks, input_segments = __encode(tokenizer, MODEL_SHAPE, review_body)

            transformed_instance = {"input_token": input_ids, "masked_token": input_masks}

            transformed_instances.append(transformed_instance)

        transformed_data = {"signature_name": "serving_default", "instances": transformed_instances}

        transformed_data_json = json.dumps(transformed_data)
        LOGGER.info("transformed_data_json: {}".format(transformed_data_json))

        return transformed_data_json
    except Exception as e:
        stacktrace = traceback.format_exc()
        LOGGER.error("{}".format(stacktrace))

        raise e

def output_handler(response, context):
    try:
        LOGGER.info("response: {}".format(response))
        response_json = response.json()
        LOGGER.info("response_json: {}".format(response_json))
        
        if "predictions" in response_json:

            predictions = response_json["predictions"]

            LOGGER.info("predictions: {}".format(predictions))

            predicted_classes = []

            for prediction in predictions:
                LOGGER.info("outputs in loop: {}".format(prediction))
                LOGGER.info("type(outputs) in loop: {}".format(type(prediction)))

                prediction_proba = [round(el, 3) for el in max(predictions)]

                prediction_dict = {
                    "prediction": str(np.argmax(prediction)),
                    "prediction_proba": str(prediction_proba)
                }

                jsonline = json.dumps(prediction_dict)
                LOGGER.info("jsonline: {}".format(jsonline))

                predicted_classes.append(jsonline)
                LOGGER.info("predicted_classes in the loop: {}".format(predicted_classes))

            predicted_classes_jsonlines = "\n".join(predicted_classes)
            LOGGER.info("predicted_classes_jsonlines: {}".format(predicted_classes_jsonlines))

            response_content_type = context.accept_header

            return predicted_classes_jsonlines, response_content_type
        else:
            LOGGER.info("{}".format(response_json))
            
            raise Exception("{}".format(response_json))
    except Exception as e:
        stacktrace = traceback.format_exc()
        LOGGER.error("{}".format(stacktrace))

        raise e
