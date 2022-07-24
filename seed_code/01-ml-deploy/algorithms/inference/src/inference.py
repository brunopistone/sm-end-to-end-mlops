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

        LOGGER.info("Request: {}".format(data))

        transformed_instances = []

        for el in data:
            body = el.decode("utf-8")
            LOGGER.info("Input data: {}".format(body))

            LOGGER.info("Input: {}".format(body))
            LOGGER.info("type: {}".format(type(body)))

            start_lan = __detect_language(body)

            if start_lan != "en":
                body = translate_service.translate_string(body, start_lan, "en")

                LOGGER.info("Translated sentence: {}".format(body))
            else:
                LOGGER.info("Detected en language")

            body = utils.clean_text(body)

            LOGGER.info("review_body: {}".format(body))

            input_ids, input_masks, input_segments = __encode(tokenizer, MODEL_SHAPE, body)

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

                prediction_proba = [round(el, 3) for el in prediction]

                predicted_classes.append(str(np.argmax(prediction)))
                predicted_classes.append(str(max(prediction_proba)))

            LOGGER.info("Predictions: {}".format(predicted_classes))

            response_content_type = context.accept_header

            LOGGER.info("Response type: {}".format(response_content_type))
            LOGGER.info("Return data: {}".format(",".join(predicted_classes)))

            return ",".join(predicted_classes), response_content_type
        else:
            LOGGER.info("{}".format(response_json))
            
            raise Exception("{}".format(response_json))
    except Exception as e:
        stacktrace = traceback.format_exc()
        LOGGER.error("{}".format(stacktrace))

        raise e
