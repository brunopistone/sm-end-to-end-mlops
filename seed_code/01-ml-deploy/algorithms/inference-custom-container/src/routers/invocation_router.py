from controllers import invocation_controller
from flask import Blueprint, jsonify, request
from flask_api import status
import logging
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

invocation_router = Blueprint("invocation_router", __name__)

# Controllers
ic = invocation_controller.InvocationController()

def __getStatusCode(response):
    if response["status"] == 200:
        tmp_status = status.HTTP_200_OK
    elif response["status"] == 400:
        tmp_status = status.HTTP_400_BAD_REQUEST
    elif response["status"] == 401:
        tmp_status = status.HTTP_401_UNAUTHORIZED
    elif response["status"] == 422:
        tmp_status = status.HTTP_200_OK
    elif response["status"] == 500:
        tmp_status = status.HTTP_500_INTERNAL_SERVER_ERROR
    elif response["status"] == 501:
        tmp_status = status.HTTP_501_NOT_IMPLEMENTED
    else:
        tmp_status = status.HTTP_500_INTERNAL_SERVER_ERROR
    return tmp_status

@invocation_router.route("", methods=['POST'])
def index():
    try:
        request_body = request.get_data()

        logger.info("Input: {}".format(request_body))
        logger.info("type: {}".format(type(request_body)))

        request_body = request_body.decode("utf-8")

        results = []

        logger.info("Request: {}".format(request_body))

        X = ic.input_handler(request_body)

        prediction = ic.predict_fn(X)

        results.append(ic.output_handler(prediction))

        return ",".join(results)
    except Exception as e:
        stacktrace = traceback.format_exc()
        logger.error("{}".format(stacktrace))

        raise e