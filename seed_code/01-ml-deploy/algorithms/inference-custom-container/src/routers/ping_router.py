from controllers import ping_controller
from flask import Blueprint, jsonify
from flask_api import status
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ping_router = Blueprint("ping_router", __name__)

# Controllers
pg = ping_controller.PingController()

def __getStatusCode(response):
    if response["status"] == 200:
        tmp_status = status.HTTP_200_OK
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

@ping_router.route("", methods=['GET'])
def index():
    final_response = jsonify(pg.handle())

    tmp_status = __getStatusCode({"status": 200})

    return final_response, tmp_status
