from flask import Flask
from routers import invocation_router, ping_router

# Build flask app
# This is in a separate file, so that it can be used with different application servers (Gunicorn, Waitress...)
def create_app():
    application = Flask(__name__)

    application.register_blueprint(invocation_router.invocation_router, url_prefix="/invocations")
    application.register_blueprint(ping_router.ping_router, url_prefix="/ping")

    return application

if __name__ == '__main__':
    application = create_app()
    application.run(threaded=True, port=8080, host='0.0.0.0')