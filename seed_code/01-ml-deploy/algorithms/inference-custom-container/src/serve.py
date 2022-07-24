#!/usr/bin/env python

import gunicorn_builder as builder

def main():
    gb = builder.create_app()

    app = gb.load()
    app.run(threaded=True, port=8080, host='0.0.0.0')

if __name__ == '__main__':
    main()
