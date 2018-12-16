import logging

import flask

app = flask.Flask(__name__, static_folder="static")


@app.route('/<path:path>')
def static_proxy(path):
    return app.send_static_file(path)


logging.info('\nGo to http://localhost:8000/peepo.html to see the example\n')
app.run(port=8000)