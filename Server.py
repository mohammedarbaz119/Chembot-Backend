import logging
import os
import sys
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout), 
                              logging.FileHandler('server.log')])
from Crag import crag
from flask_cors import CORS
from flask import Flask,Response,request,stream_with_context,jsonify

app = Flask(__name__)
cors = CORS(app)


@app.route("/",methods=["GET"])
def HomeoRTestRoute():
    app.logger.info("server is alive")
    return "hello world"


@app.route("/query", methods=["GET"])
def query_index():
    query_text = request.args.get('text', None)
    if query_text is None or query_text=="":
        return (
            "No text found, please include text query param parameter in the URL",
            400,
        )
    def generate():
        try:
            ans = crag.run(query_str=query_text)
            for token in ans.response_gen:
                yield f"{token}"
            yield f"[END]\n\n"
        except Exception as e:
            app.logger.error(f"{e}")
            yield f"Error: {str(e)}\n\n"
    try:

       response = Response(stream_with_context(generate()), mimetype="text/event-stream")
       response.headers['Cache-Control'] = 'no-cache'
       return response
    except Exception as e:
        app.logger.error(f"{e} error happened")
        return (
            f"there is some server error {e}",500,
        )



