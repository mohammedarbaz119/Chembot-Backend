import logging
import os
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
from Crag import crag
from flask_cors import CORS
from flask import Flask,Response,request,stream_with_context,jsonify

app = Flask(__name__)
cors = CORS(app)



@app.route("/query", methods=["GET"])
def query_index():
    query_text = request.args.get('text', None)
    if query_text is None:
        return (
            "No text found, please include text query param parameter in the URL",
            400,
        )

    def generate():
        try:
            ans = crag.run(query_str=query_text)
            for token in ans.response_gen:
                yield f"{token}\n\n"
            yield f"[END]\n\n"
        except Exception as e:
            yield f"Error: {str(e)}\n\n"

    response = Response(stream_with_context(generate()), mimetype="text/event-stream")
    response.headers['Cache-Control'] = 'no-cache'
    return response



