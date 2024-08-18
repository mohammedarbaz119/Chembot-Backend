import logging
import os
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
from Crag import crag
from flask import Flask
from flask_cors import CORS
from flask import request,stream_with_context,jsonify
app = Flask(__name__)
cors = CORS(app)



@app.route("/")
def home():
    return "Hello World! testing the server to see if it works"


@app.route("/query", methods=["GET"])
def query_index():
    query_text = request.args.get("text", None)
    print("calculating")
    if query_text is None:
        return (
            "No text found, please include text queryp param parameter in the URL",
            400,
        )
    def generate():
        ans = crag.run(query_str=query_text)
        for token in ans.response_gen:
            yield f"{token}"
        return "deom"

    return generate(), {"Content-Type": "text/plain"}
if __name__ == "__main__":
    app.run(debug=True)

