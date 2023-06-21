from datetime import datetime
from flask import Flask, request, json, jsonify
from flask_cors import CORS
import main as ia
import logging

app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def health():
    now = datetime.now()
    msg = str(now)
    return json.dumps(msg)


@app.route("/validate", methods=["POST"])
def flask():
    content = json.loads(request.data)
    v = float(content["variance"])
    c = float(content["curtosis"])
    s = float(content["skewness"])
    e = float(content["entropy"])
    predict = ia.user_entries(v, c, s, e)
    return jsonify(classification=str(predict))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

logging.basicConfig()
