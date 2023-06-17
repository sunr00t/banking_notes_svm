from datetime import datetime
from flask import Flask, request, json
from flask_cors import CORS
import main as ia
import logging

app = Flask(__name__)
CORS(app)

@app.route("/", methods=['GET'])
def health():
    now = datetime.now()
    msg = "HealthCheck: " + str(now)
    return msg
  
@app.route("/validate", methods=['POST'])
def flask():
    content = json.loads(request.data)
    v = json.dumps(content['variance'])
    c = json.dumps(content['curtosis'])
    s = json.dumps(content['skewness'])
    e = json.dumps(content['entropy'])
    predict = ia.user_entries(v, c, s, e)
    print("HOST:",request.remote_addr, "| Predict: " + str(predict))
    return str(predict)

if __name__ == "__main__":
  app.run(debug=True, host='0.0.0.0', port=5000)
  
logging.basicConfig()
logging.getLogger('flask_cors').level = logging.INFO
  