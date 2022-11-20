from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from bot import response_from_user

app = Flask(__name__)
CORS(app)

# simple API which allows user input to be carried on to NN
@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    response = response_from_user(text)
    msg = {"answer": response}
    return jsonify(msg)

if __name__ == "__main__":
    app.run(debug=True)