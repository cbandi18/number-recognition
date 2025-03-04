from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
with open("model/model.pkl", "rb") as f:
    W1, B1, W2, B2 = pickle.load(f)

def ReLU(X):
    return np.maximum(X, 0)

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

def forward_propagation(W1, B1, W2, B2, X):
    Z1 = W1.dot(X) + B1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + B2
    A2 = softmax(Z2)
    return np.argmax(A2, 0)

@app.route("/", methods=["GET"])
def index():
    return render_template("app/templates/index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json.get("pixels", [])
    if not data or len(data) != 784:
        return jsonify({"error": "Invalid input. Expected 784 pixels"}), 400
    
    X = np.array(data).reshape(784, 1) / 255.0  # Normalize input
    prediction = forward_propagation(W1, B1, W2, B2, X)
    
    return jsonify({"prediction": int(prediction)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)