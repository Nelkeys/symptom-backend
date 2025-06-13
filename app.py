from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

# Load the trained model
model = joblib.load("symptom_model.joblib")
class_names = model.classes_

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    user_input = data.get("text", "")

    # Get prediction and probabilities
    prediction = model.predict([user_input])[0]
    probabilities = model.predict_proba([user_input])[0]
    confidence = max(probabilities) * 100  # Convert to percentage

    return jsonify({
        "disease": prediction,
        "confidence": f"{confidence:.2f}%",  # Rounded to 2 decimal places
        "advice": "Seek medical advice if symptoms persist."
    })

if __name__ == "__main__":
    app.run(debug=True)

