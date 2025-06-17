from flask import Flask, request, jsonify
import joblib
import pandas as pd
import joblib

app = Flask(__name__)

try:
  model = joblib.load("file_path")
except FileNotFoundError as e:
  return jsonify({"error":f"file path not found :{e}"}),500


required_fields = [
    ""
]

@app.route("/", methods=["GET"])
def index():
    return "Application started"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        
        data = request.get_json(force=True)
    
        missing = [f for f in required_fields if f not in data]

        if missing:
            return jsonify({"error":f"Missing fields:{missing}"}),400

    
        input_data = [data[field] for field in required_fields]
      
        input_df = pd.DataFrame([input_data], columns=required_fields)

        # Predict
        prediction = model.predict(input_df)

        return jsonify(
            {
                "predicted_price":round(float(prediction[0]),2)
            }
        )
      
    except Exception as e:
        return jsonify({
            "error":str(e)
        }),500

if __name__ == "__main__":
    app.run(debug=True)
