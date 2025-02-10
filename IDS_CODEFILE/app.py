from flask import Flask, render_template, jsonify
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load the trained model
model = joblib.load("intrusion_detection_model.joblib")

# Sample test data (replace with real test data)
X_test = np.random.rand(20, 32)  # Assuming 32 encoded features
y_test = np.random.randint(0, 2, 20)

# Ensure stratified split to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(
    X_test, y_test, test_size=0.2, random_state=42, stratify=y_test
)

# Get predictions
y_pred = model.predict(X_test)

# Handle undefined metric warning by using zero_division parameter
classification_rep = classification_report(y_test, y_pred, output_dict=True, zero_division=1)
confusion_mat = confusion_matrix(y_test, y_pred).tolist()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/results")
def results():
    return jsonify({
        "classification_report": classification_rep,
        "confusion_matrix": confusion_mat
    })

if __name__ == "__main__":
    app.run(debug=True)
