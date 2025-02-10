import os
import openai
from flask import Flask, request, render_template, jsonify
from utils.process_docs import process_document, load_vector_store
from azure_config import get_openai_response

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load preprocessed document vectors
vector_store = load_vector_store()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["document"]
        if file:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            process_document(file_path)  # Process and store vectors
            return "File uploaded and processed successfully!"
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_query = request.json.get("query", "")
    if not user_query:
        return jsonify({"response": "Please enter a query."})

    response = get_openai_response(user_query)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
