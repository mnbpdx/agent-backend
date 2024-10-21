from flask import Flask, jsonify, request
from weather_agent import run_weather_agent
from flask_cors import CORS
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import OpenAI
from langchain.docstore.document import Document

app = Flask(__name__)

# Configure CORS
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000"], "supports_credentials": True}})

@app.route('/summarize', methods=['POST'])
def summarize():

    data = request.json
    original_text = data.get('text', '')
    if not original_text:
        return jsonify({"error": "No text provided"}), 400

    llm = OpenAI(temperature=0)
    percentage = 80  # You can adjust this value as needed
    prompt = f"""
    Summarize the following text to {percentage}% of its original length, ensuring the most important information is preserved.

    The goal is to condense the text while maintaining its core meaning and key points.

    ORIGINAL TEXT: {original_text}
    """
    original_length = len(original_text.split())
    target_length = int(original_length * (percentage / 100))
    summary = llm.invoke(prompt, max_tokens=target_length)

    return jsonify({
        "original_text": original_text,
        "summary": summary
    })

@app.route('/expand', methods=['GET'])
def expand():
    # Dummy data for expand endpoint
    expansion = {
        "original_text": "Short text.",
        "expanded_text": "This is a longer, more detailed version of the original short text."
    }
    return jsonify(expansion)

@app.route('/weather', methods=['POST'])
def weather():
    data = request.json
    query = data.get('query', '')
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    result = run_weather_agent(query)
    return jsonify({"response": result})

if __name__ == '__main__':
    app.run(debug=True)
