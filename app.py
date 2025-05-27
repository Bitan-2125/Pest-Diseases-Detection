from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from dotenv import load_dotenv
import PyPDF2
import io
import json

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize the model and tokenizer
MODEL_NAME = "microsoft/DialoGPT-medium"  # You can also use "microsoft/DialoGPT-large" for better responses
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Financial context to help the model generate more relevant responses
FINANCIAL_CONTEXT = """
You are a financial advisor AI assistant. You provide advice on:
- Investment strategies and portfolio management
- Retirement planning
- Budgeting and financial planning
- Tax optimization
- Debt management
- Insurance planning
- Emergency fund management
- Credit score improvement

Always provide practical, actionable advice while considering the user's specific situation.
"""

def generate_response(user_input, chat_history=None):
    if chat_history is None:
        chat_history = []
    
    # Combine financial context with user input
    full_input = f"{FINANCIAL_CONTEXT}\nUser: {user_input}\nAssistant:"
    
    # Encode the input
    input_ids = tokenizer.encode(full_input + tokenizer.eos_token, return_tensors='pt')
    
    # Generate response
    chat_response_ids = model.generate(
        input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=100,
        top_p=0.7,
        temperature=0.8
    )
    
    # Decode the response
    response = tokenizer.decode(chat_response_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    
    return response

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    chat_history = data.get('history', [])
    
    try:
        # Generate response using the model
        response = generate_response(user_message, chat_history)
        
        return jsonify({
            'response': response,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/analyze-income', methods=['POST'])
def analyze_income():
    data = request.json
    income = data.get('income', 0)
    
    try:
        # Generate personalized investment advice based on income
        prompt = f"Given an annual income of ${income}, provide specific investment recommendations and financial planning advice."
        response = generate_response(prompt)
        
        return jsonify({
            'recommendation': response,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/analyze-payslip', methods=['POST'])
def analyze_payslip():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided', 'status': 'error'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected', 'status': 'error'}), 400
    
    try:
        # Read PDF content
        pdf_content = file.read()
        pdf_file = io.BytesIO(pdf_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        # Extract text from PDF
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # Generate analysis using the model
        prompt = f"Analyze this payslip information and provide financial advice: {text}"
        response = generate_response(prompt)
        
        return jsonify({
            'message': 'Payslip analyzed successfully.',
            'analysis': response,
            'status': 'success'
        })
            
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

if __name__ == '__main__':
    app.run(debug=True) 
