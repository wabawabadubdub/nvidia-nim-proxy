from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import requests
import os
import json
import time

app = Flask(__name__)
CORS(app)

# NVIDIA NIM configuration
NVIDIA_API_KEY = os.environ.get('NVIDIA_API_KEY', '')
NVIDIA_BASE_URL = os.environ.get('NVIDIA_BASE_URL', 'https://integrate.api.nvidia.com/v1')

@app.after_request
def after_request(response):
    """Add CORS headers to all responses"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

@app.route('/v1', methods=['GET', 'OPTIONS'])
def v1_root():
    """V1 API root endpoint"""
    if request.method == 'OPTIONS':
        return '', 204
    return jsonify({
        'message': 'NVIDIA NIM OpenAI-compatible API',
        'endpoints': {
            '/v1/chat/completions': 'POST - Chat completions',
            '/v1/models': 'GET - List models'
        }
    })

@app.route('/v1/chat/completions', methods=['POST', 'OPTIONS'])
def chat_completions():
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        data = request.json
        
        # Extract parameters from OpenAI format
        messages = data.get('messages', [])
        model = data.get('model', 'meta/llama-3.1-70b-instruct')
        temperature = data.get('temperature', 0.7)
        max_tokens = data.get('max_tokens', 1024)
        stream = data.get('stream', False)
        
        # Prepare NVIDIA NIM request
        nvidia_payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        # Add optional parameters if present
        if 'top_p' in data:
            nvidia_payload['top_p'] = data['top_p']
        if 'frequency_penalty' in data:
            nvidia_payload['frequency_penalty'] = data['frequency_penalty']
        if 'presence_penalty' in data:
            nvidia_payload['presence_penalty'] = data['presence_penalty']
        
        headers = {
            'Authorization': f'Bearer {NVIDIA_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        # Handle streaming vs non-streaming
        if stream:
            return Response(
                stream_nvidia_response(nvidia_payload, headers),
                mimetype='text/event-stream'
            )
        else:
            response = requests.post(
                f'{NVIDIA_BASE_URL}/chat/completions',
                headers=headers,
                json=nvidia_payload,
                timeout=120
            )
            
            if response.status_code == 200:
                return jsonify(response.json())
            else:
                return jsonify({
                    'error': {
                        'message': f'NVIDIA NIM API error: {response.text}',
                        'type': 'nvidia_api_error',
                        'code': response.status_code
                    }
                }), response.status_code
                
    except Exception as e:
        return jsonify({
            'error': {
                'message': str(e),
                'type': 'proxy_error'
            }
        }), 500

def stream_nvidia_response(payload, headers):
    """Stream responses from NVIDIA NIM API"""
    try:
        response = requests.post(
            f'{NVIDIA_BASE_URL}/chat/completions',
            headers=headers,
            json=payload,
            stream=True,
            timeout=120
        )
        
        for line in response.iter_lines():
            if line:
                line_text = line.decode('utf-8')
                if line_text.startswith('data: '):
                    yield f"{line_text}\n\n"
                    
    except Exception as e:
        error_data = {
            'error': {
                'message': str(e),
                'type': 'stream_error'
            }
        }
        yield f"data: {json.dumps(error_data)}\n\n"

@app.route('/v1/models', methods=['GET', 'OPTIONS'])
def list_models():
    """List available models"""
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        headers = {
            'Authorization': f'Bearer {NVIDIA_API_KEY}'
        }
        
        response = requests.get(
            f'{NVIDIA_BASE_URL}/models',
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            # Return a default model list if API call fails
            return jsonify({
                'object': 'list',
                'data': [
                    {
                        'id': 'meta/llama-3.1-70b-instruct',
                        'object': 'model',
                        'created': int(time.time()),
                        'owned_by': 'nvidia'
                    },
                    {
                        'id': 'deepseek-ai/deepseek-r1',
                        'object': 'model',
                        'created': int(time.time()),
                        'owned_by': 'nvidia'
                    }
                ]
            })
    except Exception as e:
        return jsonify({
            'error': {
                'message': str(e),
                'type': 'models_error'
            }
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'nvidia-nim-proxy'})

@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    return jsonify({
        'message': 'NVIDIA NIM to OpenAI API Proxy',
        'endpoints': {
            '/v1/chat/completions': 'POST - Chat completions',
            '/v1/models': 'GET - List models',
            '/health': 'GET - Health check'
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
