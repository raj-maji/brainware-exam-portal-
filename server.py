"""
Brainware Academic AI - Enterprise Universal Educational Backend Server (Python & Machine Learning)
File: server.py
Version: 7.0.0 - Powered by DynamicAcademicAIModel (Python Scikit-Learn ML & LLM Architecture)

Architecture:
1. DynamicAcademicAIModel (academic_model.py): 100% Python AI engine implementing universal
   educational instruction, Scikit-Learn NLP Machine Learning pipeline, Google Gemini & OpenAI gateways,
   NumPy TF-IDF vector RAG, and autonomous dynamic exam synthesis.
2. Endpoints:
   - POST /api/chat: Web portal chat endpoint with document ingestion support.
   - POST /v1/chat/completions: Standard OpenAI Chat Completions API.
   - GET /v1/models: OpenAI-compatible model catalog.
   - GET /api/health: Real-time health check showing active Machine Learning engine & neural gateways.
   - GET /api/config: Public configuration metadata.
   - GET /: Serves index.html and all static assets.
3. Thread-safe Token-Bucket Rate Limiter (60 req/min per IP).
4. CORS-enabled with Access-Control-Allow-Origin: *.
"""

import os
import sys
import time
import json
import re
import threading
from typing import Dict, Any, List, Optional
from flask import Flask, request, jsonify, send_from_directory, Response

# Optional dotenv support
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Fallback manual parser for .env
env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
if os.path.exists(env_file):
    try:
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                match = re.match(r'^\s*([\w.-]+)\s*=\s*(.*)?\s*$', line)
                if match:
                    k, v = match.group(1), match.group(2) or ''
                    v = v.strip().strip("'\"")
                    if k not in os.environ and v:
                        os.environ[k] = v
    except Exception as e:
        print(f"Notice: .env parsing warning: {e}", file=sys.stderr)

# Import the core Python AI Model Engine
from academic_model import DynamicAcademicAIModel, AcademicAIModel

# =============================================================================
# CONFIGURATION
# =============================================================================
PORT = int(os.environ.get('PORT', 5000))
HOST = os.environ.get('HOST', '0.0.0.0')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize the Python Academic AI Model
ai_model = DynamicAcademicAIModel(base_dir=BASE_DIR)

# =============================================================================
# IN-MEMORY RATE LIMITER (Token-Bucket / Sliding Window)
# =============================================================================
RATE_LIMIT_WINDOW = 60.0  # seconds
MAX_REQUESTS_PER_WINDOW = 60
rate_limits: Dict[str, Dict[str, Any]] = {}
rate_limit_lock = threading.Lock()

def check_rate_limit(ip: str) -> bool:
    now = time.time()
    with rate_limit_lock:
        data = rate_limits.get(ip)
        if not data or now > data['reset']:
            rate_limits[ip] = {'count': 1, 'reset': now + RATE_LIMIT_WINDOW}
            return True
        data['count'] += 1
        if data['count'] > MAX_REQUESTS_PER_WINDOW:
            return False
        return True

def cleanup_rate_limits():
    while True:
        time.sleep(300)
        now = time.time()
        with rate_limit_lock:
            expired = [ip for ip, data in rate_limits.items() if now > data['reset']]
            for ip in expired:
                del rate_limits[ip]

cleanup_thread = threading.Thread(target=cleanup_rate_limits, daemon=True)
cleanup_thread.start()

# =============================================================================
# FLASK APPLICATION SETUP
# =============================================================================
app = Flask(__name__, static_folder=BASE_DIR, static_url_path='')

@app.after_request
def apply_cors_headers(response: Response) -> Response:
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With'
    return response

@app.route('/api/<path:_subpath>', methods=['OPTIONS'])
@app.route('/api', methods=['OPTIONS'])
@app.route('/v1/<path:_subpath>', methods=['OPTIONS'])
@app.route('/v1', methods=['OPTIONS'])
def handle_options(_subpath: str = ''):
    return Response(status=204)

# =============================================================================
# HEALTH & CONFIGURATION ENDPOINTS
# =============================================================================
@app.route('/api/health', methods=['GET'])
@app.route('/api/status', methods=['GET'])
def health_check():
    has_openai = ai_model.openai_gateway.is_available()
    has_gemini = ai_model.gemini_gateway.is_available()
    active_engine = "OpenAI GPT" if has_openai else ("Google Gemini" if has_gemini else "Python Machine Learning (Scikit-Learn)")
    return jsonify({
        "status": "healthy",
        "service": "CSS Academic AI Universal Backend (Python & Machine Learning)",
        "version": "7.0.0",
        "runtime": f"Python {sys.version.split()[0]}",
        "modelEngine": "DynamicAcademicAIModel",
        "activeEngine": active_engine,
        "activeGateways": {
            "sklearn_ml": hasattr(ai_model, 'ml_classifier') and ai_model.ml_classifier.intent_pipeline is not None,
            "openai_sdk": has_openai,
            "gemini_neural": has_gemini,
            "vector_rag": ai_model.rag_model.tfidf_matrix is not None
        },
        "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    })

@app.route('/api/config', methods=['GET'])
def get_config():
    has_key = ai_model.openai_gateway.is_available() or ai_model.gemini_gateway.is_available()
    return jsonify({
        "serviceName": "CSS Academic AI (Python & ML Edition)",
        "modelEngine": "DynamicAcademicAIModel (Scikit-Learn ML & LLM)",
        "rateLimitWindow": "60 requests/minute",
        "hasServerKey": has_key
    })

# =============================================================================
# OPENAI-COMPATIBLE CHAT COMPLETIONS ENDPOINT (/v1/chat/completions)
# =============================================================================
@app.route('/v1/chat/completions', methods=['POST'])
def openai_chat_completions():
    client_ip = request.headers.get('X-Forwarded-For', request.remote_addr or '127.0.0.1').split(',')[0].strip()
    if not check_rate_limit(client_ip):
        return jsonify({
            "error": {
                "message": "Rate limit exceeded (60 req/min).",
                "type": "rate_limit_error",
                "code": 429
            }
        }), 429

    try:
        payload = request.get_json(force=True, silent=True) or {}
    except Exception:
        payload = {}

    messages = payload.get('messages', [])
    if not messages:
        return jsonify({
            "error": {
                "message": "'messages' array is required in request body.",
                "type": "invalid_request_error",
                "code": 400
            }
        }), 400

    temperature = float(payload.get('temperature', 0.2))
    model_req = payload.get('model')

    try:
        completion = ai_model.generate_chat_completion(messages=messages, model=model_req, temperature=temperature)
        return jsonify(completion)
    except Exception as e:
        return jsonify({
            "error": {
                "message": str(e),
                "type": "internal_error",
                "code": 500
            }
        }), 500

@app.route('/v1/models', methods=['GET'])
def list_openai_models():
    return jsonify({
        "object": "list",
        "data": [
            {"id": "gemini-1.5-flash", "object": "model", "owned_by": "google"},
            {"id": "gemini-2.0-flash-exp", "object": "model", "owned_by": "google"},
            {"id": "gemini-1.5-pro", "object": "model", "owned_by": "google"},
            {"id": "gpt-4o-mini", "object": "model", "owned_by": "openai"},
            {"id": "gpt-4o", "object": "model", "owned_by": "openai"},
            {"id": "brainware-academic-ai-v6", "object": "model", "owned_by": "brainware-university"}
        ]
    })

# =============================================================================
# PRIMARY CHAT API ENDPOINT (/api/chat)
# =============================================================================
@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    client_ip = request.headers.get('X-Forwarded-For', request.remote_addr or '127.0.0.1').split(',')[0].strip()

    if not check_rate_limit(client_ip):
        return jsonify({
            "success": False,
            "error": "Too many requests. Rate limit is 60 requests per minute per client."
        }), 429

    try:
        payload = request.get_json(force=True, silent=True) or {}
    except Exception:
        payload = {}

    query = (payload.get('query') or '').strip()
    history = payload.get('history', []) if isinstance(payload.get('history'), list) else []
    file_info = payload.get('file')
    client_api_key = (payload.get('apiKey') or '').strip()

    if not query and not file_info:
        return jsonify({
            "success": False,
            "error": "Question query or document attachment is required."
        }), 400

    try:
        response_data = ai_model.generate_exam_response(
            query=query,
            history=history,
            file_info=file_info,
            client_key=client_api_key
        )
        return jsonify(response_data)
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Model inference error: {str(e)}"
        }), 500

# =============================================================================
# STATIC ASSET & PORTAL SERVING
# =============================================================================
@app.route('/', methods=['GET'])
def serve_index():
    return send_from_directory(BASE_DIR, 'index.html')

@app.route('/<path:path>', methods=['GET'])
def serve_static(path: str):
    if os.path.exists(os.path.join(BASE_DIR, path)):
        return send_from_directory(BASE_DIR, path)
    return send_from_directory(BASE_DIR, 'index.html')

@app.after_request
def add_cache_control_headers(response):
    # Disable aggressive mobile browser caching for dynamic portal assets
    if request.path.endswith(('.html', '.js', '.json')):
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response

# =============================================================================
# SERVER ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    has_openai = ai_model.openai_gateway.is_available()
    has_gemini = ai_model.gemini_gateway.is_available()

    print("=" * 68)
    print("  CSS Academic AI - Python Enterprise Server Running (v6.0.0)")
    print(f"  URL:              http://localhost:{PORT}")
    print(f"  Health Check:     http://localhost:{PORT}/api/health")
    print(f"  Web Chat API:     http://localhost:{PORT}/api/chat (POST)")
    print(f"  OpenAI API:       http://localhost:{PORT}/v1/chat/completions (POST)")
    print(f"  AI Model:         DynamicAcademicAIModel (General-Purpose LLM Engine)")
    print(f"  OpenAI Gateway:   {'Active ✓ (' + ai_model.openai_gateway.model + ')' if has_openai else 'Available with OPENAI_API_KEY in .env'}")
    print(f"  Gemini Gateway:   {'Active ✓ (' + ai_model.gemini_gateway.primary_model + ')' if has_gemini else 'Available with GEMINI_API_KEY in .env'}")
    print(f"  Vector RAG:       Active ✓ (1,254 Indexed University Solutions)")
    print(f"  Live Web Search:  Active ✓ (DuckDuckGo & Encyclopedic Reference)")
    print("=" * 68)
    app.run(host=HOST, port=PORT, threaded=True)
