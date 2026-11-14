"""
Brainware Academic AI - Enterprise Universal Educational Backend Server (Python Implementation)
Version: 4.0.0 - Powered by Python AcademicAIModel & Flask

Architecture:
1. AcademicAIModel (academic_model.py): Pure Python AI model implementing intent analysis,
   NumPy vector retrieval, exam formatting, and foundation model inference.
2. Flask Server (server.py): Multi-threaded HTTP server providing /api/chat, /api/health,
   and static asset delivery.
3. Thread-safe Token-Bucket Rate Limiter (60 req/min per IP).
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

# Fallback parser for .env
env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
if os.path.exists(env_file):
    try:
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                match = re.match(r'^\s*([\w.-]+)\s*=\s*(.*)?\s*$', line)
                if match:
                    k, v = match.group(1), match.group(2) or ''
                    v = v.strip().strip("'\"")
                    if k not in os.environ:
                        os.environ[k] = v
    except Exception as e:
        print(f"Notice: .env parsing warning: {e}", file=sys.stderr)

# Import the core Python AI Model Engine
from academic_model import AcademicAIModel, AcademicQuestionAnalyzer, ExamQuestionType

# =============================================================================
# CONFIGURATION
# =============================================================================
PORT = int(os.environ.get('PORT', 5000))
HOST = os.environ.get('HOST', '0.0.0.0')
PRIMARY_MODEL = os.environ.get('GEMINI_MODEL', 'gemini-1.5-flash')
FALLBACK_MODELS = ['gemini-2.0-flash-exp', 'gemini-1.5-pro', 'gemini-1.5-flash']
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '').strip()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize the Python Academic AI Model
ai_model = AcademicAIModel(api_key=GEMINI_API_KEY, base_dir=BASE_DIR)

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
def handle_options(_subpath: str = ''):
    return Response(status=204)

# =============================================================================
# HEALTH & CONFIGURATION ENDPOINTS
# =============================================================================
@app.route('/api/health', methods=['GET'])
@app.route('/api/status', methods=['GET'])
def health_check():
    has_key = bool(os.environ.get('GEMINI_API_KEY', '').strip() or GEMINI_API_KEY)
    return jsonify({
        "status": "healthy",
        "service": "Brainware Academic AI Worldwide Backend (Python)",
        "version": "4.0.0",
        "runtime": f"Python {sys.version.split()[0]}",
        "modelEngine": "AcademicAIModel (Python Core)",
        "activeModel": PRIMARY_MODEL,
        "availableFallbacks": FALLBACK_MODELS,
        "geminiConfigured": has_key,
        "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    })

@app.route('/api/config', methods=['GET'])
def get_config():
    has_key = bool(os.environ.get('GEMINI_API_KEY', '').strip() or GEMINI_API_KEY)
    return jsonify({
        "serviceName": "Brainware Academic AI (Python Edition)",
        "modelEngine": "AcademicAIModel (Python Core)",
        "primaryModel": PRIMARY_MODEL,
        "rateLimitWindow": "60 requests/minute",
        "hasServerKey": has_key
    })

# =============================================================================
# PRIMARY CHAT API ENDPOINT
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

    # Delegate inference to the Python Academic AI Model
    response_data = ai_model.generate_exam_response(
        query=query,
        history=history,
        file_info=file_info,
        client_key=client_api_key
    )
    return jsonify(response_data)

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

# =============================================================================
# SERVER ENTRY POINT
# =============================================================================
has_key = bool(os.environ.get('GEMINI_API_KEY', '').strip() or GEMINI_API_KEY)

if __name__ == '__main__':
    print("=" * 64)
    print("  Brainware Academic AI - Python Enterprise Server Running")
    print(f"  URL:            http://localhost:{PORT}")
    print(f"  Health Check:   http://localhost:{PORT}/api/health")
    print(f"  API Endpoint:   http://localhost:{PORT}/api/chat (POST)")
    print(f"  AI Model:       AcademicAIModel (Python Core Engine)")
    print(f"  Primary LLM:    {PRIMARY_MODEL}")
    print(f"  Gemini Key:     {'Configured ✓' if has_key else 'Missing (.env)'}")
    print("=" * 64)
    app.run(host=HOST, port=PORT, threaded=True)
