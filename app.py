"""
Brainware Academic AI - Application Entrypoint
Provides standard 'app' instance for WSGI/ASGI servers (Gunicorn, Waitress, Uvicorn)
or direct execution: python app.py
"""

from server import app, PORT, HOST, PRIMARY_MODEL, has_key

if __name__ == '__main__':
    print("=" * 64)
    print("  Brainware Academic AI - Python Flask Application Running")
    print(f"  URL:            http://localhost:{PORT}")
    print(f"  Health Check:   http://localhost:{PORT}/api/health")
    print(f"  API Endpoint:   http://localhost:{PORT}/api/chat (POST)")
    print(f"  Primary Model:  {PRIMARY_MODEL}")
    print(f"  Gemini Key:     {'Configured ✓' if has_key else 'Missing (.env)'}")
    print("=" * 64)
    app.run(host=HOST, port=PORT, threaded=True)
