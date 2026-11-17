# Brainware Academic AI - Cloud Hosting & Deployment Guide

This application is powered by a **100% Python backend** (`server.py`) using Flask, Scikit-Learn NLP pipelines, NumPy Vector RAG, and multi-model neural inference.

Because the AI and ML models run natively in Python, the application requires a Python-capable web host (not static-only hosts like GitHub Pages).

Below are the **top recommended ways to host this site**, ranging from free 1-click cloud platforms to instant public URLs.

---

## Option 1: Render.com (Recommended Free Cloud Hosting)

Render provides free hosting for Python web services with automatic HTTPS, continuous deployment from GitHub, and zero configuration.

### Steps:
1. Go to **[https://render.com](https://render.com)** and sign in (using GitHub).
2. Click **New +** -> **Web Service**.
3. Connect your repository: `https://github.com/raj-maji/brainware-exam-portal-`.
4. Render will auto-detect the configuration, or enter:
   - **Name:** `brainware-academic-ai`
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn server:app --workers 2 --threads 4 --timeout 120`
   - **Plan:** Free
5. (Optional) Under **Environment Variables**, add:
   - `OPENAI_API_KEY`: *(your OpenAI key if using GPT-4o)*
   - `GEMINI_API_KEY`: *(your Google Gemini key if using Gemini 2.0 Flash)*
6. Click **Deploy Web Service**.
7. In ~2 minutes, your website is live worldwide at `https://brainware-academic-ai.onrender.com`!

---

## Option 2: Railway.app (Fastest 1-Click Deployment)

Railway provides instant containerized deployments with automatic Nixpacks builds.

### Steps:
1. Go to **[https://railway.app](https://railway.app)** and log in with GitHub.
2. Click **New Project** -> **Deploy from GitHub repo**.
3. Select `raj-maji/brainware-exam-portal-`.
4. Railway reads the included `railway.json` and `requirements.txt` automatically.
5. In **Settings** -> **Networking**, click **Generate Domain**.
6. Your portal is live instantly at `https://brainware-exam-portal-production.up.railway.app`!

---

## Option 3: Instant Public URL with Cloudflare Tunnel (Free, No Sign-up Required)

If you want to immediately share or access your currently running local server (`http://127.0.0.1:5000`) on your phone or with classmates right now without deploying to the cloud:

### Run in PowerShell:
```powershell
# Download Cloudflared (if not already installed)
winget install Cloudflare.cloudflared

# Expose your running local server to a secure public https:// URL:
cloudflared tunnel --url http://127.0.0.1:5000
```
This generates a secure, free public URL (e.g. `https://random-words.trycloudflare.com`) that routes directly to your Python AI engine.

---

## Option 4: Docker / Cloud Run / Hugging Face Spaces

The included `Dockerfile` allows deployment to any container platform:
- **Hugging Face Spaces:** Create a new Space -> SDK: Docker -> Push repository.
- **Google Cloud Run:** `gcloud run deploy --source .`
- **AWS App Runner / DigitalOcean App Platform:** Select Dockerfile or GitHub repo.

---

## Verification
Once deployed, verify the endpoints:
- **Web Interface:** `https://<your-domain>/`
- **Health Check:** `https://<your-domain>/api/health`
- **AI Chat API:** `POST https://<your-domain>/api/chat`
- **Question Set PDF:** `https://<your-domain>/Question%20Set/MCA37510A_Android_Application_Development_Complete_Solutions.pdf`
