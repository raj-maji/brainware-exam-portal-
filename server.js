/**
 * Brainware Academic AI - Secure Backend Server
 * 
 * Capabilities:
 * 1. Secure API Proxy for Google Gemini 1.5 Flash
 * 2. Multi-Turn Conversation History Support
 * 3. In-Memory Rate Limiting (60 requests/min per IP)
 * 4. Input Validation & Structured Error Responses
 * 5. Document & Multimodal Vision Support (Base64 Images, PDF context)
 * 6. Zero-Dependency Standalone Mode (runs natively with pure Node.js)
 * 7. Built-in Static File Server for Local Portal Hosting
 */

const http = require('http');
const fs = require('fs');
const path = require('path');
const url = require('url');

// Load environment variables from .env if present
const envPath = path.join(__dirname, '.env');
if (fs.existsSync(envPath)) {
  try {
    const envContent = fs.readFileSync(envPath, 'utf8');
    envContent.split('\n').forEach(line => {
      const match = line.match(/^\s*([\w.-]+)\s*=\s*(.*)?\s*$/);
      if (match) {
        const key = match[1];
        let val = match[2] || '';
        val = val.trim().replace(/^['"]|['"]$/g, '');
        if (!process.env[key]) {
          process.env[key] = val;
        }
      }
    });
  } catch (e) {
    console.warn('Notice: Could not parse .env file:', e.message);
  }
}

const PORT = parseInt(process.env.PORT || '5000', 10);
const GEMINI_API_KEY = process.env.GEMINI_API_KEY || '';

// Rate limiting store: IP -> { count, resetTime }
const rateLimitMap = new Map();
const RATE_LIMIT_WINDOW_MS = 60 * 1000; // 1 minute
const MAX_REQUESTS_PER_WINDOW = 60; // 60 req/min

function checkRateLimit(ip) {
  const now = Date.now();
  let client = rateLimitMap.get(ip);
  if (!client || now > client.resetTime) {
    client = { count: 1, resetTime: now + RATE_LIMIT_WINDOW_MS };
    rateLimitMap.set(ip, client);
    return true;
  }
  client.count++;
  if (client.count > MAX_REQUESTS_PER_WINDOW) {
    return false;
  }
  return true;
}

// Clean up stale rate limits every 5 minutes
setInterval(() => {
  const now = Date.now();
  for (const [ip, data] of rateLimitMap.entries()) {
    if (now > data.resetTime) rateLimitMap.delete(ip);
  }
}, 5 * 60 * 1000);

// Academic AI System Prompt
const SYSTEM_INSTRUCTION = `You are Brainware Academic AI, an expert University Professor, Academic Tutor, Computer Science Specialist, Mathematics Instructor, and Research Mentor.

You are capable of answering educational questions across all disciplines, including:
- Computer Science, IT, Programming (Java, Python, C, C++, JavaScript, Web Development)
- Data Structures, Algorithms, DBMS, Operating Systems, Networking
- Artificial Intelligence, Machine Learning, Data Science
- Mathematics, Calculus, Algebra, Statistics, Physics, Chemistry, Biology
- Medical, Health Sciences, and Engineering
- Business, Management, Economics, Finance
- History, Geography, Political Science, Literature, Languages, Philosophy
- Research Methodology and General Academic Studies

Follow these strict educational standards:
1. Answer only what is asked, directly, accurately, and with academic depth.
2. Break down complex concepts into clear, logical, step-by-step explanations.
3. For mathematics, provide step-by-step derivations using LaTeX ($...$ for inline and $$...$$ for block formulas).
4. For programming, write robust, verified code with comments, time/space complexity, and expected outputs.
5. Use clean Markdown tables when comparing or contrasting concepts.
6. When reference material or uploaded documents are provided, prioritize the document's actual facts and cite them accurately.
7. Understand follow-up questions contextually based on previous conversation history.
8. If uncertain, acknowledge limitations honestly without inventing facts or citations.`;

// MIME type map for static file serving
const MIME_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.webp': 'image/webp',
  '.svg': 'image/svg+xml',
  '.pdf': 'application/pdf',
  '.ico': 'image/x-icon'
};

const server = http.createServer(async (req, res) => {
  // CORS Headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With');

  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }

  const parsedUrl = url.parse(req.url, true);
  const pathname = parsedUrl.pathname;
  const clientIp = req.headers['x-forwarded-for'] || req.socket.remoteAddress || 'unknown';

  // --- HEALTH CHECK ROUTE ---
  if (req.method === 'GET' && pathname === '/api/health') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      status: 'healthy',
      service: 'Brainware Academic AI Backend',
      version: '2.0.0',
      geminiConfigured: !!(process.env.GEMINI_API_KEY || GEMINI_API_KEY),
      timestamp: new Date().toISOString()
    }));
    return;
  }

  // --- SECURE CHAT API ENDPOINT ---
  if (req.method === 'POST' && pathname === '/api/chat') {
    // 1. Rate Limiting Check
    if (!checkRateLimit(clientIp)) {
      res.writeHead(429, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        success: false,
        error: 'Too many requests. Rate limit is 60 requests per minute per client.'
      }));
      return;
    }

    // 2. Read Request Body
    let body = '';
    req.on('data', chunk => {
      body += chunk;
      if (body.length > 20 * 1024 * 1024) { // 20MB limit (for base64 images)
        res.writeHead(413, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: false, error: 'Payload too large (max 20MB)' }));
        req.destroy();
      }
    });

    req.on('end', async () => {
      try {
        const payload = JSON.parse(body || '{}');
        const query = (payload.query || '').trim();
        const history = Array.isArray(payload.history) ? payload.history : [];
        const file = payload.file || null;
        const clientApiKey = payload.apiKey || '';

        if (!query && !file) {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ success: false, error: 'A query or attached file is required.' }));
          return;
        }

        // Active key priority: Server Environment Variable -> Client-provided Key
        const activeKey = process.env.GEMINI_API_KEY || GEMINI_API_KEY || clientApiKey;

        if (!activeKey) {
          res.writeHead(401, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({
            success: false,
            error: 'No Gemini API Key configured. Please set GEMINI_API_KEY in the server .env file or provide an API key.',
            requiresApiKey: true
          }));
          return;
        }

        // Format contents array for Gemini API
        const contents = [];

        // Add history (sliding window of last 12 messages)
        const recentHistory = history.slice(-12);
        for (const item of recentHistory) {
          if (item && item.text) {
            contents.push({
              role: item.role === 'model' || item.role === 'assistant' ? 'model' : 'user',
              parts: [{ text: item.text }]
            });
          }
        }

        // Prepare current user turn
        const currentParts = [];

        // Multimodal image support
        if (file && file.base64 && file.base64.startsWith('data:image/')) {
          const base64Data = file.base64.split(',')[1];
          currentParts.push({
            inlineData: {
              mimeType: file.mimeType || 'image/jpeg',
              data: base64Data
            }
          });
        }

        // Document text context
        let promptText = query;
        if (file && file.text) {
          promptText = `[ATTACHED DOCUMENT: ${file.name} (${file.type || 'FILE'})]\n${file.text.substring(0, 15000)}\n\n[STUDENT QUESTION]:\n${query || 'Please explain and analyze this document thoroughly.'}`;
        }

        currentParts.push({ text: promptText });
        contents.push({
          role: 'user',
          parts: currentParts
        });

        // Gemini 1.5 Flash endpoint
        const geminiEndpoint = `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${encodeURIComponent(activeKey)}`;

        const geminiRequestBody = {
          contents: contents,
          systemInstruction: {
            parts: [{ text: SYSTEM_INSTRUCTION }]
          },
          generationConfig: {
            temperature: 0.2,
            topP: 0.95,
            maxOutputTokens: 3500
          }
        };

        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 25000); // 25s timeout

        const apiResponse = await fetch(geminiEndpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(geminiRequestBody),
          signal: controller.signal
        });
        clearTimeout(timeout);

        if (!apiResponse.ok) {
          const errData = await apiResponse.json().catch(() => ({}));
          console.error('Gemini API Error from Google:', apiResponse.status, errData);
          res.writeHead(apiResponse.status, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({
            success: false,
            status: apiResponse.status,
            error: errData.error?.message || `Google Gemini API returned status ${apiResponse.status}`
          }));
          return;
        }

        const data = await apiResponse.json();
        const candidate = data.candidates && data.candidates[0];

        if (candidate && candidate.finishReason === 'SAFETY') {
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({
            success: true,
            text: '*The safety filters flagged this request. Please rephrase using standard educational and academic terminology.*',
            model: 'gemini-1.5-flash',
            source: 'gemini'
          }));
          return;
        }

        const responseText = candidate?.content?.parts?.[0]?.text || '';

        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
          success: true,
          text: responseText,
          model: 'gemini-1.5-flash',
          source: 'gemini'
        }));

      } catch (err) {
        console.error('Server Request Handling Error:', err);
        const isTimeout = err.name === 'AbortError';
        res.writeHead(isTimeout ? 504 : 500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
          success: false,
          error: isTimeout ? 'Request timed out waiting for AI response.' : err.message || 'Internal server error'
        }));
      }
    });

    return;
  }

  // --- STATIC FILE SERVING (Serves Portal Frontend Locally) ---
  if (req.method === 'GET') {
    let filePath = path.join(__dirname, pathname === '/' ? 'index.html' : pathname);

    // Prevent directory traversal attacks
    if (!filePath.startsWith(__dirname)) {
      res.writeHead(403);
      res.end('Access Denied');
      return;
    }

    fs.stat(filePath, (err, stats) => {
      if (err || !stats.isFile()) {
        res.writeHead(404, { 'Content-Type': 'text/plain' });
        res.end('404 Not Found');
        return;
      }

      const ext = path.extname(filePath).toLowerCase();
      const contentType = MIME_TYPES[ext] || 'application/octet-stream';

      res.writeHead(200, { 'Content-Type': contentType });
      fs.createReadStream(filePath).pipe(res);
    });
    return;
  }

  res.writeHead(405, { 'Content-Type': 'text/plain' });
  res.end('Method Not Allowed');
});

server.listen(PORT, () => {
  console.log(`====================================================`);
  console.log(`  Brainware Academic AI - Server Running`);
  console.log(`  URL: http://localhost:${PORT}`);
  console.log(`  Health Check: http://localhost:${PORT}/api/health`);
  console.log(`  API Endpoint: http://localhost:${PORT}/api/chat (POST)`);
  console.log(`  Gemini Key Configured: ${!!(process.env.GEMINI_API_KEY || GEMINI_API_KEY)}`);
  console.log(`====================================================`);
});

// Graceful termination
process.on('SIGINT', () => {
  console.log('\nShutting down Brainware Academic AI Server...');
  server.close(() => process.exit(0));
});
