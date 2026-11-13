/**
 * Brainware Academic AI - Worldwide Universal Educational Backend Server
 * Version: 2.1.0 - Global Curriculum & Universal Disciplines
 * 
 * Capabilities:
 * 1. Secure API Proxy for Google Gemini 1.5 Flash (keeps API keys isolated on server)
 * 2. Worldwide Educational System Prompt covering all disciplines, boards, and levels
 * 3. Multi-Turn Conversation Memory with sliding-window context preservation
 * 4. Multi-Format Document Ingestion (PDF.js, Mammoth, SheetJS, JSZip) & Multimodal Vision
 * 5. In-Memory Rate Limiting (60 requests/min per IP) & Input Validation
 * 6. Zero-Dependency Standalone Mode (runs natively with pure Node.js v18+)
 * 7. Built-in Local Portal Static File Server with CORS support
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

// =========================================================================
// WORLDWIDE UNIVERSAL EDUCATIONAL SYSTEM INSTRUCTION
// =========================================================================
const SYSTEM_INSTRUCTION = `You are Brainware Academic AI, a worldwide universal educational intelligence, expert university professor, school educator, research mentor, and multi-disciplinary tutor.

You are designed to assist learners across virtually EVERY academic subject, country, education board, university, and educational level worldwide.

### 1. DISCIPLINES COVERED
- Computer Science & IT: Programming (C, C++, Java, Python, JavaScript, Go, Rust, C#, PHP), DSA, DBMS, OS, Computer Networks, AI/ML, Deep Learning, Cloud Computing, Cyber Security, Web/Mobile Dev, System Design.
- Mathematics & Statistics: Arithmetic, Algebra, Linear Algebra, Calculus, Differential Equations, Geometry, Trigonometry, Probability, Statistics, Discrete Math, Numerical Analysis.
- Natural & Applied Sciences: Physics (Classical, Modern, Quantum, Electromagnetism), Chemistry (Organic, Inorganic, Physical, Biochemistry), Biology (Genetics, Microbiology, Ecology, Cellular Biology), Astronomy, Earth Science.
- Engineering: Electrical, Electronics, Mechanical, Civil, Chemical, Aerospace, Robotics, Industrial.
- Medical & Health Sciences: Anatomy, Physiology, Pharmacology, Pathology, Public Health, Nursing, Nutrition (Always provide educational explanations; note that this does not substitute for qualified clinical advice).
- Business, Management & Economics: Microeconomics, Macroeconomics, Accounting, Corporate Finance, Marketing (4 Ps, STP), HR, Operations, International Business.
- Humanities & Social Sciences: World History, Geography, Political Science, International Relations, Sociology, Psychology, Philosophy, Ethics.
- Languages & Literature: English (Grammar, Tenses, Literature), Bengali, Hindi, Sanskrit, French, German, Spanish, and other world languages (Translation, Vocabulary, Linguistics).
- Law & Governance: Constitutional Studies, International Law, Jurisprudence, Civil/Criminal Legal Concepts (Provide educational information; not jurisdiction-specific legal advice).
- Other Fields: Education & Pedagogy, Research Methodology, Agriculture, Architecture, Fine Arts, Environmental Studies, General Knowledge.

### 2. WORLDWIDE EDUCATION SYSTEMS & CURRICULA
Understand and adapt to curricula from:
- India: CBSE, ICSE, State Boards, UGC, AICTE, IIT JEE, NEET, GATE.
- International: Cambridge (IGCSE, A-Levels), International Baccalaureate (IB DP/MYP).
- United States: K-12, AP (Advanced Placement), College/University undergraduate & graduate.
- United Kingdom, Canada, Australia, Europe, and other national educational frameworks.
- Never assume all students follow one single syllabus. When a question is board-specific, identify the relevant context.

### 3. ALL EDUCATIONAL LEVELS
Adapt explanation depth appropriately:
- Primary / Middle School: Intuitive analogies, clear everyday language, foundational clarity.
- Secondary / High School: Systematic explanations with formal definitions and standard formulas.
- Undergraduate / Postgraduate: Technically rigorous, formal notation, academic citations of laws/principles.
- Research / Doctoral: Critical analysis, methodologies, mathematical rigor, limitations, and empirical frameworks.

### 4. ACADEMIC FORMATTING STANDARDS
- Definition Questions: Provide formal definition, conceptual breakdown, and clear example.
- Mathematical Questions: State given values, required result, formula, step-by-step derivation using LaTeX ($...$ inline, $$...$$ block), and final verified answer.
- Programming Questions: Explain algorithm approach, provide complete working code with syntax highlighting, display expected output, explain key lines, and state Time & Space Complexity using Big-O notation.
- Comparison Questions: Use clean Markdown tables with distinct evaluation criteria columns.
- Exam Answers by Marks:
  - 2 Marks: Concise, point-focused (30-50 words).
  - 5 Marks: Definition, 3-4 structured bullet points with formula/diagram outline (100-150 words).
  - 10/15 Marks: Introduction, detailed architecture/mechanism, evaluation points, comparison table, and summary conclusion (250-400 words).
- Multilingual: If the question is asked in Bengali, Hindi, French, Spanish, or another language, reply fluently and accurately in that language with correct academic terminology.
- Follow-ups: Retain conversation memory contextually (e.g. "What are its advantages?" refers to the preceding topic).
- Document Context: If an attached document or image is provided, prioritize its actual contents and never hallucinate outside the text.`;

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
      service: 'Brainware Academic AI Worldwide Backend',
      version: '2.1.0',
      geminiConfigured: !!(process.env.GEMINI_API_KEY || GEMINI_API_KEY),
      timestamp: new Date().toISOString()
    }));
    return;
  }

  // --- SECURE CHAT API ENDPOINT ---
  if (req.method === 'POST' && pathname === '/api/chat') {
    if (!checkRateLimit(clientIp)) {
      res.writeHead(429, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        success: false,
        error: 'Too many requests. Rate limit is 60 requests per minute per client.'
      }));
      return;
    }

    let body = '';
    req.on('data', chunk => {
      body += chunk;
      if (body.length > 25 * 1024 * 1024) { // 25MB limit (for base64 documents/images)
        res.writeHead(413, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: false, error: 'Payload too large (max 25MB)' }));
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

        const contents = [];
        const recentHistory = history.slice(-14);
        for (const item of recentHistory) {
          if (item && item.text) {
            contents.push({
              role: item.role === 'model' || item.role === 'assistant' ? 'model' : 'user',
              parts: [{ text: item.text }]
            });
          }
        }

        const currentParts = [];
        if (file && file.base64 && file.base64.startsWith('data:image/')) {
          const base64Data = file.base64.split(',')[1];
          currentParts.push({
            inlineData: {
              mimeType: file.mimeType || 'image/jpeg',
              data: base64Data
            }
          });
        }

        let promptText = query;
        if (file && file.text) {
          promptText = `[ATTACHED EDUCATIONAL DOCUMENT: ${file.name} (${file.type || 'FILE'})]\n${file.text.substring(0, 18000)}\n\n[STUDENT QUESTION]:\n${query || 'Please analyze this document thoroughly and answer all questions.'}`;
        }

        currentParts.push({ text: promptText });
        contents.push({
          role: 'user',
          parts: currentParts
        });

        const geminiEndpoint = `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${encodeURIComponent(activeKey)}`;

        const geminiRequestBody = {
          contents: contents,
          systemInstruction: {
            parts: [{ text: SYSTEM_INSTRUCTION }]
          },
          generationConfig: {
            temperature: 0.2,
            topP: 0.95,
            maxOutputTokens: 4000
          }
        };

        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 28000);

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

  // --- STATIC FILE SERVING ---
  if (req.method === 'GET') {
    let filePath = path.join(__dirname, pathname === '/' ? 'index.html' : pathname);

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
  console.log(`  Brainware Academic AI - Universal Server Running`);
  console.log(`  URL: http://localhost:${PORT}`);
  console.log(`  Health Check: http://localhost:${PORT}/api/health`);
  console.log(`  API Endpoint: http://localhost:${PORT}/api/chat (POST)`);
  console.log(`  Gemini Key Configured: ${!!(process.env.GEMINI_API_KEY || GEMINI_API_KEY)}`);
  console.log(`====================================================`);
});

process.on('SIGINT', () => {
  console.log('\nShutting down Brainware Academic AI Server...');
  server.close(() => process.exit(0));
});
