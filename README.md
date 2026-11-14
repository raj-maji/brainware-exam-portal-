# Brainware Academic AI - Enterprise Universal Educational Assistant

> **A worldwide, multi-disciplinary, LLM-powered educational AI assistant engineered for students, educators, and researchers across all academic levels, countries, and educational boards.**

---

## 1. System Architecture

```
[Student / User]
       │
       ▼
[Frontend Web Interface] (index.html, ai_tutor_engine.js)
       │  • PDF / Word / Excel / PPTX Document Parsing
       │  • Image Vision Preprocessing
       │  • Multi-turn Conversation State
       │
       ▼ (HTTP POST /api/chat)
[Secure Backend Server] (server.js - Node.js v18+)
       │  • Token-Bucket Rate Limiter (60 req/min per IP)
       │  • Isolated Secret Management (.env)
       │  • Master System Instruction (Global Curricula)
       │  • Multi-Model Auto-Fallback & Exponential Retry
       │
       ▼ (HTTPS Official Google REST API)
[Google Gemini 1.5 Flash / Pro / 2.0 Flash]
       │  • Deep Multi-Disciplinary Reasoning
       │  • Multimodal Document & Vision OCR
       │  • Google Search Grounding Tool
       │
       ▼
[Response Validation & Markdown Formatting]
       │
       ▼
[Rendered Academic Solution in Client]
```

---

## 2. Windows Installation & Quick Start

### Step 1: Prerequisites
- Ensure **Node.js (v18.0.0 or higher)** is installed on your computer.
- Verify in PowerShell or Command Prompt:
  ```powershell
  node -v
  ```

### Step 2: Configure Environment Secret
1. In the project root directory, copy `.env.example` to `.env`:
   ```powershell
   Copy-Item .env.example .env
   ```
2. Open `.env` in any text editor (Notepad, VS Code, etc.) and paste your free Google Gemini API Key:
   ```env
   # Obtain free key at: https://aistudio.google.com/app/apikey
   GEMINI_API_KEY=AIzaSyYourActualKeyHere
   PORT=5000
   GEMINI_MODEL=gemini-1.5-flash
   ```

### Step 3: Launch the Backend Server
Run the standard start command:
```powershell
npm start
```
*Or directly via Node.js:*
```powershell
node server.js
```

You will see:
```
================================================================
  Brainware Academic AI - Enterprise Universal Server Running
  URL: http://localhost:5000
  Health Check: http://localhost:5000/api/health
  API Endpoint: http://localhost:5000/api/chat (POST)
  Primary Model: gemini-1.5-flash
  Gemini Key Configured: true
================================================================
```

### Step 4: Open in Your Browser
Open: **`http://localhost:5000`** in Google Chrome, Microsoft Edge, or Mozilla Firefox.

---

## 3. Mandatory 20-Benchmark Verification Suite

Execute the built-in automated test suite covering all 20 required disciplines:
```powershell
node ../test_20_mandatory_questions.js
```

### Verified Benchmark Coverage:
1. **What is Java?** *(Object-Oriented Architecture, JVM Bytecode, Garbage Collection, Working Code)*
2. **What is a C++ programming language?** *(Multi-Paradigm, STL, Manual Memory Control, Sorting Vector)*
3. **Explain Data Structures.** *(Linear vs Non-Linear Hierarchy, Big-O Complexity Matrix)*
4. **What is DBMS?** *(ACID Transaction Properties, Relational RDBMS vs NoSQL Table)*
5. **Explain polymorphism in Java with an example.** *(Compile-Time vs Runtime, Java Code Example)*
6. **Write a JavaScript program to check whether a number is prime.** *($O(\sqrt{n})$ Optimal Algorithm with $6k \pm 1$ Optimization)*
7. **What is the difference between HTTP and HTTPS?** *(Port 80 vs 443, TLS 1.3 Cryptographic Handshake)*
8. **Solve the equation 2x + 5 = 15.** *(Subtraction & Division Properties of Equality, Verification Proof)*
9. **Explain Newton's second law of motion.** *(Momentum Derivation $\vec{F} = \frac{d\vec{p}}{dt} = m\vec{a}$)*
10. **What is photosynthesis?** *(Balanced Equation, Light-Dependent Phase, Thylakoid, Calvin Cycle)*
11. **Explain the process of DNA replication.** *(Semiconservative Model, Helicase, Primase, Polymerase III, Ligase)*
12. **What is opportunity cost in economics?** *(Scarcity, PPF Marginal Rate of Transformation, Trade-offs)*
13. **Explain the causes of World War I.** *(M-A-I-N Framework + Assassination of Archduke Franz Ferdinand)*
14. **What is machine learning?** *(Supervised, Unsupervised, Reinforcement Learning Paradigms)*
15. **Explain the working of a neural network.** *(Input/Hidden/Output Layers, ReLU/Sigmoid, Forward & Backpropagation)*
16. **What is the difference between mitosis and meiosis?** *(Somatic vs Germ Cells, Diploid vs Haploid, Crossing Over)*
17. **Explain the law of demand.** *(Ceteris Paribus, Downward Slope, Substitution & Income Effects, Giffen Goods)*
18. **What is research methodology?** *(Problem Formulation, Literature Review, Sampling, Statistical Testing)*
19. **Explain the water cycle.** *(Evaporation, Transpiration, Adiabatic Condensation, Precipitation, Runoff)*
20. **Explain Java in Bengali.** *(বাংলা ভাষায় বিশুদ্ধ ব্যাখ্যা, WORA, বাইটকোড, JVM, ক্লাস ও অবজেক্ট)*

---

## 4. Multi-Format Document Ingestion
The platform supports multi-page document parsing and OCR via the paperclip attachment button:
- **PDF Documents (`.pdf`):** Multi-page full text extraction via PDF.js.
- **Word Documents (`.docx`, `.doc`):** Raw text extraction via Mammoth.
- **Excel & Spreadsheets (`.xlsx`, `.xls`, `.csv`):** Tabular sheet extraction via SheetJS.
- **PowerPoint (`.pptx`, `.ppt`):** XML slide text extraction via JSZip.
- **Source Code & Text:** Python, C, C++, Java, JavaScript, TypeScript, SQL, JSON, Markdown.
- **Images (`.png`, `.jpg`, `.jpeg`, `.webp`):** Base64 multimodal vision extraction via Gemini Vision.
