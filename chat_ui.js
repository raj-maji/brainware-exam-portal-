/**
 * Brainware Academic AI - Enterprise Multi-Device Frontend UI Controller
 * File: chat_ui.js
 * Version: 7.2.0 - Universal Device Compatibility & Resilient Campus Fallback
 * 
 * Responsibilities:
 * - Seamless operation on ANY device (Desktop, Laptop, iPhone, Android, Tablet)
 * - Automatic Python Machine Learning Backend Discovery (Local, LAN Wi-Fi IP, Render Cloud, Custom URL)
 * - Zero-Failure Client-Side Campus RAG Fallback (2,264 verified university questions & academic knowledge base)
 * - Interactive Server Settings Modal with connection testing
 * - Markdown, KaTeX LaTeX math, Mermaid diagrams, and zoomable image modal
 */

(function () {
  'use strict';

  // --- UI CLIENT STATE ---
  let isGenerating = false;
  let chatHistory = []; // { role: 'user' | 'assistant' | 'model', text: string }
  let attachedFileData = null;
  let activeBackendUrl = '/api/chat';
  let isBackendOnline = false;

  // =========================================================================
  // 1. SMART BACKEND DISCOVERY & MULTI-DEVICE NETWORKING
  // =========================================================================
  async function resolveBackendEndpoint() {
    const isGitHubPages = window.location.hostname.endsWith('github.io');
    const customUrl = localStorage.getItem('brainware_backend_url');

    const candidates = [];
    if (customUrl) {
      candidates.push(customUrl.replace(/\/$/, '') + (customUrl.endsWith('/api/chat') ? '' : '/api/chat'));
    }

    // Relative endpoint (active when served by Python server directly or on Render/Railway)
    if (!isGitHubPages) {
      candidates.push('/api/chat');
    }

    // Standard local & Wi-Fi LAN IP candidates (enables phones on same Wi-Fi to reach host PC)
    candidates.push('http://localhost:5000/api/chat');
    candidates.push('http://127.0.0.1:5000/api/chat');
    candidates.push('http://10.152.113.39:5000/api/chat');

    for (const cand of candidates) {
      try {
        const testUrl = cand.replace(/\/chat$/, '/health');
        const ctrl = new AbortController();
        const tid = setTimeout(() => ctrl.abort(), 1800);
        const res = await fetch(testUrl, { method: 'GET', signal: ctrl.signal });
        clearTimeout(tid);
        if (res.ok) {
          activeBackendUrl = cand;
          isBackendOnline = true;
          updateBackendStatusUI(true, cand);
          return activeBackendUrl;
        }
      } catch (e) {
        // Candidate unreachable
      }
    }

    isBackendOnline = false;
    updateBackendStatusUI(false, null);
    return null;
  }

  function updateBackendStatusUI(online, url) {
    const dots = document.querySelectorAll('.backend-status-dot');
    const labels = document.querySelectorAll('.backend-status-label');

    dots.forEach(d => {
      d.className = `w-2 h-2 rounded-full inline-block ${online ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'}`;
    });

    labels.forEach(l => {
      l.textContent = online ? 'Python AI (Online)' : 'Campus Mode (Offline)';
    });
  }

  // =========================================================================
  // 2. DOCUMENT ATTACHMENT HANDLER (PDF, Word, Excel, PPTX, Code, Images)
  // =========================================================================
  async function parseUploadedDocument(file) {
    const fileName = file.name || 'uploaded_document';
    const fileSize = (file.size / 1024).toFixed(1) + ' KB';
    const ext = fileName.split('.').pop().toLowerCase();
    let extractedText = '';
    let base64Data = null;
    const mimeType = file.type || '';

    try {
      if (ext === 'pdf') {
        if (typeof pdfjsLib !== 'undefined') {
          try {
            const arrayBuf = await file.arrayBuffer();
            const loadingTask = pdfjsLib.getDocument({ data: arrayBuf });
            const pdfDoc = await loadingTask.promise;
            const pageTexts = [];
            const maxPages = Math.min(pdfDoc.numPages, 40);
            for (let pageNum = 1; pageNum <= maxPages; pageNum++) {
              const page = await pdfDoc.getPage(pageNum);
              const content = await page.getTextContent();
              const pageStrings = content.items.map(item => item.str).filter(Boolean);
              if (pageStrings.length > 0) {
                pageTexts.push(`[Page ${pageNum}]: ` + pageStrings.join(' '));
              }
            }
            extractedText = pageTexts.join('\n\n');
          } catch (pdfErr) {
            console.warn('PDF extraction notice:', pdfErr);
          }
        }
        if (!extractedText) extractedText = `[PDF Document: ${fileName}, size: ${fileSize}]`;
      } else if (ext === 'docx') {
        if (typeof mammoth !== 'undefined') {
          try {
            const arrayBuffer = await file.arrayBuffer();
            const result = await mammoth.extractRawText({ arrayBuffer: arrayBuffer });
            extractedText = result.value || '';
          } catch (e) {
            console.warn('Mammoth extraction notice:', e);
          }
        }
        if (!extractedText) extractedText = `[Word Document: ${fileName}, size: ${fileSize}]`;
      } else if (['xlsx', 'xls', 'csv'].includes(ext)) {
        if (typeof XLSX !== 'undefined') {
          try {
            const data = await file.arrayBuffer();
            const workbook = XLSX.read(data, { type: 'array' });
            const sheetTexts = [];
            workbook.SheetNames.forEach(sheetName => {
              const sheet = workbook.Sheets[sheetName];
              const csv = XLSX.utils.sheet_to_csv(sheet);
              sheetTexts.push(`--- Sheet: ${sheetName} ---\n${csv.substring(0, 6000)}`);
            });
            extractedText = sheetTexts.join('\n\n');
          } catch (e) {
            console.warn('SheetJS extraction notice:', e);
          }
        }
        if (!extractedText) extractedText = `[Spreadsheet: ${fileName}, size: ${fileSize}]`;
      } else if (ext === 'pptx') {
        if (typeof JSZip !== 'undefined') {
          try {
            const zip = await JSZip.loadAsync(file);
            const slideTexts = [];
            const slideFiles = Object.keys(zip.files).filter(f => f.match(/ppt\/slides\/slide\d+\.xml/));
            slideFiles.sort((a, b) => {
              const numA = parseInt(a.match(/\d+/)[0], 10);
              const numB = parseInt(b.match(/\d+/)[0], 10);
              return numA - numB;
            });
            for (let i = 0; i < Math.min(slideFiles.length, 30); i++) {
              const xml = await zip.file(slideFiles[i]).async('text');
              const textMatches = xml.match(/<a:t>([^<]+)<\/a:t>/g) || [];
              const slideContent = textMatches.map(t => t.replace(/<\/?a:t>/g, '')).join(' ');
              if (slideContent.trim()) {
                slideTexts.push(`[Slide ${i + 1}]: ${slideContent}`);
              }
            }
            extractedText = slideTexts.join('\n\n');
          } catch (e) {
            console.warn('PPTX extraction notice:', e);
          }
        }
        if (!extractedText) extractedText = `[Presentation: ${fileName}, size: ${fileSize}]`;
      } else if (file.type && file.type.startsWith('image/')) {
        base64Data = await new Promise((resolve) => {
          const reader = new FileReader();
          reader.onload = (e) => resolve(e.target.result);
          reader.onerror = () => resolve(null);
          reader.readAsDataURL(file);
        });
        extractedText = `[Image Attached: ${fileName} (${fileSize})]`;
      } else {
        extractedText = await new Promise((resolve) => {
          const reader = new FileReader();
          reader.onload = (e) => resolve(e.target.result);
          reader.onerror = () => resolve('');
          reader.readAsText(file);
        });
      }
    } catch (err) {
      console.error('File parsing error:', err);
      extractedText = `[Uploaded File: ${fileName} (${fileSize})]`;
    }

    return {
      name: fileName,
      type: ext.toUpperCase(),
      size: fileSize,
      mimeType: mimeType,
      text: extractedText,
      base64: base64Data
    };
  }

  // =========================================================================
  // 3. HTTP CLIENT CALL TO PYTHON BACKEND (/api/chat)
  // =========================================================================
  async function callPythonAcademicBackend(query, history = [], fileSnapshot = null) {
    const endpoint = await resolveBackendEndpoint();
    if (!endpoint || !isBackendOnline) {
      throw new Error('BACKEND_OFFLINE');
    }

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 60000);

    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query,
          history: history.slice(-16),
          file: fileSnapshot
        }),
        signal: controller.signal
      });
      clearTimeout(timeout);

      if (response.ok) {
        const data = await response.json();
        if (data && data.success && data.text) {
          return {
            text: data.text,
            source: data.source || 'python-ml-model',
            model: data.model || 'Brainware Academic AI'
          };
        } else if (data && data.error) {
          throw new Error(data.error);
        }
      }
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.error || `HTTP ${response.status} from backend`);
    } catch (err) {
      clearTimeout(timeout);
      throw err;
    }
  }

  // =========================================================================
  // 4. RESILIENT CLIENT-SIDE ACADEMIC RAG & KNOWLEDGE ENGINE
  // =========================================================================
  function generateClientSideAcademicResponse(query, history = [], fileSnapshot = null, originalError = null) {
    const cleanQ = query.toLowerCase().trim();
    const stopwords = new Set(['what', 'is', 'the', 'a', 'an', 'and', 'in', 'of', 'to', 'for', 'explain', 'define', 'how', 'describe', 'discuss', 'with', 'an', 'example', 'type', 'types']);
    const tokens = cleanQ.replace(/[^a-z0-9\s]/g, ' ').split(/\s+/).filter(t => t.length > 2 && !stopwords.has(t));

    // A. Check against 2,264 indexed university syllabus QA solutions
    if (window.portalQAIndex && Array.isArray(window.portalQAIndex) && window.portalQAIndex.length > 0 && tokens.length > 0) {
      let bestItem = null;
      let bestScore = -1;

      for (const item of window.portalQAIndex) {
        const itemQ = (item.q || '').toLowerCase();
        let score = 0;

        // Exact substring
        if (itemQ.includes(cleanQ)) {
          score += 60;
        }

        // Word-level matching with word boundaries
        for (const token of tokens) {
          const regex = new RegExp('\\b' + token + '\\b', 'i');
          if (regex.test(itemQ)) {
            score += 15;
          }
        }

        if (score > bestScore) {
          bestScore = score;
          bestItem = item;
        }
      }

      if (bestScore >= 15 && bestItem) {
        let answerText = `### ${bestItem.q}\n\n`;
        answerText += `**Academic Curriculum:** ${bestItem.subj || 'University Master Question Bank'} | **Assessment Classification:** ${bestItem.kind || 'Curricular Solution'}\n\n---\n\n`;
        answerText += convertHtmlToCleanMarkdown(bestItem.a);

        if (!isBackendOnline) {
          answerText += `\n\n> ℹ️ *Answered via Campus Knowledge Base (2,264 Verified Solutions). To connect this mobile device to your live Python AI server, tap **Server** (⚙️) in the header.*`;
        }

        return {
          text: answerText,
          source: 'campus-syllabus-rag',
          model: 'Campus Knowledge Base (2,264 Verified QA Cache)'
        };
      }
    }

    // B. Built-in Academic Synthesizer for Core Computer Science & STEM Concepts
    const synthesized = synthesizeFundamentalConcept(cleanQ);
    if (synthesized) {
      if (!isBackendOnline) {
        synthesized.text += `\n\n> ℹ️ *Operating in Campus Knowledge Mode. To connect this device to your live Python AI server, tap **Server** (⚙️) above or deploy to Render.*`;
      }
      return synthesized;
    }

    // C. Fallback Educational Response with Connection Diagnostics
    let fallbackText = `### University Examination Analysis: ${capitalizeTitle(query)}\n\n`;
    fallbackText += `**Disciplinary Classification:** Computer Science & Academic Curricula  \n`;
    fallbackText += `**Execution Mode:** Client Campus Knowledge Base (Offline-Resilient Engine)\n\n---\n\n`;
    fallbackText += `### 1. Conceptual Framework & Definition\n`;
    fallbackText += `In university curricula and academic examinations, **${query}** represents an essential core topic requiring thorough analytical comprehension, theoretical grounding, and practical application.\n\n`;
    fallbackText += `### 2. Core Curricular Principles\n`;
    fallbackText += `• **Foundational Principle:** Emphasizes systematic abstraction, modularity, and reproducible computation.\n`;
    fallbackText += `• **Architectural Rigor:** Requires adherence to formal specifications, optimal algorithmic time/space complexities, and clean separation of concerns.\n`;
    fallbackText += `• **Examination Criteria:** In degree examinations (such as MCA, B.Tech, BCA), responses must combine precise definitions, structural diagrams, and verified code implementations.\n\n`;
    fallbackText += `### 3. Practical Summary & Next Steps\n`;
    fallbackText += `For comprehensive derivations and interactive neural generation across unfamiliar queries, ensure your Python Machine Learning server is reachable from this device.\n\n`;
    fallbackText += `> 💡 **Multi-Device Connection Tip:** If accessing from a smartphone on the same Wi-Fi, click **Server (⚙️)** in the top header and enter your PC's IP address: \`http://10.152.113.39:5000\`.`;

    return {
      text: fallbackText,
      source: 'campus-concept-engine',
      model: 'Campus Academic Synthesizer'
    };
  }

  function synthesizeFundamentalConcept(query) {
    // Java
    if (/\bjava\b/.test(query) && !/\bjavascript\b/.test(query)) {
      return {
        text: [
          '### University Examination Analysis: Java Programming Language & Architecture',
          '',
          '**Disciplinary Classification:** Computer Science & Software Engineering',
          '**Curricular Reference:** Object-Oriented Software Development',
          '',
          '---',
          '',
          '### 1. Conceptual Definition & Paradigm',
          '**Java** is a high-level, class-based, object-oriented, concurrent, and secure programming language developed by Sun Microsystems (now Oracle Corporation) in 1995. It was engineered under the core philosophy of **WORA ("Write Once, Run Anywhere")**, enabling compiled Java code to execute across any operating system without recompilation.',
          '',
          '### 2. Architectural Pillars',
          '• **Java Virtual Machine (JVM):** Abstract execution engine that loads, verifies, and executes Java bytecode, performing Just-In-Time (JIT) compilation to native CPU instructions.',
          '• **Java Runtime Environment (JRE):** Provides class libraries and runtime binaries necessary to run Java programs.',
          '• **Java Development Kit (JDK):** Comprehensive development suite containing the compiler (`javac`), debugger, archiver (`jar`), and JRE.',
          '• **Bytecode (`.class` files):** Intermediate machine-independent bytecode produced by `javac`.',
          '• **Automatic Memory Management:** Garbage Collector (GC) dynamically reclaims unreferenced heap objects, preventing dangling pointers and memory leaks.',
          '',
          '### 3. Core Object-Oriented Principles (OOP)',
          '1. **Encapsulation:** Binding data and methods into a single unit (class) with access specifiers (`private`, `protected`, `public`).',
          '2. **Inheritance:** Hierarchical code reusability using the `extends` keyword.',
          '3. **Polymorphism:** Method Overloading (compile-time) and Method Overriding (runtime).',
          '4. **Abstraction:** Hiding complex implementation details using `abstract` classes and `interface` contracts.',
          '',
          '### 4. Implementation Demonstration',
          '```java',
          '// Demonstration of OOP principles and execution in Java',
          'public class JavaDemo {',
          '    public static void main(String[] args) {',
          '        System.out.println("Java: Robust, Platform-Independent, Object-Oriented Execution");',
          '    }',
          '}',
          '```',
          '',
          '### 5. Examination Summary',
          'Java delivers cross-platform portability through bytecode interpretation on the JVM, robust memory safety through garbage collection, and strict type safety across enterprise and academic systems.'
        ].join('\n'),
        source: 'campus-concept-engine',
        model: 'Universal Computer Science Knowledge Base'
      };
    }

    // Python
    if (/\bpython\b/.test(query)) {
      return {
        text: [
          '### University Examination Analysis: Python Programming Language',
          '',
          '**Disciplinary Classification:** Computer Science & Data Intelligence',
          '**Curricular Reference:** Interpreted High-Level Programming & Machine Learning',
          '',
          '---',
          '',
          '### 1. Conceptual Definition & Paradigm',
          '**Python** is an interpreted, high-level, dynamically typed, multi-paradigm programming language created by Guido van Rossum in 1991. It emphasizes exceptional code readability and clean syntax through PEP 8 standards, supporting object-oriented, functional, and procedural programming.',
          '',
          '### 2. Architectural Features',
          '• **Dynamic Typing:** Variable types are resolved at runtime without explicit declarations.',
          '• **Comprehensive Ecosystem:** De-facto standard for AI, Machine Learning, Data Science (NumPy, SciPy, Pandas, Scikit-Learn, PyTorch, TensorFlow).',
          '• **Automatic Memory Management:** Utilizes reference counting and cyclic generational garbage collection.',
          '• **Interpreted Execution:** Source code is translated into bytecode (`.pyc`) and executed by the Python Virtual Machine (PVM).'
        ].join('\n'),
        source: 'campus-concept-engine',
        model: 'Universal Computer Science Knowledge Base'
      };
    }

    // DBMS
    if (/\bdbms\b/.test(query) || /\bdatabase management\b/.test(query)) {
      return {
        text: [
          '### University Examination Analysis: Database Management System (DBMS)',
          '',
          '**Disciplinary Classification:** Computer Science & Information Technology',
          '**Curricular Reference:** Relational Database Systems & Transaction Processing',
          '',
          '---',
          '',
          '### 1. Definition & System Architecture',
          'A **Database Management System (DBMS)** is specialized system software designed to define, construct, manipulate, query, and maintain persistent, structured databases while guaranteeing data integrity, security, and multi-user concurrency control.',
          '',
          '### 2. The ACID Guarantee Matrix',
          '| Property | Academic Definition | Mechanism |',
          '| :--- | :--- | :--- |',
          '| **Atomicity** | "All or nothing" transaction execution | Write-Ahead Logging (WAL) & Rollbacks |',
          '| **Consistency** | Database transitions from one valid state to another | Integrity constraints & Foreign Keys |',
          '| **Isolation** | Concurrent transactions execute without interference | Multi-Version Concurrency Control (MVCC) / Two-Phase Locking (2PL) |',
          '| **Durability** | Committed data persists across system crashes | Non-volatile disk writes & Redo logs |'
        ].join('\n'),
        source: 'campus-concept-engine',
        model: 'Universal Computer Science Knowledge Base'
      };
    }

    // Android
    if (/\bandroid\b/.test(query) && (/\btechnology\b/.test(query) || /\bwhat is\b/.test(query) || /\barchitecture\b/.test(query))) {
      return {
        text: [
          '### University Examination Analysis: Android Operating System & Architecture',
          '',
          '**Disciplinary Classification:** Mobile Computing & Embedded Systems',
          '**Curricular Reference:** MCA37510A - Android Application Development',
          '',
          '---',
          '',
          '### 1. Definition',
          '**Android** is an open-source, Linux-based mobile operating system developed by Google and the Open Handset Alliance. It is engineered for touchscreen mobile devices, smart TVs, wearables, and embedded automotive interfaces.',
          '',
          '### 2. Five-Layer Architecture',
          '1. **Linux Kernel:** Device drivers (Camera, Display, Wi-Fi), power management, process scheduling, and low-level memory allocation.',
          '2. **Hardware Abstraction Layer (HAL):** Standard interfaces exposing device hardware capabilities to high-level Java API framework.',
          '3. **Android Runtime (ART):** Ahead-of-Time (AOT) and Just-in-Time (JIT) compiler converting DEX (Dalvik Executable) bytecode into native machine instructions.',
          '4. **Application Framework:** Core APIs including ActivityManager, NotificationManager, ContentProviders, and WindowManager.',
          '5. **Applications:** Built-in and third-party user-facing apps.',
          '',
          '### 3. Core Android Components',
          '• **Activity:** Single UI screen with user interaction.',
          '• **Service:** Background operation with no UI.',
          '• **Broadcast Receiver:** System-wide message listener.',
          '• **Content Provider:** Manages shared application data (e.g. SQLite databases).'
        ].join('\n'),
        source: 'campus-concept-engine',
        model: 'Universal Computer Science Knowledge Base'
      };
    }

    return null;
  }

  function convertHtmlToCleanMarkdown(html) {
    if (!html) return '';
    let md = html;
    // Extract images
    md = md.replace(/<img[^>]+src=["']([^"']+)["'][^>]*alt=["']([^"']*)["'][^>]*>/gi, '![$2]($1)');
    md = md.replace(/<img[^>]+src=["']([^"']+)["'][^>]*>/gi, '![Diagram]($1)');
    // List items
    md = md.replace(/<li[^>]*>([\s\S]*?)<\/li>/gi, '- $1\n');
    // Strong / Bold
    md = md.replace(/<(?:strong|b)[^>]*>([\s\S]*?)<\/(?:strong|b)>/gi, '**$1**');
    // Paragraphs & Divs
    md = md.replace(/<p[^>]*>([\s\S]*?)<\/p>/gi, '$1\n\n');
    md = md.replace(/<div class="[^"]*font-bold[^"]*">([\s\S]*?)<\/div>/gi, '**$1**\n\n');
    md = md.replace(/<div class="[^"]*p-3 bg-emerald-50[^"]*">([\s\S]*?)<\/div>/gi, '> $1\n\n');
    md = md.replace(/<[^>]+>/g, '');
    md = md.replace(/&quot;/g, '"').replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&#39;/g, "'");
    return md.trim();
  }

  function capitalizeTitle(str) {
    return str.split(' ').map(w => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase()).join(' ');
  }

  // =========================================================================
  // 5. MARKDOWN & KATEX LATEX RENDERING
  // =========================================================================
  function renderMarkdown(md) {
    if (!md) return '';
    let html = md;

    // Mermaid diagrams
    html = html.replace(/```mermaid\n([\s\S]*?)```/g, (match, code) => {
      return `
        <div class="my-3 p-3 bg-slate-50/80 border border-slate-200 rounded-xl overflow-x-auto flex justify-center shadow-sm">
          <div class="mermaid text-xs font-sans">${code.trim()}</div>
        </div>
      `;
    });

    // Code blocks with syntax highlighting & Copy Button
    html = html.replace(/```([a-zA-Z0-9_-]+)?\n([\s\S]*?)```/g, (match, lang, code) => {
      const language = lang || 'code';
      const escapedCode = code.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
      return `
        <div class="relative group my-3 rounded-xl overflow-hidden border border-slate-700 bg-slate-950 text-slate-100 font-mono text-xs shadow-md">
          <div class="flex items-center justify-between px-3.5 py-1.5 bg-slate-900 border-b border-slate-800 text-[11px] text-slate-400 font-sans">
            <span class="font-semibold uppercase tracking-wider">${language}</span>
            <button type="button" onclick="BrainwareAITutor.copyCodeSnippet(this)" class="hover:text-white transition flex items-center gap-1">
              <i data-lucide="copy" class="w-3 h-3"></i> Copy
            </button>
          </div>
          <pre class="p-3.5 overflow-x-auto leading-relaxed"><code>${escapedCode}</code></pre>
        </div>
      `;
    });

    // Markdown Images: ![alt](src)
    html = html.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, (match, alt, src) => {
      const cleanAlt = alt || 'Exam Diagram / Figure';
      const cleanSrc = src.trim();
      const escapedAlt = cleanAlt.replace(/'/g, "\\'");
      return `
        <div class="my-4 p-3 bg-slate-50/90 border border-slate-200/90 rounded-2xl shadow-sm text-center exam-image-container">
          <div class="overflow-hidden rounded-xl bg-white border border-slate-100 flex items-center justify-center p-2.5">
            <img src="${cleanSrc}" alt="${cleanAlt}" loading="lazy" class="max-w-full h-auto max-h-96 mx-auto rounded-lg object-contain cursor-zoom-in hover:scale-[1.01] transition" onclick="BrainwareAITutor.zoomImage(this.src, '${escapedAlt}')" onerror="this.parentElement.innerHTML='<div class=\\'p-3 text-xs text-slate-400 italic\\'>Diagram Image Preview: ${cleanAlt}</div>'"/>
          </div>
          <div class="flex items-center justify-between mt-2.5 px-2 text-[11px] text-slate-600 font-medium">
            <span class="flex items-center gap-1.5"><i data-lucide="image" class="w-3.5 h-3.5 text-blue-600"></i> ${cleanAlt}</span>
            <button type="button" onclick="BrainwareAITutor.zoomImage('${cleanSrc}', '${escapedAlt}')" class="text-blue-600 hover:text-blue-800 font-bold flex items-center gap-1 text-[11px]">
              <i data-lucide="maximize-2" class="w-3 h-3"></i> Enlarge
            </button>
          </div>
        </div>
      `;
    });

    html = html.replace(/`([^`]+)`/g, '<code class="px-1.5 py-0.5 rounded bg-slate-100 border border-slate-200 text-blue-700 font-mono text-xs">$1</code>');
    html = html.replace(/^### (.*$)/gim, '<h3 class="text-sm sm:text-base font-bold text-slate-900 mt-3 mb-1.5 flex items-center gap-1.5">$1</h3>');
    html = html.replace(/^## (.*$)/gim, '<h2 class="text-base sm:text-lg font-black text-slate-950 mt-4 mb-2">$1</h2>');
    html = html.replace(/^# (.*$)/gim, '<h1 class="text-lg sm:text-xl font-black text-blue-900 mt-4 mb-2">$1</h1>');
    html = html.replace(/^---$/gim, '<hr class="my-3 border-t border-slate-200">');
    html = html.replace(/^> (.*$)/gim, '<blockquote class="border-l-4 border-blue-600 pl-3 py-1 my-2 bg-blue-50/60 text-slate-800 rounded-r text-xs leading-relaxed">$1</blockquote>');
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong class="font-bold text-slate-900">$1</strong>');
    html = html.replace(/\*([^*]+)\*/g, '<em class="italic text-slate-700">$1</em>');

    // Tables
    html = html.replace(/((?:\|[^\n]+\|(?:\r?\n|$))+)/g, (tableText) => {
      const rows = tableText.trim().split('\n').filter(r => r.trim());
      if (rows.length < 2) return tableText;
      let tableHtml = '<div class="overflow-x-auto my-3 rounded-xl border border-slate-200 shadow-sm"><table class="min-w-full text-xs text-left divide-y divide-slate-200">';
      
      rows.forEach((row, idx) => {
        if (row.includes('---')) return;
        const cells = row.split('|').filter((c, i, a) => i > 0 && i < a.length - 1);
        if (idx === 0) {
          tableHtml += '<thead class="bg-slate-100 font-bold text-slate-800"><tr>';
          cells.forEach(c => tableHtml += `<th class="px-3 py-2 whitespace-nowrap">${c.trim()}</th>`);
          tableHtml += '</tr></thead><tbody class="divide-y divide-slate-100 bg-white">';
        } else {
          tableHtml += '<tr class="hover:bg-slate-50 transition">';
          cells.forEach(c => tableHtml += `<td class="px-3 py-2 text-slate-700 leading-relaxed">${c.trim()}</td>`);
          tableHtml += '</tr>';
        }
      });
      tableHtml += '</tbody></table></div>';
      return tableHtml;
    });

    // Unordered lists
    html = html.replace(/^\s*[-*•]\s+(.*$)/gim, '<li class="text-xs sm:text-sm text-slate-700 leading-relaxed ml-4 list-disc">$1</li>');

    // Paragraph breaks
    html = html.replace(/\n\n+/g, '</p><p class="text-xs sm:text-sm text-slate-700 leading-relaxed my-2">');
    html = html.replace(/\n/g, '<br>');

    return `<p class="text-xs sm:text-sm text-slate-700 leading-relaxed my-2">${html}</p>`;
  }

  function appendMessageToUI(role, content, source = null, model = null) {
    const thread = document.getElementById('aiChatThread');
    const popupThread = document.getElementById('fbChatMessages');
    const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    const formattedContent = role === 'user' ? escapeHtml(content).replace(/\n/g, '<br>') : renderMarkdown(content);
    const sourceBadge = source ? `<span class="text-[10px] font-mono px-2 py-0.5 rounded-full bg-slate-100 border border-slate-200 text-slate-600 font-semibold ml-2">${source}</span>` : '';

    if (thread) {
      const bubble = document.createElement('div');
      bubble.className = `flex gap-3 mb-4 animate-fade-in ${role === 'user' ? 'justify-end' : ''}`;
      if (role === 'user') {
        bubble.innerHTML = `
          <div class="max-w-2xl bg-gradient-to-tr from-blue-600 to-indigo-600 text-white rounded-2xl rounded-tr-sm p-3.5 shadow-md text-xs sm:text-sm leading-relaxed">
            ${formattedContent}
            <div class="text-[10px] text-blue-200 text-right mt-1.5 font-sans">${timeStr}</div>
          </div>
        `;
      } else {
        bubble.innerHTML = `
          <div class="w-8 h-8 rounded-xl bg-gradient-to-tr from-blue-700 to-indigo-600 text-white flex items-center justify-center font-black flex-shrink-0 text-xs shadow-md">
            <i data-lucide="bot" class="w-4 h-4"></i>
          </div>
          <div class="max-w-3xl bg-white border border-slate-200/90 rounded-2xl rounded-tl-sm p-4 shadow-sm text-xs sm:text-sm text-slate-800 leading-relaxed flex-1">
            <div class="flex items-center justify-between pb-2 mb-2 border-b border-slate-100 text-[11px] text-slate-400">
              <span class="font-bold text-slate-700 flex items-center">Brainware Academic AI ${sourceBadge}</span>
              <div class="flex items-center gap-2">
                <button type="button" onclick="BrainwareAITutor.copyMessageText(this)" class="hover:text-blue-600 transition flex items-center gap-1 text-[11px] font-medium" title="Copy answer">
                  <i data-lucide="copy" class="w-3 h-3"></i> Copy
                </button>
                <button type="button" onclick="BrainwareAITutor.speakMessage(this)" class="hover:text-blue-600 transition flex items-center gap-1 text-[11px] font-medium" title="Read aloud">
                  <i data-lucide="volume-2" class="w-3 h-3"></i>
                </button>
              </div>
            </div>
            <div class="ai-rendered-content">${formattedContent}</div>
            <div class="text-[10px] text-slate-400 text-right mt-2 font-sans">${timeStr}</div>
          </div>
        `;
      }
      thread.appendChild(bubble);
      thread.scrollTop = thread.scrollHeight;
    }

    if (popupThread) {
      const popBubble = document.createElement('div');
      popBubble.className = `flex gap-2 mb-3 ${role === 'user' ? 'justify-end' : ''}`;
      if (role === 'user') {
        popBubble.innerHTML = `
          <div class="max-w-[85%] bg-blue-600 text-white rounded-xl rounded-tr-sm p-2.5 text-xs shadow-sm">
            ${formattedContent}
          </div>
        `;
      } else {
        popBubble.innerHTML = `
          <div class="w-6 h-6 rounded-lg bg-blue-600 text-white flex items-center justify-center font-bold flex-shrink-0 text-[10px]">AI</div>
          <div class="max-w-[85%] bg-white border border-slate-200 rounded-xl rounded-tl-sm p-3 text-xs text-slate-800 leading-relaxed shadow-sm">
            <div class="ai-rendered-content">${formattedContent}</div>
          </div>
        `;
      }
      popupThread.appendChild(popBubble);
      popupThread.scrollTop = popupThread.scrollHeight;
    }

    // KaTeX LaTeX math auto-render
    if (typeof renderMathInElement !== 'undefined') {
      try {
        const lastEl = thread ? thread.lastElementChild : null;
        if (lastEl) {
          renderMathInElement(lastEl, {
            delimiters: [
              { left: '$$', right: '$$', display: true },
              { left: '$', right: '$', display: false }
            ],
            throwOnError: false
          });
        }
      } catch (e) {}
    }

    // Mermaid diagram auto-render
    if (typeof mermaid !== 'undefined') {
      try {
        const lastEl = thread ? thread.lastElementChild : null;
        if (lastEl) {
          const mNodes = lastEl.querySelectorAll('.mermaid');
          if (mNodes.length > 0) {
            mermaid.init(undefined, mNodes);
          }
        }
      } catch (e) {}
    }

    if (window.lucide && typeof window.lucide.createIcons === 'function') {
      window.lucide.createIcons();
    }
  }

  function showLoadingIndicator(isPopup = false) {
    const threadId = isPopup ? 'fbChatMessages' : 'aiChatThread';
    const thread = document.getElementById(threadId);
    if (!thread) return null;

    const loader = document.createElement('div');
    loader.id = isPopup ? 'fbAILoader' : 'mainAILoader';
    loader.className = 'flex gap-3 mb-4 animate-fade-in items-center text-slate-500 text-xs';
    loader.innerHTML = `
      <div class="w-8 h-8 rounded-xl bg-blue-50 border border-blue-200 text-blue-600 flex items-center justify-center font-black flex-shrink-0">
        <i data-lucide="loader-2" class="w-4 h-4 animate-spin"></i>
      </div>
      <div class="bg-white border border-slate-200 rounded-2xl rounded-tl-sm px-4 py-3 shadow-sm flex items-center gap-2">
        <span class="inline-block w-2 h-2 rounded-full bg-blue-600 animate-ping"></span>
        <span class="font-medium text-slate-700">${isBackendOnline ? 'Synthesizing response with Python Machine Learning Model...' : 'Searching Campus Knowledge Base (2,264 Verified Solutions)...'}</span>
      </div>
    `;
    thread.appendChild(loader);
    thread.scrollTop = thread.scrollHeight;
    if (window.lucide && typeof window.lucide.createIcons === 'function') window.lucide.createIcons();
    return loader;
  }

  function removeLoadingIndicator(isPopup = false) {
    const loaderId = isPopup ? 'fbAILoader' : 'mainAILoader';
    const loader = document.getElementById(loaderId);
    if (loader) loader.remove();
  }

  function escapeHtml(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  // =========================================================================
  // 6. PUBLIC UI CONTROLLER (window.BrainwareAITutor)
  // =========================================================================
  window.BrainwareAITutor = {
    handleFileUpload: async function (event, isPopup = false) {
      const file = event.target.files && event.target.files[0];
      if (!file) return;

      const previewId = isPopup ? 'fbFilePreview' : 'aiFilePreview';
      const nameId = isPopup ? 'fbFileName' : 'aiFileName';
      const sizeId = isPopup ? null : 'aiFileSize';

      const prevEl = document.getElementById(previewId);
      const nameEl = document.getElementById(nameId);
      const sizeEl = sizeId ? document.getElementById(sizeId) : null;

      if (prevEl && nameEl) {
        nameEl.innerText = file.name;
        if (sizeEl) sizeEl.innerText = '(Reading...)';
        prevEl.classList.remove('hidden');
        prevEl.style.display = 'flex';
      }

      try {
        attachedFileData = await parseUploadedDocument(file);
        if (sizeEl && attachedFileData) {
          sizeEl.innerText = `(${attachedFileData.size})`;
        }
      } catch (err) {
        console.error('File parsing notice:', err);
      }
    },

    removeAttachedFile: function (isPopup = false) {
      attachedFileData = null;
      const fileInputId = isPopup ? 'fbFileInput' : 'aiFileInput';
      const previewId = isPopup ? 'fbFilePreview' : 'aiFilePreview';

      const inputEl = document.getElementById(fileInputId);
      if (inputEl) inputEl.value = '';

      const prevEl = document.getElementById(previewId);
      if (prevEl) {
        prevEl.classList.add('hidden');
        prevEl.style.display = 'none';
      }
    },

    askQuestion: async function (customQuery = null, isPopup = false) {
      if (isGenerating) return;

      const inputId = isPopup ? 'fbChatInput' : 'aiTutorInput';
      const input = document.getElementById(inputId);
      let query = customQuery || (input ? input.value : '');

      if ((!query || !query.trim()) && attachedFileData) {
        query = `Analyze the attached educational document "${attachedFileData.name}" (${attachedFileData.type}) in detail. Provide an executive summary, list core formulas, and answer any embedded questions.`;
      }

      if (!query || !query.trim()) return;

      if (input && !customQuery) {
        input.value = '';
      }

      const fileSnapshot = attachedFileData ? { ...attachedFileData } : null;
      window.BrainwareAITutor.removeAttachedFile(isPopup);

      appendMessageToUI('user', query);
      chatHistory.push({ role: 'user', text: query });

      isGenerating = true;
      const sendBtn = document.getElementById(isPopup ? 'fbSendBtn' : 'aiSendBtn');
      if (sendBtn) sendBtn.disabled = true;

      showLoadingIndicator(isPopup);

      try {
        let response;
        if (isBackendOnline) {
          try {
            response = await callPythonAcademicBackend(query, chatHistory, fileSnapshot);
          } catch (backendErr) {
            console.warn('Backend call failed, using client campus fallback:', backendErr);
            response = generateClientSideAcademicResponse(query, chatHistory, fileSnapshot, backendErr);
          }
        } else {
          // Client-side instant campus search (2,264 university questions & solutions)
          response = generateClientSideAcademicResponse(query, chatHistory, fileSnapshot, null);
        }

        removeLoadingIndicator(isPopup);
        appendMessageToUI('model', response.text, response.source, response.model);
        chatHistory.push({ role: 'model', text: response.text });

      } catch (err) {
        console.error('AI Generation Error:', err);
        removeLoadingIndicator(isPopup);
        const fallback = generateClientSideAcademicResponse(query, chatHistory, fileSnapshot, err);
        appendMessageToUI('model', fallback.text, fallback.source, fallback.model);
        chatHistory.push({ role: 'model', text: fallback.text });
      } finally {
        isGenerating = false;
        if (sendBtn) sendBtn.disabled = false;
      }
    },

    copyMessageText: function (btn) {
      const bubble = btn.closest('.max-w-3xl') || btn.closest('.bg-white');
      const content = bubble ? bubble.querySelector('.ai-rendered-content') : null;
      if (content) {
        navigator.clipboard.writeText(content.innerText).then(() => {
          btn.innerHTML = '<i data-lucide="check" class="w-3 h-3 text-emerald-600"></i> Copied!';
          setTimeout(() => {
            btn.innerHTML = '<i data-lucide="copy" class="w-3 h-3"></i> Copy';
            if (window.lucide) window.lucide.createIcons();
          }, 2000);
        });
      }
    },

    copyCodeSnippet: function (btn) {
      const container = btn.closest('.relative');
      const code = container ? container.querySelector('code') : null;
      if (code) {
        navigator.clipboard.writeText(code.innerText).then(() => {
          btn.innerHTML = '<i data-lucide="check" class="w-3 h-3 text-emerald-400"></i> Copied!';
          setTimeout(() => {
            btn.innerHTML = '<i data-lucide="copy" class="w-3 h-3"></i> Copy';
            if (window.lucide) window.lucide.createIcons();
          }, 2000);
        });
      }
    },

    speakMessage: function (btn) {
      const bubble = btn.closest('.max-w-3xl') || btn.closest('.bg-white');
      const content = bubble ? bubble.querySelector('.ai-rendered-content') : null;
      if (!content || !window.speechSynthesis) return;

      if (window.speechSynthesis.speaking) {
        window.speechSynthesis.cancel();
        return;
      }

      const text = content.innerText.replace(/[\*\#\_\`]/g, '');
      const utterance = new SpeechSynthesisUtterance(text.substring(0, 1000));
      utterance.rate = 1.0;
      utterance.pitch = 1.0;
      window.speechSynthesis.speak(utterance);
    },

    clearChat: function () {
      chatHistory = [];
      const thread = document.getElementById('aiChatThread');
      if (thread) {
        thread.innerHTML = `
          <div class="flex gap-3 mb-4 animate-fade-in">
            <div class="w-8 h-8 rounded-xl bg-blue-600 text-white flex items-center justify-center font-black flex-shrink-0 text-xs shadow-sm">
              <i data-lucide="bot" class="w-4 h-4"></i>
            </div>
            <div class="bg-white border border-slate-200/80 rounded-2xl rounded-tl-sm p-3.5 shadow-sm text-xs sm:text-sm text-slate-800 leading-relaxed max-w-xl">
              <p class="font-bold text-slate-900 mb-1">Hello! I am Brainware Academic AI.</p>
              <p class="text-xs text-slate-600">Powered by Python & Machine Learning. How can I help you today?</p>
            </div>
          </div>
        `;
      }
      const popupThread = document.getElementById('fbChatMessages');
      if (popupThread) {
        popupThread.innerHTML = `
          <div class="flex gap-2 mb-3">
            <div class="w-7 h-7 rounded-lg bg-blue-600 text-white flex items-center justify-center font-bold flex-shrink-0 text-[10px]">AI</div>
            <div class="bg-white border border-slate-200 rounded-xl rounded-tl-sm p-2.5 text-xs text-slate-800 leading-relaxed shadow-sm">
              <p class="font-bold text-slate-900">How can I help you?</p>
              <p class="text-[11px] text-slate-600">Ask any educational question across all academic subjects.</p>
            </div>
          </div>
        `;
      }
      if (window.lucide && typeof window.lucide.createIcons === 'function') {
        window.lucide.createIcons();
      }
    },

    toggleChatPopup: function () {
      const popup = document.getElementById('fbChatPopup');
      if (!popup) return;
      if (popup.classList.contains('hidden') || popup.style.display === 'none') {
        popup.classList.remove('hidden');
        popup.style.display = 'flex';
      } else {
        popup.classList.add('hidden');
        popup.style.display = 'none';
      }
    },

    closeChatPopup: function () {
      const popup = document.getElementById('fbChatPopup');
      if (popup) {
        popup.classList.add('hidden');
        popup.style.display = 'none';
      }
    },

    openServerSettings: function () {
      let modal = document.getElementById('serverConfigModal');
      if (!modal) {
        modal = document.createElement('div');
        modal.id = 'serverConfigModal';
        modal.className = 'fixed inset-0 z-50 bg-black/75 backdrop-blur-sm flex items-center justify-center p-4 transition duration-200';
        modal.onclick = (e) => {
          if (e.target === modal || e.target.closest('#closeServerModalBtn')) {
            modal.classList.add('hidden');
          }
        };
        modal.innerHTML = `
          <div class="relative max-w-lg w-full bg-white rounded-3xl shadow-2xl p-6 flex flex-col overflow-hidden" onclick="event.stopPropagation()">
            <div class="flex items-center justify-between pb-3 mb-4 border-b border-slate-100">
              <h3 class="text-base font-bold text-slate-900 flex items-center gap-2">
                <i data-lucide="server" class="w-5 h-5 text-blue-600"></i> AI Backend Server Settings
              </h3>
              <button id="closeServerModalBtn" class="p-1.5 rounded-xl text-slate-400 hover:text-slate-700 hover:bg-slate-100 transition" onclick="document.getElementById('serverConfigModal').classList.add('hidden')">
                ✕
              </button>
            </div>
            
            <div class="space-y-4 text-xs text-slate-700 leading-relaxed">
              <div class="p-3.5 rounded-2xl bg-slate-50 border border-slate-200 flex items-center justify-between">
                <div>
                  <div class="font-bold text-slate-900 text-sm flex items-center gap-2">
                    <span class="backend-status-dot w-2.5 h-2.5 rounded-full ${isBackendOnline ? 'bg-emerald-500' : 'bg-amber-500'}"></span>
                    <span class="backend-status-label">${isBackendOnline ? 'Python Machine Learning Model (Online)' : 'Campus Knowledge Mode (Offline-Resilient)'}</span>
                  </div>
                  <div class="text-[11px] text-slate-500 mt-0.5">Active endpoint: <code class="text-blue-700 font-mono">${activeBackendUrl || 'None'}</code></div>
                </div>
              </div>

              <div>
                <label class="block font-bold text-slate-800 mb-1.5 text-xs">Custom Backend Server URL</label>
                <div class="flex gap-2">
                  <input type="text" id="customBackendInput" placeholder="http://10.152.113.39:5000 or https://your-app.onrender.com" class="flex-1 px-3.5 py-2.5 rounded-xl border border-slate-300 text-xs text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500 font-mono" value="${localStorage.getItem('brainware_backend_url') || ''}" />
                  <button type="button" onclick="BrainwareAITutor.testAndSaveServer()" class="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white font-bold rounded-xl text-xs transition shadow-sm">
                    Connect
                  </button>
                </div>
                <p id="serverTestFeedback" class="text-[11px] mt-1.5 font-medium"></p>
              </div>

              <div class="pt-2 border-t border-slate-100">
                <span class="font-bold text-slate-800 block mb-2 text-xs">Quick Presets for Other Devices:</span>
                <div class="grid grid-cols-1 sm:grid-cols-2 gap-2 text-[11px]">
                  <button type="button" onclick="BrainwareAITutor.applyServerPreset('http://10.152.113.39:5000')" class="p-2.5 rounded-xl bg-blue-50 hover:bg-blue-100 border border-blue-200 text-blue-900 text-left font-mono">
                    <strong class="block font-sans font-bold">Local Wi-Fi IP</strong>
                    http://10.152.113.39:5000
                  </button>
                  <button type="button" onclick="BrainwareAITutor.applyServerPreset('http://localhost:5000')" class="p-2.5 rounded-xl bg-slate-100 hover:bg-slate-200 border border-slate-200 text-slate-800 text-left font-mono">
                    <strong class="block font-sans font-bold">This Computer</strong>
                    http://localhost:5000
                  </button>
                </div>
                <p class="text-[11px] text-slate-500 mt-2">
                  To access from your mobile phone on the same Wi-Fi, select <strong>Local Wi-Fi IP</strong> and tap Connect.
                </p>
              </div>
            </div>
          </div>
        `;
        document.body.appendChild(modal);
      }
      modal.classList.remove('hidden');
      if (window.lucide && typeof window.lucide.createIcons === 'function') {
        window.lucide.createIcons();
      }
    },

    applyServerPreset: function (url) {
      const input = document.getElementById('customBackendInput');
      if (input) {
        input.value = url;
        this.testAndSaveServer();
      }
    },

    testAndSaveServer: async function () {
      const input = document.getElementById('customBackendInput');
      const feedback = document.getElementById('serverTestFeedback');
      if (!input || !feedback) return;

      const rawUrl = input.value.trim();
      if (!rawUrl) {
        localStorage.removeItem('brainware_backend_url');
        feedback.className = 'text-[11px] mt-1.5 text-slate-600';
        feedback.textContent = 'Cleared custom server URL. Using auto-discovery.';
        await resolveBackendEndpoint();
        return;
      }

      feedback.className = 'text-[11px] mt-1.5 text-blue-600 font-medium';
      feedback.textContent = 'Testing connection...';

      let testUrl = rawUrl.replace(/\/$/, '');
      if (!testUrl.endsWith('/api/health')) {
        testUrl = testUrl.replace(/\/api\/chat$/, '') + '/api/health';
      }

      try {
        const ctrl = new AbortController();
        const tid = setTimeout(() => ctrl.abort(), 3500);
        const res = await fetch(testUrl, { method: 'GET', signal: ctrl.signal });
        clearTimeout(tid);

        if (res.ok) {
          const chatUrl = rawUrl.replace(/\/$/, '').replace(/\/api\/health$/, '') + '/api/chat';
          localStorage.setItem('brainware_backend_url', chatUrl);
          activeBackendUrl = chatUrl;
          isBackendOnline = true;
          updateBackendStatusUI(true, chatUrl);
          feedback.className = 'text-[11px] mt-1.5 text-emerald-600 font-bold';
          feedback.textContent = '✓ Successfully connected to Python AI server!';
          setTimeout(() => {
            const modal = document.getElementById('serverConfigModal');
            if (modal) modal.classList.add('hidden');
          }, 1200);
        } else {
          throw new Error(`Server returned HTTP ${res.status}`);
        }
      } catch (err) {
        feedback.className = 'text-[11px] mt-1.5 text-rose-600 font-bold';
        feedback.textContent = `✕ Could not reach server: ${err.message || 'Connection Refused'}. Operating in Campus Mode.`;
      }
    },

    zoomImage: function (src, alt) {
      let modal = document.getElementById('examImageModal');
      if (!modal) {
        modal = document.createElement('div');
        modal.id = 'examImageModal';
        modal.className = 'fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4 transition duration-200';
        modal.onclick = (e) => {
          if (e.target === modal || e.target.closest('#closeImageModalBtn')) {
            modal.classList.add('hidden');
          }
        };
        modal.innerHTML = `
          <div class="relative max-w-4xl w-full max-h-[92vh] bg-white rounded-2xl shadow-2xl p-4 flex flex-col overflow-hidden" onclick="event.stopPropagation()">
            <div class="flex items-center justify-between pb-3 mb-2 border-b border-slate-100">
              <h4 id="examImageModalTitle" class="text-sm font-bold text-slate-800 truncate pr-4 flex items-center gap-2">
                <i data-lucide="image" class="w-4 h-4 text-blue-600"></i>
                <span id="examImageModalTitleText"></span>
              </h4>
              <div class="flex items-center gap-2">
                <a id="examImageDownloadBtn" href="" download="" target="_blank" class="px-3 py-1.5 text-xs font-semibold bg-blue-50 text-blue-700 rounded-lg hover:bg-blue-100 transition flex items-center gap-1.5">
                  <i data-lucide="download" class="w-3.5 h-3.5"></i> Download
                </a>
                <button id="closeImageModalBtn" class="p-1.5 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-100 transition" onclick="document.getElementById('examImageModal').classList.add('hidden')">
                  ✕
                </button>
              </div>
            </div>
            <div class="overflow-auto flex-1 flex items-center justify-center p-3 bg-slate-50/60 rounded-xl">
              <img id="examImageModalImg" src="" alt="" class="max-w-full max-h-[75vh] object-contain rounded-lg shadow-sm" />
            </div>
          </div>
        `;
        document.body.appendChild(modal);
      }
      const titleElem = document.getElementById('examImageModalTitleText');
      if (titleElem) titleElem.textContent = alt || 'Exam Diagram / Figure';
      const imgElem = document.getElementById('examImageModalImg');
      if (imgElem) {
        imgElem.src = src;
        imgElem.alt = alt || 'Exam Diagram';
      }
      const dl = document.getElementById('examImageDownloadBtn');
      if (dl) {
        dl.href = src;
        dl.download = src.split('/').pop() || 'exam_diagram.png';
      }
      modal.classList.remove('hidden');
      if (window.lucide && typeof window.lucide.createIcons === 'function') {
        window.lucide.createIcons();
      }
    }
  };

  document.addEventListener('DOMContentLoaded', () => {
    resolveBackendEndpoint();
    if (window.lucide && typeof window.lucide.createIcons === 'function') {
      window.lucide.createIcons();
    }
  });

})();
