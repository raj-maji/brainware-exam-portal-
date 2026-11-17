/**
 * Brainware Academic AI - Clean Frontend UI Controller
 * File: chat_ui.js
 * Version: 7.0.0 - Pure UI Client (Zero AI logic in JavaScript)
 * 
 * Responsibilities:
 * - DOM event handling (Input, Submit, File Attachment, Drag-and-Drop)
 * - Markdown & KaTeX LaTeX formatting
 * - Speech synthesis & Clipboard helpers
 * - Pure asynchronous API client: delegates 100% of intelligence to Python Machine Learning Backend (/api/chat)
 */

(function () {
  'use strict';

  // --- UI CLIENT STATE ---
  let isGenerating = false;
  let chatHistory = []; // { role: 'user' | 'assistant' | 'model', text: string }
  let attachedFileData = null;
  let activeBackendUrl = '/api/chat';

  // =========================================================================
  // 1. BACKEND DISCOVERY
  // =========================================================================
  async function resolveBackendEndpoint() {
    const candidates = ['/api/chat', 'http://localhost:5000/api/chat', 'http://127.0.0.1:5000/api/chat'];
    for (const cand of candidates) {
      try {
        const testUrl = cand.replace(/\/chat$/, '/health');
        const ctrl = new AbortController();
        const tid = setTimeout(() => ctrl.abort(), 1500);
        const res = await fetch(testUrl, { method: 'GET', signal: ctrl.signal });
        clearTimeout(tid);
        if (res.ok) {
          activeBackendUrl = cand;
          return activeBackendUrl;
        }
      } catch (e) {}
    }
    return activeBackendUrl;
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
  // 3. PURE HTTP CLIENT CALL TO PYTHON BACKEND (/api/chat)
  // =========================================================================
  async function callPythonAcademicBackend(query, history = [], fileSnapshot = null) {
    const endpoint = await resolveBackendEndpoint();
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
  // 4. MARKDOWN & KATEX LATEX RENDERING
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
        const cells = row.split('|').filter((_, i, arr) => i > 0 && i < arr.length - 1).map(c => c.trim());
        if (idx === 0) {
          tableHtml += '<thead class="bg-slate-100 text-slate-700 font-bold uppercase tracking-wider"><tr>';
          cells.forEach(c => tableHtml += `<th class="px-3 py-2 border-r border-slate-200 last:border-r-0">${c}</th>`);
          tableHtml += '</tr></thead><tbody class="divide-y divide-slate-200 bg-white">';
        } else {
          tableHtml += '<tr class="hover:bg-slate-50 transition">';
          cells.forEach(c => tableHtml += `<td class="px-3 py-2 border-r border-slate-200 last:border-r-0 text-slate-800">${c}</td>`);
          tableHtml += '</tr>';
        }
      });
      tableHtml += '</tbody></table></div>';
      return tableHtml;
    });

    // Unordered lists
    html = html.replace(/(?:^\s*[-*]\s+.*(?:\r?\n|$))+/gm, (listBlock) => {
      const items = listBlock.trim().split('\n').map(line => {
        const content = line.replace(/^\s*[-*]\s+/, '').trim();
        return `<li class="ml-4 list-disc text-slate-700 leading-relaxed mb-1">${content}</li>`;
      }).join('');
      return `<ul class="my-2.5 pl-2 space-y-1">${items}</ul>`;
    });

    // Ordered numbered lists
    html = html.replace(/(?:^\s*\d+\.\s+.*(?:\r?\n|$))+/gm, (listBlock) => {
      const items = listBlock.trim().split('\n').map(line => {
        const m = line.match(/^\s*(\d+)\.\s+(.*)$/);
        if (m) {
          return `<li class="ml-5 list-decimal text-slate-700 leading-relaxed mb-1"><span class="font-semibold text-slate-900">${m[1]}.</span> ${m[2]}</li>`;
        }
        return `<li class="ml-5 list-decimal text-slate-700 leading-relaxed mb-1">${line}</li>`;
      }).join('');
      return `<ol class="my-2.5 pl-2 space-y-1 list-none">${items}</ol>`;
    });

    html = html.replace(/\n\n+/g, '</p><p class="mb-2 leading-relaxed text-slate-800">');

    return `<p class="mb-2 leading-relaxed text-slate-800">${html}</p>`;
  }

  // =========================================================================
  // 5. DOM CHAT RENDERING
  // =========================================================================
  function appendMessageToUI(role, rawContent, source = null, model = null) {
    const thread = document.getElementById('aiChatThread');
    const popupThread = document.getElementById('fbChatMessages');

    const formattedContent = role === 'user' ? escapeHtml(rawContent) : renderMarkdown(rawContent);
    const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    let sourceBadge = '';
    if (source === 'google-gemini') {
      sourceBadge = `<span class="inline-flex items-center gap-1 text-[10px] text-blue-700 bg-blue-50 border border-blue-200 px-2 py-0.5 rounded-full font-medium ml-2"><i data-lucide="sparkles" class="w-3 h-3"></i> Google Gemini</span>`;
    } else if (source === 'openai-neural') {
      sourceBadge = `<span class="inline-flex items-center gap-1 text-[10px] text-emerald-700 bg-emerald-50 border border-emerald-200 px-2 py-0.5 rounded-full font-medium ml-2"><i data-lucide="zap" class="w-3 h-3"></i> OpenAI</span>`;
    } else if (source === 'python-ml-model' || source === 'machine-learning') {
      sourceBadge = `<span class="inline-flex items-center gap-1 text-[10px] text-violet-700 bg-violet-50 border border-violet-200 px-2 py-0.5 rounded-full font-medium ml-2"><i data-lucide="cpu" class="w-3 h-3"></i> Python ML</span>`;
    }

    if (thread) {
      const bubble = document.createElement('div');
      bubble.className = `flex gap-3 mb-4 animate-fade-in ${role === 'user' ? 'justify-end' : ''}`;
      
      if (role === 'user') {
        bubble.innerHTML = `
          <div class="max-w-2xl bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-2xl rounded-tr-sm p-3.5 shadow-md text-xs sm:text-sm font-medium leading-relaxed">
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
        <span class="font-medium text-slate-700">Synthesizing response with Python Machine Learning Model...</span>
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
        // Pure delegation to the Python backend Machine Learning model
        const response = await callPythonAcademicBackend(query, chatHistory, fileSnapshot);
        removeLoadingIndicator(isPopup);

        appendMessageToUI('model', response.text, response.source, response.model);
        chatHistory.push({ role: 'model', text: response.text });

      } catch (err) {
        console.error('AI Backend Error:', err);
        removeLoadingIndicator(isPopup);
        appendMessageToUI('model', `### System Notice: Python Backend Connection Error\n\nUnable to reach Python Machine Learning engine: **${err.message || 'Service Unavailable'}**.\n\nPlease ensure \`python server.py\` is actively running on port 5000.`);
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

    updateStatusBadge: function () {
      // Status badge updated from server health if present
    }
  };

  document.addEventListener('DOMContentLoaded', () => {
    resolveBackendEndpoint();
    if (window.lucide && typeof window.lucide.createIcons === 'function') {
      window.lucide.createIcons();
    }
  });

})();
