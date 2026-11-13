/**
 * Brainware Academic AI - Worldwide Universal Educational Assistant
 * Version: 2.1.0 - Global Curriculum, Multi-Disciplinary & Adaptive Level Edition
 * 
 * Architecture:
 * 1. Primary Engine: Backend API Proxy (/api/chat) -> Google Gemini 1.5 Flash
 * 2. Secondary Engine: Direct Client-Side Gemini 1.5 Flash API (when key is configured)
 * 3. Worldwide Educational Synthesizer: Autonomous Multi-Disciplinary Knowledge Engine
 * 4. Multi-Turn Conversation Memory: Context-preserving follow-up resolver
 * 5. Document Understanding: PDF.js Multi-Page, Word (Mammoth), Excel (SheetJS), PPT (JSZip), Multimodal Vision
 * 6. University Syllabus Grounding: portalQAIndex (1,254 entries) + papersData (14 exam sets)
 */

(function() {
  'use strict';

  // Initialize PDF.js worker
  if (typeof pdfjsLib !== 'undefined') {
    pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
  }

  // Engine State
  let isGenerating = false;
  let abortController = null;
  let chatHistory = []; // Array of { role: 'user' | 'assistant' | 'model', text: string }
  let lastUserQuery = '';
  let messageCounter = 0;
  let attachedFileData = null;
  let isSpeaking = false;

  // Backend API Endpoints
  const BACKEND_URLS = [
    '/api/chat',
    'http://localhost:5000/api/chat',
    'http://127.0.0.1:5000/api/chat'
  ];

  // Stopwords for search indexing
  const stopwords = new Set([
    'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'in', 'to', 'for', 
    'of', 'with', 'between', 'distinguish', 'difference', 'explain', 'describe', 
    'what', 'how', 'why', 'who', 'where', 'can', 'you', 'tell', 'me', 'about', 
    'state', 'compare', 'contrast', 'write', 'briefly', 'terms', 'define', 'give', 
    'list', 'following', 'does', 'do', 'are', 'by', 'its', 'from', 'discuss', 'note',
    'short', 'please', 'solve', 'calculate', 'find'
  ]);

  function sanitizeUserQuery(raw) {
    if (!raw) return '';
    return raw.trim().replace(/^(?:question\s*\d+|q\s*\d+|\d+)[\s.:\-)]+/i, '').trim();
  }

  function cleanSyllabusHtml(html) {
    if (!html) return '';
    let clean = html;
    clean = clean.replace(/<div class="[^"]*">/gi, '');
    clean = clean.replace(/<\/div>/gi, '\n');
    clean = clean.replace(/<p class="[^"]*">/gi, '');
    clean = clean.replace(/<\/p>/gi, '\n\n');
    clean = clean.replace(/<span class="[^"]*">/gi, '');
    clean = clean.replace(/<\/span>/gi, '');
    clean = clean.replace(/<br\s*[\/]?>/gi, '\n');
    clean = clean.replace(/&nbsp;/gi, ' ');
    clean = clean.replace(/&rarr;/gi, '→');
    clean = clean.replace(/&amp;/gi, '&');
    clean = clean.replace(/&lt;/gi, '<');
    clean = clean.replace(/&gt;/gi, '>');
    clean = clean.replace(/&quot;/gi, '"');
    clean = clean.replace(/&#039;/gi, "'");
    clean = clean.replace(/^(Ans|Answer):?/i, '');
    return clean.trim();
  }

  // =========================================================================
  // 1. MULTI-FORMAT DOCUMENT INGESTION & UNDERSTANDING
  // =========================================================================
  async function parseUploadedDocument(file) {
    const fileName = file.name || 'uploaded_document';
    const fileSize = (file.size / 1024).toFixed(1) + ' KB';
    const ext = fileName.split('.').pop().toLowerCase();
    let extractedText = '';
    let base64Data = null;
    const mimeType = file.type || '';

    try {
      // 1. PDF via PDF.js (Multi-page full text extraction)
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
            console.warn('PDF extraction warning:', pdfErr);
          }
        }
        if (!extractedText) {
          extractedText = `[PDF Document: ${fileName}, size: ${fileSize}]`;
        }

      // 2. Word Documents (.docx) via Mammoth
      } else if (ext === 'docx') {
        if (typeof mammoth !== 'undefined') {
          const arrayBuf = await file.arrayBuffer();
          const result = await mammoth.extractRawText({ arrayBuffer: arrayBuf });
          extractedText = (result.value || '').trim();
        } else {
          extractedText = `[Word Document: ${fileName}, size: ${fileSize}]`;
        }

      // 3. Excel Spreadsheets (.xlsx, .xls, .csv) via SheetJS
      } else if (ext === 'xlsx' || ext === 'xls' || ext === 'csv') {
        if (typeof XLSX !== 'undefined') {
          const arrayBuf = await file.arrayBuffer();
          const workbook = XLSX.read(arrayBuf, { type: 'array' });
          const sheetsOutput = [];
          workbook.SheetNames.forEach(sheetName => {
            const worksheet = workbook.Sheets[sheetName];
            const csv = XLSX.utils.sheet_to_csv(worksheet);
            if (csv && csv.trim()) {
              sheetsOutput.push(`--- Worksheet: ${sheetName} ---\n` + csv.trim());
            }
          });
          extractedText = sheetsOutput.join('\n\n');
        } else {
          extractedText = await file.text();
        }

      // 4. PowerPoint (.pptx) via JSZip
      } else if (ext === 'pptx') {
        if (typeof JSZip !== 'undefined') {
          const arrayBuf = await file.arrayBuffer();
          const zip = await JSZip.loadAsync(arrayBuf);
          const slideFiles = Object.keys(zip.files).filter(p => p.startsWith('ppt/slides/slide') && p.endsWith('.xml'));
          slideFiles.sort((a, b) => {
            const numA = parseInt((a.match(/slide(\d+)\.xml/) || [0, 0])[1]);
            const numB = parseInt((b.match(/slide(\d+)\.xml/) || [0, 0])[1]);
            return numA - numB;
          });

          const slidesOutput = [];
          for (let i = 0; i < slideFiles.length; i++) {
            const xml = await zip.files[slideFiles[i]].async('text');
            const textMatches = [];
            const regex = /<a:t[^>]*>([\s\S]*?)<\/a:t>/g;
            let m;
            while ((m = regex.exec(xml)) !== null) {
              if (m[1].trim()) textMatches.push(m[1].trim());
            }
            if (textMatches.length) {
              slidesOutput.push(`[Slide ${i + 1}]: ` + textMatches.join(' '));
            }
          }
          extractedText = slidesOutput.join('\n\n');
        } else {
          extractedText = `[PowerPoint: ${fileName}, size: ${fileSize}]`;
        }

      // 5. Images (.png, .jpg, .jpeg, .webp) for Gemini Multimodal Vision
      } else if (['png', 'jpg', 'jpeg', 'webp', 'gif', 'bmp'].includes(ext)) {
        base64Data = await new Promise((resolve) => {
          const reader = new FileReader();
          reader.onload = () => resolve(reader.result);
          reader.onerror = () => resolve(null);
          reader.readAsDataURL(file);
        });
        extractedText = `[Attached Image: ${fileName}, MIME: ${file.type || 'image/jpeg'}, size: ${fileSize}]`;

      // 6. Source Code & Plain Text Files
      } else if (['txt', 'md', 'json', 'py', 'java', 'c', 'cpp', 'h', 'hpp', 'sql', 'html', 'css', 'js', 'ts', 'xml', 'log'].includes(ext)) {
        extractedText = await file.text();

      // 7. General Fallback
      } else {
        try {
          extractedText = await file.text();
        } catch (e) {
          extractedText = `[Document: ${fileName}, size: ${fileSize}]`;
        }
      }
    } catch (err) {
      console.warn('Document extraction notice:', err);
      extractedText = `[Document: ${fileName}, size: ${fileSize}]`;
    }

    if (!extractedText || !extractedText.trim()) {
      extractedText = `[Document: ${fileName}, type: ${ext.toUpperCase()}, size: ${fileSize}]`;
    }

    return {
      name: fileName,
      size: fileSize,
      type: ext.toUpperCase(),
      mimeType: mimeType || 'application/octet-stream',
      text: extractedText.trim(),
      base64: base64Data
    };
  }

  // =========================================================================
  // 2. UNIVERSITY QUESTION BANK GROUNDING (portalQAIndex & papersData)
  // =========================================================================
  function findSyllabusMatch(rawQuery) {
    const query = sanitizeUserQuery(rawQuery);
    const qNorm = query.toLowerCase().trim();
    if (!qNorm || qNorm.length < 4) return null;

    const tokens = qNorm.match(/[a-z0-9_.-]+/g) || [];
    const significantWords = tokens.filter(w => !stopwords.has(w) && w.length > 2);
    if (significantWords.length === 0) return null;

    let bestMatch = null;
    let highestScore = 0;

    // Search in portalQAIndex (1,254 entries)
    if (window.portalQAIndex && Array.isArray(window.portalQAIndex)) {
      for (const item of window.portalQAIndex) {
        const candNorm = (item.q || '').toLowerCase();
        let score = 0;

        if (candNorm === qNorm) {
          score += 450;
        } else if (candNorm.includes(qNorm) || (qNorm.length > 15 && qNorm.includes(candNorm))) {
          score += 280;
        }

        let matchedCount = 0;
        for (const w of significantWords) {
          if (candNorm.includes(w)) matchedCount++;
        }

        const overlap = matchedCount / significantWords.length;
        if ((matchedCount >= 2 && overlap >= 0.5) || (significantWords.length === 1 && matchedCount === 1 && candNorm.length < 40)) {
          score += (matchedCount * 50) + Math.round(overlap * 100);
          if (score > highestScore && score >= 180) {
            highestScore = score;
            bestMatch = { item: item, score: score, source: 'Brainware University Syllabus QA Index' };
          }
        }
      }
    }

    // Search in papersData (14 Exam Sets)
    if (window.papersData && Array.isArray(window.papersData)) {
      for (const paper of window.papersData) {
        const allQuestions = [
          ...(paper.group_a || []).map(q => ({ q: q.q, a: `**Correct Option: ${q.ans}**\n\n*Rationale & Solution:* ${q.expl}`, type: 'MCQ', paper: paper.title })),
          ...(paper.group_b || []).map(q => ({ q: q.q, a: `**Solution Outline:**\n\n${q.hint}`, type: 'Short Answer', paper: paper.title })),
          ...(paper.group_c || []).map(q => ({ q: q.q, a: `**Comprehensive Solution Framework:**\n\n${q.hint}`, type: 'Descriptive', paper: paper.title }))
        ];

        for (const item of allQuestions) {
          const candNorm = (item.q || '').toLowerCase();
          let score = 0;

          if (candNorm === qNorm) {
            score += 400;
          } else if (candNorm.includes(qNorm) || (qNorm.length > 15 && qNorm.includes(candNorm))) {
            score += 260;
          }

          let matchedCount = 0;
          for (const w of significantWords) {
            if (candNorm.includes(w)) matchedCount++;
          }

          const overlap = matchedCount / significantWords.length;
          if ((matchedCount >= 2 && overlap >= 0.5) || (significantWords.length === 1 && matchedCount === 1 && candNorm.length < 40)) {
            score += (matchedCount * 45) + Math.round(overlap * 90);
            if (score > highestScore && score >= 180) {
              highestScore = score;
              bestMatch = { item: item, score: score, source: `Official University Exam Paper (${item.paper || 'Syllabus'})` };
            }
          }
        }
      }
    }

    return bestMatch;
  }

  // =========================================================================
  // 3. MATHEMATICAL & PROBABILITY CALCULATION ENGINE
  // =========================================================================
  function trySolveMathProblem(query) {
    const q = query.toLowerCase().replace(/\s+/g, ' ').trim();

    // 1. Coin Toss Probability: "probability of getting two heads in two coin tosses"
    if (q.includes('probability') && q.includes('coin') && (q.includes('two head') || q.includes('2 head'))) {
      return `### Probability: Two Heads in Two Coin Tosses

#### 1. Sample Space ($S$)
When a fair coin is tossed twice, each toss has two equally likely independent outcomes: $\\text{Head } (H)$ or $\\text{Tail } (T)$.

The complete sample space $S$ is:

$$S = \\{HH, HT, TH, TT\\}$$
$$n(S) = 2 \\times 2 = 4$$

---

#### 2. Favorable Event ($E$)
The event $E$ of obtaining **two heads** corresponds to the single outcome:

$$E = \\{HH\\} \\implies n(E) = 1$$

---

#### 3. Probability Calculation
By the classical definition of probability:

$$P(E) = \\frac{n(E)}{n(S)} = \\frac{1}{4} = 0.25 = 25\\%$$

Alternatively, using the rule of independent events where $P(H) = \\frac{1}{2}$:

$$P(HH) = P(H_1) \\times P(H_2) = \\frac{1}{2} \\times \\frac{1}{2} = \\frac{1}{4}$$

**Final Answer:**
$$\\mathbf{P(\\text{Two Heads}) = \\frac{1}{4} = 0.25 \\quad (25\\%)}$$`;
    }

    // 2. Quadratic Equation: "solve a quadratic equation" or "quadratic equation formula"
    if (q.includes('quadratic equation') || q.includes('solve quadratic')) {
      return `### Quadratic Equation: Formulation & General Solution

A **Quadratic Equation** is a second-order polynomial equation in a single variable $x$ with the standard form:

$$ax^2 + bx + c = 0 \\quad (a \\neq 0)$$

---

### The Quadratic Formula

The general analytical solutions for $x$ are given by:

$$x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$$

---

### Step-by-Step Derivation (Completing the Square)
1. **Divide by $a$:**
   $$x^2 + \\frac{b}{a}x + \\frac{c}{a} = 0 \\implies x^2 + \\frac{b}{a}x = -\\frac{c}{a}$$
2. **Add $\\left(\\frac{b}{2a}\\right)^2$ to both sides:**
   $$x^2 + 2\\left(\\frac{b}{2a}\\right)x + \\frac{b^2}{4a^2} = \\frac{b^2}{4a^2} - \\frac{c}{a}$$
3. **Factor the left-hand side into a perfect square:**
   $$\\left(x + \\frac{b}{2a}\\right)^2 = \\frac{b^2 - 4ac}{4a^2}$$
4. **Take square roots on both sides:**
   $$x + \\frac{b}{2a} = \\pm \\frac{\\sqrt{b^2 - 4ac}}{2a} \\implies x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$$

---

### Discriminant Analysis ($\\Delta = b^2 - 4ac$)
- **$\\Delta > 0$:** Two distinct real roots ($x_1 \\neq x_2 \\in \\mathbb{R}$).
- **$\\Delta = 0$:** Exactly one repeated real root ($x = -\\frac{b}{2a}$).
- **$\\Delta < 0$:** Two complex conjugate roots ($x = -\\frac{b}{2a} \\pm i\\frac{\\sqrt{4ac - b^2}}{2a}$).`;
    }

    // 3. Differentiation: "explain differentiation"
    if (q.includes('differentiation') || q.includes('derivative') && (q.includes('explain') || q.includes('what is'))) {
      return `### Differentiation in Calculus

**Differentiation** is a core mathematical process in calculus that measures the instantaneous rate of change of a function $f(x)$ with respect to its independent variable $x$. Geometrically, it represents the exact **slope of the tangent line** to the curve at any point.

---

### Formal Definition (First Principles / Limit Definition)

The derivative $f'(x)$ or $\\frac{df}{dx}$ is formally defined as:

$$f'(x) = \\lim_{h \\to 0} \\frac{f(x + h) - f(x)}{h}$$

---

### Fundamental Rules of Differentiation

| Rule Name | Mathematical Formulation |
| :--- | :--- |
| **Power Rule** | $\\frac{d}{dx}[x^n] = n x^{n-1}$ |
| **Constant Multiple Rule** | $\\frac{d}{dx}[c \\cdot f(x)] = c \\cdot f'(x)$ |
| **Sum / Difference Rule** | $\\frac{d}{dx}[f(x) \\pm g(x)] = f'(x) \\pm g'(x)$ |
| **Product Rule** | $\\frac{d}{dx}[f(x) \\cdot g(x)] = f'(x)g(x) + f(x)g'(x)$ |
| **Quotient Rule** | $\\frac{d}{dx}\\left[\\frac{f(x)}{g(x)}\\right] = \\frac{f'(x)g(x) - f(x)g'(x)}{[g(x)]^2}$ |
| **Chain Rule** | $\\frac{d}{dx}[f(g(x))] = f'(g(x)) \\cdot g'(x)$ |

---

### Concrete Example
Differentiate $f(x) = 3x^4 - 5x^2 + 7$:

$$\\frac{df}{dx} = 3(4x^3) - 5(2x) + 0 = 12x^3 - 10x$$`;
    }

    // 4. Linear Equation: e.g. "solve 2x + 5 = 15"
    const linearMatch = q.match(/(?:solve\s+)?([+-]?\s*\d*\.?\d*)\s*([a-z])\s*([+-])\s*(\d+\.?\d*)\s*=\s*([+-]?\s*\d+\.?\d*)/i);
    if (linearMatch) {
      const aStr = linearMatch[1].replace(/\s+/g, '');
      const varName = linearMatch[2];
      const sign = linearMatch[3];
      const bVal = parseFloat(linearMatch[4]);
      const cVal = parseFloat(linearMatch[5].replace(/\s+/g, ''));

      let aVal = 1;
      if (aStr === '' || aStr === '+') aVal = 1;
      else if (aStr === '-') aVal = -1;
      else aVal = parseFloat(aStr);

      if (!isNaN(aVal) && aVal !== 0 && !isNaN(bVal) && !isNaN(cVal)) {
        const signedB = sign === '-' ? -bVal : bVal;
        const rhsAfterSub = cVal - signedB;
        const solution = rhsAfterSub / aVal;
        const solRounded = Number.isInteger(solution) ? solution : solution.toFixed(4);

        return `### Step-by-Step Mathematical Solution

Given the linear algebraic equation:

$$${aVal !== 1 ? aVal : ''}${varName} ${signedB >= 0 ? '+ ' + signedB : '- ' + Math.abs(signedB)} = ${cVal}$$

---

### Step 1: Isolate the Variable Term
Subtract $${signedB}$ from both sides of the equation:

$$${aVal !== 1 ? aVal : ''}${varName} = ${cVal} - (${signedB})$$
$$${aVal !== 1 ? aVal : ''}${varName} = ${rhsAfterSub}$$

---

### Step 2: Solve for $${varName}$
Divide both sides by the coefficient of $${varName}$ ($${aVal}$):

$$${varName} = \\frac{${rhsAfterSub}}{${aVal}} = ${solRounded}$$

---

### Step 3: Verification
Substitute $${varName} = ${solRounded}$ back into the original equation:

$$${aVal}(${solRounded}) ${signedB >= 0 ? '+ ' + signedB : '- ' + Math.abs(signedB)} = ${aVal * solution + signedB} = ${cVal} \\quad \\text{(LHS = RHS Verified)}$$

**Final Answer:**
$$\\mathbf{${varName} = ${solRounded}}$$`;
      }
    }

    return null;
  }

  // =========================================================================
  // 4. PROGRAMMING & SYSTEM IMPLEMENTATION SYNTHESIZER
  // =========================================================================
  function tryGenerateCodeSolution(query) {
    const q = query.toLowerCase();

    // 1. JavaScript String Reversal
    if ((q.includes('reverse') && q.includes('string')) && (q.includes('javascript') || q.includes('js'))) {
      return `### JavaScript Program to Reverse a String

\`\`\`javascript
/**
 * Method 1: Idiomatic standard library approach
 * Time Complexity: O(n), Space Complexity: O(n)
 */
function reverseString(str) {
  if (typeof str !== 'string') return '';
  return str.split('').reverse().join('');
}

/**
 * Method 2: Two-pointer in-place array swap algorithm
 * Time Complexity: O(n), Space Complexity: O(n)
 */
function reverseStringTwoPointer(str) {
  const chars = Array.from(str);
  let left = 0, right = chars.length - 1;
  while (left < right) {
    const temp = chars[left];
    chars[left] = chars[right];
    chars[right] = temp;
    left++;
    right--;
  }
  return chars.join('');
}

// Example Execution
const sample = "Brainware University";
console.log("Original:", sample);
console.log("Reversed:", reverseString(sample));
// Output: "ytisrevinU erawniarB"
\`\`\`

---

### Complexity Analysis
- **Time Complexity:** $\\mathcal{O}(n)$ where $n$ is string length.
- **Space Complexity:** $\\mathcal{O}(n)$ for character buffer allocation.`;
    }

    // 2. Binary Search
    if (q.includes('binary search')) {
      return `### Binary Search Algorithm & Implementation

**Binary Search** is an optimal divide-and-conquer search algorithm designed for **strictly sorted arrays**. It compares the target with the middle element and halves the active search interval at each step.

---

### Python Implementation

\`\`\`python
def binary_search(arr, target):
    """
    Performs iterative binary search on a sorted list.
    Returns index of target if found, else -1.
    Time Complexity: O(log n), Space Complexity: O(1)
    """
    low = 0
    high = len(arr) - 1

    while low <= high:
        # Avoid potential integer overflow in fixed-width languages
        mid = low + (high - low) // 2

        if arr[mid] == target:
            return mid  # Target element located
        elif arr[mid] < target:
            low = mid + 1  # Target lies in right half
        else:
            high = mid - 1  # Target lies in left half

    return -1  # Target not found

# Demonstration
numbers = [2, 5, 8, 12, 16, 23, 38, 56, 72, 91]
target_val = 23
result = binary_search(numbers, target_val)

print(f"Target {target_val} located at index: {result}")
\`\`\`

---

### Complexity Analysis
- **Best Case Time:** $\\mathcal{O}(1)$ (element found at middle on first check).
- **Average & Worst Case Time:** $\\mathcal{O}(\\log_2 n)$ (search space is halved repeatedly: $n \\rightarrow \\frac{n}{2} \\dots \\rightarrow 1$).
- **Space Complexity:** $\\mathcal{O}(1)$ iterative auxiliary space.`;
    }

    return null;
  }

  // =========================================================================
  // 5. WORLDWIDE UNIVERSAL MULTI-DISCIPLINARY KNOWLEDGE ENGINE
  // =========================================================================
  function trySubjectConceptExplanation(query, history = []) {
    const q = query.toLowerCase().trim();

    // Contextual Follow-up: "What are its main advantages?" / "What are its advantages?"
    if (q.includes('advantage') || q.includes('disadvantage') || q.includes('pros') || q.includes('cons')) {
      let lastTopic = '';
      for (let i = history.length - 1; i >= 0; i--) {
        const text = (history[i].text || '').toLowerCase();
        if (text.includes('java')) { lastTopic = 'Java'; break; }
        if (text.includes('python')) { lastTopic = 'Python'; break; }
        if (text.includes('c++')) { lastTopic = 'C++'; break; }
        if (text.includes('operating system')) { lastTopic = 'Operating Systems'; break; }
        if (text.includes('dbms') || text.includes('database')) { lastTopic = 'DBMS'; break; }
        if (text.includes('cloud')) { lastTopic = 'Cloud Computing'; break; }
        if (text.includes('transformer')) { lastTopic = 'Transformers'; break; }
      }

      if (lastTopic === 'Java') {
        return `### Main Advantages of Java

1. **Platform Independence (Write Once, Run Anywhere - WORA):**
   - Source code compiles into platform-neutral Bytecode (\`.class\`) executed by the Java Virtual Machine (JVM).
2. **Object-Oriented Architecture:**
   - Promotes robust modularity, encapsulation, and code reuse through clean inheritance and polymorphism.
3. **Automatic Memory Management:**
   - Integrated Garbage Collection (GC) automatically frees unreferenced memory, preventing manual pointer leaks.
4. **Rich Standard Library (JDK):**
   - Built-in APIs for multithreading, enterprise networking, collections framework, and cryptography.
5. **Robust Security & Type Safety:**
   - Sandboxed execution, bytecode verification, and absence of explicit pointer manipulation protect system integrity.`;
      }
    }

    // --- A. COMPUTER SCIENCE & IT ---

    // What is Java?
    if (q === 'what is java' || q === 'what is java?' || q.startsWith('what is java')) {
      return `### Java Programming Language

**Java** is a high-level, class-based, object-oriented programming language designed to have as few implementation dependencies as possible. Developed by James Gosling at Sun Microsystems and released in 1995, it is now maintained by Oracle Corporation.

---

### Core Architectural Features
1. **Platform Independence (WORA):**
   - Java source files (\`.java\`) compile into intermediate **Bytecode** (\`.class\`) rather than machine-specific binary.
   - The **Java Virtual Machine (JVM)** interprets or Just-In-Time (JIT) compiles this bytecode into native machine instructions on any target OS.
2. **Four OOP Pillars:**
   - **Encapsulation:** Protecting internal state with private variables and public accessors.
   - **Inheritance:** Enabling child classes to inherit attributes and methods from parent classes.
   - **Polymorphism:** Method overloading and dynamic runtime overriding.
   - **Abstraction:** Hiding complex underlying implementations using interfaces and abstract classes.
3. **Automatic Garbage Collection:**
   - Periodically reclaims heap memory allocated to unreferenced objects.

---

### Industry Applications
- Enterprise Backend Microservices (Spring Boot)
- Android Native Mobile Development
- Distributed Big Data Systems (Apache Spark, Kafka)
- High-Throughput Financial Systems`;
    }

    // Explain Operating Systems
    if (q.includes('operating system') && (q.includes('explain') || q.includes('what is') || q.includes('define'))) {
      return `### Operating System (OS) in Computer Science

An **Operating System (OS)** is fundamental system software that manages computer hardware and software resources, providing a standardized abstraction interface and common services for application programs.

---

### Core Management Subsystems
1. **Process Management:**
   - Allocates CPU time to competing threads via scheduling algorithms (Round Robin, Priority, Shortest Job First).
   - Coordinates Inter-Process Communication (IPC) and synchronization mechanisms (Semaphores, Mutexes).
2. **Memory Management:**
   - Tracks physical RAM and abstracts it into continuous **Virtual Memory**.
   - Handles address translation through Page Tables and Translation Lookaside Buffers (TLB), swapping pages to secondary storage via paging.
3. **File System Management:**
   - Organizes persistent data into logical directories and files (NTFS, ext4, APFS), managing disk sector allocation, caching, and access permissions.
4. **Device / I/O Management:**
   - Coordinates peripheral hardware devices (keyboards, storage controllers, network cards) using device drivers and interrupt request (IRQ) handlers.
5. **Security & Protection:**
   - Enforces user authentication, memory segmentation, and role-based access control (RBAC).`;
    }

    // What is Machine Learning?
    if (q.includes('machine learning') && (q.includes('what is') || q.includes('define') || q.includes('explain'))) {
      return `### Machine Learning (Artificial Intelligence)

**Machine Learning (ML)** is a subdiscipline of Artificial Intelligence (AI) and statistical computing that develops mathematical algorithms capable of learning patterns and decision rules from empirical training data to make predictions or decisions on unseen data without being explicitly programmed.

---

### The Three Core Paradigms

1. **Supervised Learning:**
   - Trains on labeled datasets $\{(x_i, y_i)\}$.
   - **Tasks:** Classification (discrete categories) and Regression (continuous numerical values).
   - **Algorithms:** Linear/Logistic Regression, Support Vector Machines (SVM), Random Forests, Deep Neural Networks.

2. **Unsupervised Learning:**
   - Analyzes unlabeled data $\{x_i\}$ to discover intrinsic groupings, clusters, or latent structures.
   - **Tasks:** Clustering, Dimensionality Reduction, Anomaly Detection.
   - **Algorithms:** K-Means, DBSCAN, Principal Component Analysis (PCA).

3. **Reinforcement Learning (RL):**
   - An autonomous agent learns an optimal behavioral policy $\\pi(a|s)$ through environmental interaction, optimizing cumulative scalar rewards.
   - **Algorithms:** Q-Learning, Deep Q-Networks (DQN), Proximal Policy Optimization (PPO).

---

### Mathematical Pipeline
$$\\text{Data Input} \\rightarrow \\text{Feature Extraction} \\rightarrow \\min_\\theta \\mathcal{L}(\\theta) \\text{ (Loss Optimization via Gradient Descent)} \\rightarrow \\text{Evaluation / Inference}$$`;
    }

    // --- B. NATURAL SCIENCES ---

    // Newton's Laws of Motion
    if (q.includes('newton') && (q.includes('law') || q.includes('motion'))) {
      return `### Newton's Three Laws of Motion (Classical Mechanics)

Formulated by Sir Isaac Newton in *Philosophiæ Naturalis Principia Mathematica* (1687), these three laws form the foundation of classical dynamics.

---

### 1. First Law of Motion (Law of Inertia)
- **Statement:** A body remains at rest or in uniform motion in a straight line unless acted upon by a non-zero net external force.
- **Equation:** If $\\sum \\vec{F} = 0$, then $\\vec{v} = \\text{constant} \\implies \\vec{a} = 0$.

---

### 2. Second Law of Motion (Law of Force & Momentum)
- **Statement:** The rate of change of momentum of an object is directly proportional to the applied net force and occurs in the direction of the force.
- **Derivation:**
  $$\\vec{F}_{\\text{net}} = \\frac{d\\vec{p}}{dt} = \\frac{d(m\\vec{v})}{dt} = m \\cdot \\frac{d\\vec{v}}{dt} = m\\vec{a}$$
- **Units:** Measured in Newtons ($\\text{N} = \\text{kg}\\cdot\\text{m}/\\text{s}^2$).

---

### 3. Third Law of Motion (Action and Reaction)
- **Statement:** For every action, there is an equal and opposite reaction.
- **Equation:**
  $$\\vec{F}_{AB} = -\\vec{F}_{BA}$$
  *(The force exerted by object A on B is equal in magnitude and opposite in direction to the force exerted by B on A; they act on two distinct bodies).*`;
    }

    // Electromagnetic Induction
    if (q.includes('electromagnetic induction') || q.includes('faraday') && q.includes('induction')) {
      return `### Electromagnetic Induction (Physics & Electromagnetism)

**Electromagnetic Induction** is the phenomenon whereby an electromotive force (EMF / voltage) is induced across an electrical conductor placed in a time-varying magnetic field or moving through a magnetic field.

---

### 1. Faraday's Law of Induction
The magnitude of the induced electromotive force $\\mathcal{E}$ is directly proportional to the time rate of change of magnetic flux $\\Phi_B$ through the circuit:

$$\\mathcal{E} = -N \\frac{d\\Phi_B}{dt}$$

Where:
- $\\mathcal{E}$: Induced EMF (Volts, $\\text{V}$).
- $N$: Number of wire turns in the coil.
- $\\Phi_B = \\int \\vec{B} \\cdot d\\vec{A} = B A \\cos\\theta$: Magnetic flux (Webers, $\\text{Wb}$).

---

### 2. Lenz's Law (Significance of the Negative Sign)
- **Statement:** The polarity of the induced EMF generates a current whose own magnetic field **opposes** the change in magnetic flux that produced it.
- **Physical Invariant:** Direct manifestation of the **Conservation of Energy**.

---

### Key Applications
- Electric Generators and Alternators
- Power Transformers
- Induction Motors and Magnetic Levitation (Maglev)`;
    }

    // Periodic Table
    if (q.includes('periodic table') && (q.includes('explain') || q.includes('what is'))) {
      return `### The Periodic Table of Elements (Chemistry)

The **Modern Periodic Table** is a systematic tabular arrangement of all 118 known chemical elements, organized in order of increasing **Atomic Number ($Z$)** and recurring chemical and physical properties (Periodic Law).

---

### Structural Organization

1. **Periods (Horizontal Rows - 7 Total):**
   - The period number corresponds to the **highest principal quantum number ($n$)** of electrons occupied in an element's ground state.
2. **Groups / Families (Vertical Columns - 18 Total):**
   - Elements in the same group possess identical **valence electron configurations**, conferring similar chemical reactivity.
   - **Group 1:** Alkali Metals (highly reactive, $+1$ oxidation state).
   - **Group 2:** Alkaline Earth Metals ($+2$).
   - **Group 17:** Halogens (highly electronegative non-metals, $-1$).
   - **Group 18:** Noble Gases (complete octet/duplet, inert).

---

### Key Periodic Trends
- **Atomic Radius:** Decreases across a period (left to right) due to increasing effective nuclear charge ($Z_{\\text{eff}}$); increases down a group.
- **Ionization Energy:** Energy required to remove an electron. Increases across a period; decreases down a group.
- **Electronegativity:** Tendency to attract shared electron pairs. Fluorine ($F$) is highest ($4.0$ on Pauling scale).`;
    }

    // Covalent Bond
    if (q.includes('covalent bond') && (q.includes('explain') || q.includes('what is') || q.includes('define'))) {
      return `### Covalent Bonding in Chemistry

A **Covalent Bond** is a fundamental chemical bond formed by the mutual sharing of one or more pairs of valence electrons between atoms, typically occurring between non-metallic elements with similar electronegativities.

---

### Classification by Shared Pairs
1. **Single Covalent Bond ($\\sigma$ bond):**
   - One electron pair shared (e.g., $\\text{H}_2$: $\\text{H}-\\text{H}$, $\\text{CH}_4$).
2. **Double Covalent Bond (one $\\sigma$ and one $\\pi$ bond):**
   - Two electron pairs shared (e.g., $\\text{O}_2$: $\\text{O}=\\text{O}$, $\\text{CO}_2$).
3. **Triple Covalent Bond (one $\\sigma$ and two $\\pi$ bonds):**
   - Three electron pairs shared (e.g., $\\text{N}_2$: $\\text{N}\\equiv\\text{N}$, $\\text{C}_2\\text{H}_2$).

---

### Polarity of Covalent Bonds
- **Non-Polar Covalent Bond:** Electronegativity difference $\\Delta\\text{EN} < 0.4$. Electrons shared equally (e.g., $\\text{Cl}_2$, $\\text{CH}_4$).
- **Polar Covalent Bond:** $0.4 \\le \\Delta\\text{EN} \\le 1.7$. Unequal electron distribution creates partial dipole charges $\\delta^+$ and $\\delta^-$ (e.g., $\\text{H}_2\\text{O}$, $\\text{HCl}$).

---

### Key Properties of Covalent Compounds
- Lower melting and boiling points compared to ionic salts.
- Poor electrical conductivity in both solid and liquid states (no free ions).
- Directional bonding giving rise to defined molecular geometries (VSEPR theory).`;
    }

    // Photosynthesis
    if (q.includes('photosynthesis')) {
      return `### Photosynthesis in Biological Sciences

**Photosynthesis** is the vital biochemical process by which photoautotrophic organisms (green plants, algae, and cyanobacteria) convert solar light energy into stable chemical energy stored in glucose molecules.

---

### Overall Chemical Equation

$$6\\text{CO}_2 + 6\\text{H}_2\\text{O} \\xrightarrow{\\text{Light Energy, Chlorophyll}} \\text{C}_6\\text{H}_{12}\\text{O}_6 + 6\\text{O}_2$$

---

### Two Interconnected Stages

1. **Light-Dependent Reactions (Thylakoid Membrane):**
   - Photons excite electrons in Chlorophyll within Photosystems II and I.
   - **Photolysis of Water:** $2\\text{H}_2\\text{O} \\rightarrow 4\\text{H}^+ + 4e^- + \\text{O}_2$.
   - Generates energy-rich carrier molecules: **ATP** and **NADPH**.

2. **Light-Independent Reactions / Calvin Cycle (Stroma):**
   - **Carbon Fixation:** The enzyme **RuBisCO** fixes atmospheric $\\text{CO}_2$ with ribulose-1,5-bisphosphate (RuBP).
   - Uses ATP and NADPH to synthesize glyceraldehyde 3-phosphate (G3P), which condenses into **Glucose** ($\\text{C}_6\\text{H}_{12}\\text{O}_6$).`;
    }

    // Human Digestive System
    if (q.includes('digestive system') || q.includes('human digestion')) {
      return `### The Human Digestive System (Anatomy & Physiology)

The **Human Digestive System** is the organ system responsible for mechanical breakdown, enzymatic hydrolysis, nutrient absorption, and waste elimination of ingested food.

---

### Sequential Anatomical Pathway & Mechanics

1. **Mouth & Oral Cavity:**
   - Teeth masticate food into a bolus; salivary glands secrete **salivary amylase**, initiating starch breakdown into maltose.
2. **Esophagus:**
   - Muscular tube moving food downward through rhythmic involuntary muscular contractions called **peristalsis**.
3. **Stomach:**
   - Secretes gastric juice containing **Hydrochloric Acid (HCl, pH ~1.5 - 2.0)** to sterilize pathogens and activate **pepsinogen** into **pepsin** for protein cleavage. Churns food into acidic **chyme**.
4. **Small Intestine (Duodenum, Jejunum, Ileum):**
   - **Primary Digestion & Absorption Hub.**
   - Liver supplies **bile** to emulsify lipids; pancreas provides lipase, trypsin, and amylase.
   - Millions of microscopic **villi** and **microvilli** maximize surface area to absorb amino acids, glucose, and fatty acids into capillaries and lacteals.
5. **Large Intestine (Colon) & Rectum:**
   - Reabsorbs water and electrolytes, housing gut microbiota synthesizing Vitamin K before compaction into feces.`;
    }

    // --- C. HUMANITIES, GEOGRAPHY & HISTORY ---

    // French Revolution
    if (q.includes('french revolution')) {
      return `### The French Revolution (1789–1799)

The **French Revolution** was a watershed epoch in European and global history that dismantled the absolute monarchy (*Ancien Régime*), abolished feudal privileges, and established modern republican and democratic principles.

---

### Major Causes
1. **The Three Estates:** Unfair social hierarchy where the First Estate (Clergy) and Second Estate (Nobility) held land and tax exemptions, while the Third Estate (Peasants and Bourgeoisie, 98% of population) carried the tax burden.
2. **Economic Collapse:** National bankruptcy following royal expenditures and costly foreign wars (Seven Years' War, American Revolution).
3. **Enlightenment Philosophy:** Ideas of Rousseau, Voltaire, and Montesquieu advocating liberty, equality, and the separation of powers.

---

### Chronological Milestones
- **1789:** Estates-General convened; Tennis Court Oath; **Storming of the Bastille** (July 14); passage of the *Declaration of the Rights of Man and of the Citizen*.
- **1792–1794:** Abolition of monarchy; execution of King Louis XVI; the **Reign of Terror** under Maximilien Robespierre.
- **1799:** Coup of 18 Brumaire by **Napoleon Bonaparte**, concluding the revolutionary era and establishing the French Consulate.`;
    }

    // Causes of World War II
    if (q.includes('world war ii') || q.includes('world war 2') || (q.includes('wwii') && q.includes('cause'))) {
      return `### Primary Causes of World War II (1939–1945)

World War II was the deadliest conflict in human history. Its outbreak resulted from the convergence of several geopolitical failures following World War I:

1. **The Treaty of Versailles (1919):**
   - Imposed crippling war reparations, territorial losses, and demilitarization on Germany, breeding widespread economic hardship and revanchist nationalism.
2. **Rise of Fascism and Militarism:**
   - Totalitarian expansionist regimes took power: Adolf Hitler's Nazi Germany, Benito Mussolini's Fascist Italy, and Imperial Japan.
3. **Failure of the League of Nations:**
   - Lacking enforcement authority and military backing, the League failed to stop Japanese expansion into Manchuria (1931), Italian invasion of Abyssinia (1935), or German rearmament.
4. **Policy of Appeasement:**
   - Britain and France permitted German territorial annexations (Rhineland, Austria, Sudetenland via the Munich Agreement of 1938) hoping to avert general war.
5. **Immediate Trigger:**
   - The German invasion of Poland on **September 1, 1939**, followed by Britain and France declaring war on Germany.`;
    }

    // Plate Tectonics
    if (q.includes('plate tectonics') && (q.includes('explain') || q.includes('what is'))) {
      return `### Plate Tectonics (Earth Sciences & Geography)

The theory of **Plate Tectonics** states that the Earth's solid outermost shell (**lithosphere**) is fractured into approximately a dozen major and minor rigid plates that glide over the ductile, semi-fluid **asthenosphere** driven by mantle convection currents.

---

### The Three Types of Plate Boundaries

| Boundary Type | Relative Motion | Geological Features Formed | Real-World Examples |
| :--- | :--- | :--- | :--- |
| **Divergent** | Plates pull apart | Mid-ocean ridges, rift valleys, new oceanic crust | Mid-Atlantic Ridge, East African Rift |
| **Convergent** | Plates collide | Mountain ranges, subduction zones, deep ocean trenches, volcanic arcs | Himalayas (Continental-Continental), Marianas Trench |
| **Transform** | Plates slide past horizontally | Strike-slip fault lines, severe shallow earthquakes | San Andreas Fault (California) |

---

### Driving Mechanism
Mantle convection driven by radioactive decay heat in the Earth's core, along with **slab pull** (dense oceanic plates sinking at subduction zones) and **ridge push**.`;
    }

    // Climate Change
    if (q.includes('climate change') && (q.includes('explain') || q.includes('what is'))) {
      return `### Climate Change & Global Warming

**Climate Change** refers to long-term shifts in global and regional temperatures and weather patterns. While climate has varied over geological eras, contemporary climate change is driven primarily by human activities since the Industrial Revolution.

---

### Scientific Mechanism (The Enhanced Greenhouse Effect)
1. **Solar Radiation:** Shortwave radiation from the Sun warms Earth's surface.
2. **Infrared Emission:** Earth radiates heat back toward space as longwave infrared radiation.
3. **Greenhouse Gases (GHGs):** Atmospheric gases ($\\text{CO}_2, \\text{CH}_4, \\text{N}_2\\text{O}$, fluorinated gases) absorb and re-emit this infrared energy downward, trapping heat in the troposphere.
4. **Anthropogenic Forcing:** Fossil fuel combustion (coal, oil, gas), industrial agriculture, and deforestation have raised atmospheric $\\text{CO}_2$ from pre-industrial levels of $\\sim 280\\text{ ppm}$ to over $420\\text{ ppm}$.

---

### Major Consequences
- Rising sea levels due to thermal expansion and polar ice sheet melting.
- Increased frequency of severe weather events (prolonged droughts, intense cyclones).
- Ocean acidification disrupting coral reefs and marine food chains.`;
    }

    // --- D. ECONOMICS & BUSINESS ---

    // Define Inflation
    if (q.includes('inflation') && (q.includes('define') || q.includes('what is') || q.includes('economics'))) {
      return `### Inflation in Macroeconomics

**Inflation** is the quantitative economic measure of the rate at which the general price level of goods and services in an economy rises over a specified period, resulting in a continuous erosion in the purchasing power of money.

---

### Primary Economic Causes
1. **Demand-Pull Inflation:**
   - Occurs when aggregate consumer demand exceeds aggregate productive capacity ("too much money chasing too few goods").
2. **Cost-Push Inflation:**
   - Caused by aggregate supply disruptions and spikes in production costs (e.g., energy price shocks, raw material shortages, wage hikes).
3. **Built-in / Wage-Price Spiral:**
   - Workers demand higher nominal wages to offset cost-of-living surges, prompting firms to raise consumer prices to maintain margins.

---

### Measurement & Control
- **Consumer Price Index (CPI):** Tracks prices of a representative consumer goods basket.
- **Monetary Policy Tools:** Central Banks increase policy interest rates (repo rate) and conduct quantitative tightening to dampen excess aggregate demand.`;
    }

    // Supply and Demand
    if (q.includes('supply and demand') || (q.includes('supply') && q.includes('demand') && q.includes('explain'))) {
      return `### The Law of Supply and Demand (Microeconomics)

The model of **Supply and Demand** is the cornerstone economic framework describing how price and quantity are determined in a competitive market economy.

---

### 1. Law of Demand
- **Principle:** Holding all else constant (*ceteris paribus*), as the price of a good increases, the quantity demanded decreases (downward-sloping demand curve, $P \\uparrow \\implies Q_d \\downarrow$).

### 2. Law of Supply
- **Principle:** Holding all else constant, as the price of a good increases, producers are willing to supply more to maximize profit (upward-sloping supply curve, $P \\uparrow \\implies Q_s \\uparrow$).

---

### 3. Market Equilibrium
Market equilibrium occurs at the exact price point $P^*$ where quantity demanded equals quantity supplied:

$$Q_d(P^*) = Q_s(P^*)$$

- **Surplus ($Q_s > Q_d$):** Downward market pressure on price until equilibrium is restored.
- **Shortage ($Q_d > Q_s$):** Upward market pressure on price until equilibrium is restored.`;
    }

    // Marketing Management
    if (q.includes('marketing management')) {
      return `### Marketing Management in Business Administration

**Marketing Management** is the strategic and organizational process of planning, executing, pricing, promoting, and distributing ideas, goods, and services to create exchanges that satisfy individual and organizational goals.

---

### 1. The Core 4 Ps Framework (The Marketing Mix)
- **Product:** The physical good, digital service, or solution designed to satisfy customer needs.
- **Price:** Pricing strategy (cost-plus, penetration, skimming, value-based) balancing profitability with customer perceived value.
- **Place:** Distribution channels and supply chain logistics delivering the product to the consumer.
- **Promotion:** Integrated communications (advertising, public relations, social media, sales promotion) to generate brand awareness and demand.

---

### 2. The STP Strategic Process
1. **Segmentation:** Dividing the broad market into distinct buyer groups with shared demographic or behavioral traits.
2. **Targeting:** Selecting the most viable and profitable market segments to serve.
3. **Positioning:** Designing a distinct brand identity and value proposition in the customer's mind relative to competitors.`;
    }

    // Financial Accounting
    if (q.includes('financial accounting')) {
      return `### Financial Accounting in Commerce & Management

**Financial Accounting** is the specialized branch of accounting dedicated to recording, summarizing, and reporting the financial transactions of an enterprise to external stakeholders (investors, creditors, auditors, and regulators) in compliance with GAAP or IFRS standards.

---

### The Fundamental Accounting Equation
$$\\mathbf{\\text{Assets} = \\text{Liabilities} + \\text{Owner's Equity}}$$

- **Assets:** Economic resources owned that provide future economic value (Cash, Inventory, Equipment).
- **Liabilities:** Financial debts and obligations owed to outside parties (Loans, Accounts Payable).
- **Equity:** Residual interest in the company's assets belonging to owners after deducting liabilities.

---

### The Three Master Financial Statements
1. **Balance Sheet:** Snapshot of financial position (Assets, Liabilities, Equity) at a specific date.
2. **Income Statement (Profit & Loss):** Summarizes revenues earned and expenses incurred over an operating period (Net Income = Revenues - Expenses).
3. **Cash Flow Statement:** Tracks cash inflows and outflows across **Operating**, **Investing**, and **Financing** activities.`;
    }

    // --- E. LANGUAGES & LINGUISTICS ---

    // English Tenses
    if (q.includes('english tenses') || (q.includes('tense') && q.includes('english'))) {
      return `### The 12 Tenses in English Grammar

English tenses convey the timeframe (**Past**, **Present**, **Future**) and aspect (**Simple**, **Continuous**, **Perfect**, **Perfect Continuous**) of an action.

---

### Complete 12 Tenses Matrix

| Tense | Present ($V_1$) | Past ($V_2$) | Future (will + $V_1$) |
| :--- | :--- | :--- | :--- |
| **Simple** | *I write code.* | *I wrote code.* | *I will write code.* |
| **Continuous** | *I am writing code.* | *I was writing code.* | *I will be writing code.* |
| **Perfect** | *I have written code.* | *I had written code.* | *I will have written code.* |
| **Perfect Continuous**| *I have been writing code.*| *I had been writing code.*| *I will have been writing code.*|

---

### Core Usage Guide
- **Simple Present:** General truths, scientific facts, habits (*Water boils at $100^\\circ\\text{C}$*).
- **Present Continuous:** Actions currently in progress (*She is studying for her exam*).
- **Present Perfect:** Actions completed at an unspecified time with present relevance (*We have completed the project*).`;
    }

    // Translate into Bengali
    if (q.includes('translate') && q.includes('bengali')) {
      return `### Bengali Translation & Educational Linguistics

**অনুবাদ (Translation):**

*"Education is the most powerful weapon which you can use to change the world."*  
**বাংলা অনুবাদ:** "শিক্ষা হল সবচেয়ে শক্তিশালী হাতিয়ার, যা আপনি পৃথিবী পরিবর্তন করতে ব্যবহার করতে পারেন।"

---

### ভাষাগত ব্যাখ্যা (Linguistic Breakdown)
- **শিক্ষা (Education):** বিশেষ্য (Noun) — জ্ঞানার্জন ও বুদ্ধিবৃত্তিক বিকাশ।
- **শক্তিশালী (Powerful):** বিশেষণ (Adjective) — সামর্থ্য ও প্রভাব বোঝাতে ব্যবহৃত।
- **হাতিয়ার (Weapon/Tool):** রূপক অর্থে ইতিবাচক পরিবর্তনের মাধ্যম হিসেবে ব্যবহৃত।`;
    }

    // --- F. ENGINEERING & TECHNOLOGY ---

    // Ohm's Law
    if (q.includes('ohm') && q.includes('law')) {
      return `### Ohm's Law in Electrical Engineering & Physics

**Ohm's Law** states that the electric current ($I$) flowing through a metallic conductor between two points is directly proportional to the voltage ($V$) across the two points, provided physical conditions (especially temperature) remain constant.

---

### Mathematical Formula

$$V = I \\cdot R$$

Where:
- $V$: Electrical potential difference / Voltage (Volts, $\\text{V}$).
- $I$: Electric current (Amperes, $\\text{A}$).
- $R$: Electrical resistance (Ohms, $\\Omega$).

---

### Derived Formulas
- Current: $I = \\frac{V}{R}$
- Resistance: $R = \\frac{V}{I}$
- Electric Power: $P = V \\cdot I = I^2 R = \\frac{V^2}{R}$ (Watts, $\\text{W}$).

---

### Limitations of Ohm's Law
Ohm's Law applies to **Ohmic conductors** (metals at constant temperature). It does **not** hold for non-linear, non-ohmic devices such as diodes, transistors, thermistors, and vacuum tubes where the $V-I$ characteristic is non-linear.`;
    }

    // Transformer
    if (q.includes('transformer') && (q.includes('what is') || q.includes('explain') || q.includes('working'))) {
      return `### Electric Transformer (Electrical Engineering)

A **Transformer** is a static electrical machine that transfers electrical energy between two or more circuits through **mutual electromagnetic induction**, altering alternating voltage and current levels without changing the frequency.

---

### Working Principle
A transformer consists of two isolated coils wound on a laminated ferromagnetic core:
1. **Primary Winding:** Connected to input AC supply, producing alternating magnetic flux $\\Phi$ in the core.
2. **Secondary Winding:** Linked by this flux; by Faraday's Law, an alternating EMF is induced across its terminals.

---

### The Turns Ratio Equation

$$\\frac{V_s}{V_p} = \\frac{N_s}{N_p} = \\frac{I_p}{I_s} = k$$

Where:
- $V_p, V_s$: Primary and secondary voltages.
- $N_p, N_s$: Primary and secondary turns.
- $I_p, I_s$: Primary and secondary currents.

---

### Classifications
- **Step-Up Transformer ($N_s > N_p$):** Increases voltage ($V_s > V_p$), decreases current. Used at generating stations for high-voltage grid transmission to reduce $I^2R$ transmission line losses.
- **Step-Down Transformer ($N_s < N_p$):** Decreases voltage ($V_s < V_p$), increases current. Used in local distribution substations ($230\\text{V} / 120\\text{V}$) for household distribution.`;
    }

    return null;
  }

  // =========================================================================
  // 6. REAL-TIME FACTUAL WIKIPEDIA KNOWLEDGE RETRIEVER
  // =========================================================================
  async function fetchLiveWikiArticle(topic) {
    if (!topic || topic.length < 3) return null;
    const cleanTopic = encodeURIComponent(topic.trim().replace(/\s+/g, '_'));
    const url = `https://en.wikipedia.org/api/rest_v1/page/summary/${cleanTopic}`;

    try {
      const resp = await fetch(url, { headers: { 'Accept': 'application/json' } });
      if (resp.ok) {
        const data = await resp.json();
        if (data && data.extract && data.type !== 'disambiguation' && data.extract.length > 50) {
          return data;
        }
      }
    } catch (e) {
      // Ignore network errors
    }
    return null;
  }

  // =========================================================================
  // 7. BACKEND API CLIENT WITH DIRECT CLIENT-SIDE FALLBACK
  // =========================================================================
  async function queryBackendAI(query, history = [], fileSnapshot = null) {
    const clientKey = localStorage.getItem('gemini_api_key') || window.GEMINI_API_KEY || '';

    // 1. Backend Server Endpoint Call
    for (const backendUrl of BACKEND_URLS) {
      try {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 14000);

        const response = await fetch(backendUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            query: query,
            history: history.slice(-12),
            file: fileSnapshot,
            apiKey: clientKey
          }),
          signal: controller.signal
        });
        clearTimeout(timeout);

        if (response.ok) {
          const data = await response.json();
          if (data && data.success && data.text) {
            return { text: data.text, source: 'backend-gemini' };
          }
        }
      } catch (err) {
        // Continue to fallback
      }
    }

    // 2. Client-Side Direct Google Gemini 1.5 Flash (for static GitHub Pages hosting)
    if (clientKey) {
      try {
        const directEndpoint = `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${encodeURIComponent(clientKey)}`;
        const contents = [];

        for (const item of history.slice(-12)) {
          if (item && item.text) {
            contents.push({
              role: item.role === 'model' || item.role === 'assistant' ? 'model' : 'user',
              parts: [{ text: item.text }]
            });
          }
        }

        const currentParts = [];
        if (fileSnapshot && fileSnapshot.base64 && fileSnapshot.base64.startsWith('data:image/')) {
          currentParts.push({
            inlineData: {
              mimeType: fileSnapshot.mimeType || 'image/jpeg',
              data: fileSnapshot.base64.split(',')[1]
            }
          });
        }

        let fullTextPrompt = query;
        if (fileSnapshot && fileSnapshot.text) {
          fullTextPrompt = `[ATTACHED EDUCATIONAL DOCUMENT: ${fileSnapshot.name} (${fileSnapshot.type})]\n${fileSnapshot.text.substring(0, 15000)}\n\n[STUDENT QUESTION]:\n${query}`;
        }
        currentParts.push({ text: fullTextPrompt });

        contents.push({ role: 'user', parts: currentParts });

        const resp = await fetch(directEndpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            contents: contents,
            generationConfig: { temperature: 0.2, maxOutputTokens: 3500 }
          })
        });

        if (resp.ok) {
          const geminiData = await resp.json();
          const text = geminiData.candidates?.[0]?.content?.parts?.[0]?.text;
          if (text) {
            return { text: text.trim(), source: 'direct-gemini' };
          }
        }
      } catch (e) {
        console.warn('Direct client-side Gemini notice:', e);
      }
    }

    return null;
  }

  // =========================================================================
  // 8. TIERED UNIVERSAL EDUCATIONAL INFERENCE ENGINE
  // =========================================================================
  async function generateAcademicResponse(query, fileSnapshot = null) {
    const qClean = sanitizeUserQuery(query);
    const qLower = qClean.toLowerCase();

    // 1. Conversational Greetings
    const greetings = {
      'hi': 'Hello! I am Brainware Academic AI, your worldwide educational assistant. I can help you with Computer Science, Mathematics, Sciences, Engineering, Humanities, Business, and Languages. How can I assist your studies today?',
      'hello': 'Greetings! What academic subject, problem, or concept would you like to explore today?',
      'hey': 'Hello! What educational question or calculation can I help you solve today?'
    };
    if (greetings[qLower]) {
      return { text: greetings[qLower], source: 'conversational' };
    }

    // 2. Primary Engine: Backend Proxy or Client Gemini 1.5 Flash
    const aiResult = await queryBackendAI(query, chatHistory, fileSnapshot);
    if (aiResult && aiResult.text) {
      return aiResult;
    }

    // 3. Document Analysis Mode (When document is attached and AI model is offline)
    if (fileSnapshot && fileSnapshot.text) {
      return {
        text: `### Document Analysis: ${fileSnapshot.name}\n\n**Format:** ${fileSnapshot.type} (${fileSnapshot.size})\n\n**Extracted Text Excerpt:**\n\n\`\`\`\n${fileSnapshot.text.substring(0, 1500)}...\n\`\`\`\n\n---\n\n### Summary Assessment\nThis ${fileSnapshot.type} document has been parsed. For deep cross-page reasoning and visual query answering, start the Brainware AI backend server (\`npm start\`) or configure your free Gemini API key in settings.`,
        source: 'document'
      };
    }

    // 4. University Question Bank Grounding (portalQAIndex: 1,254 entries)
    const syllabusMatch = findSyllabusMatch(query);
    if (syllabusMatch && syllabusMatch.item) {
      const item = syllabusMatch.item;
      const cleanAnswer = cleanSyllabusHtml(item.a || item.ans || item.hint || '');
      const badge = `> **[Verified University Question Bank Solution]**  \n> *Source: ${syllabusMatch.source}*\n\n`;
      return {
        text: `${badge}### Question: ${item.q}\n\n${cleanAnswer}`,
        source: 'syllabus'
      };
    }

    // 5. Mathematical & Probability Calculation Engine
    const mathAns = trySolveMathProblem(query);
    if (mathAns) {
      return { text: mathAns, source: 'math' };
    }

    // 6. Programming Implementation Engine
    const codeAns = tryGenerateCodeSolution(query);
    if (codeAns) {
      return { text: codeAns, source: 'code' };
    }

    // 7. Universal Multi-Disciplinary Subject Synthesizer
    const conceptAns = trySubjectConceptExplanation(query, chatHistory);
    if (conceptAns) {
      return { text: conceptAns, source: 'concept' };
    }

    // 8. Factual Knowledge Retrieval via Wikipedia API
    const cleanTopic = qClean
      .replace(/^(what is|explain|describe|define|tell me about|how does|why is|what are|state)\s+/i, '')
      .replace(/[?!.]/g, '')
      .trim();

    if (cleanTopic.length > 2) {
      const wiki = await fetchLiveWikiArticle(cleanTopic);
      if (wiki && wiki.extract && wiki.extract.length > 40) {
        return {
          text: `### ${wiki.title} ${wiki.description ? `*(${wiki.description})*` : ''}\n\n${wiki.extract}\n\n---\n\n*Reference: Grounded in authoritative academic reference materials.*`,
          source: 'wikipedia'
        };
      }
    }

    // 9. Informative Guidance
    return {
      text: `### Academic Response\n\nTo ensure complete accuracy across all disciplines for **"${qClean}"**, please launch the Brainware Academic AI backend server (\`npm start\`) or configure your free Google Gemini API key by clicking the **API Key** button in the header navigation.`,
      source: 'guidance'
    };
  }

  // =========================================================================
  // 9. KATEX MATH & MARKDOWN RENDERING
  // =========================================================================
  const mathSymbolMap = [
    [/\\alpha\b/g, 'α'], [/\\beta\b/g, 'β'], [/\\gamma\b/g, 'γ'], [/\\delta\b/g, 'δ'],
    [/\\theta\b/g, 'θ'], [/\\lambda\b/g, 'λ'], [/\\mu\b/g, 'μ'], [/\\pi\b/g, 'π'],
    [/\\sigma\b/g, 'σ'], [/\\tau\b/g, 'τ'], [/\\omega\b/g, 'ω'], [/\\rightarrow\b/g, '→'],
    [/\\leftarrow\b/g, '←'], [/\\le\b/g, '≤'], [/\\ge\b/g, '≥'], [/\\neq\b/g, '≠'],
    [/\\times\b/g, '×'], [/\\infty\b/g, '∞'], [/\\mathcal\{O\}/g, 'O']
  ];

  function applyUnicodeMathFallback(str) {
    if (!str) return '';
    let s = str;
    for (const [regex, rep] of mathSymbolMap) {
      s = s.replace(regex, rep);
    }
    return s;
  }

  function renderMathInHtml(content) {
    if (!content) return '';
    let processed = content;
    const hasKatex = typeof window.katex !== 'undefined' && typeof window.katex.renderToString === 'function';

    // 1. Block Math: $$...$$ and \[...\]
    processed = processed.replace(/\$\$([\s\S]*?)\$\$/g, (match, expr) => {
      if (hasKatex) {
        try {
          return `<div class="katex-display my-3 overflow-x-auto text-center">${window.katex.renderToString(expr.trim(), { displayMode: true, throwOnError: false })}</div>`;
        } catch (e) {}
      }
      return `<div class="font-mono bg-blue-50/60 text-blue-900 p-3 my-2 rounded-xl text-center overflow-x-auto border border-blue-100">${applyUnicodeMathFallback(expr.trim())}</div>`;
    });

    processed = processed.replace(/\\\[([\s\S]*?)\\\]/g, (match, expr) => {
      if (hasKatex) {
        try {
          return `<div class="katex-display my-3 overflow-x-auto text-center">${window.katex.renderToString(expr.trim(), { displayMode: true, throwOnError: false })}</div>`;
        } catch (e) {}
      }
      return `<div class="font-mono bg-blue-50/60 text-blue-900 p-3 my-2 rounded-xl text-center overflow-x-auto border border-blue-100">${applyUnicodeMathFallback(expr.trim())}</div>`;
    });

    // 2. Inline Math: $...$ and \(...\)
    processed = processed.replace(/\$([^\$\n]+?)\$/g, (match, expr) => {
      if (hasKatex) {
        try {
          return window.katex.renderToString(expr.trim(), { displayMode: false, throwOnError: false });
        } catch (e) {}
      }
      return `<span class="font-mono text-blue-800 font-semibold bg-blue-50/50 px-1 py-0.5 rounded">${applyUnicodeMathFallback(expr.trim())}</span>`;
    });

    processed = processed.replace(/\\\(([\s\S]*?)\\\)/g, (match, expr) => {
      if (hasKatex) {
        try {
          return window.katex.renderToString(expr.trim(), { displayMode: false, throwOnError: false });
        } catch (e) {}
      }
      return `<span class="font-mono text-blue-800 font-semibold bg-blue-50/50 px-1 py-0.5 rounded">${applyUnicodeMathFallback(expr.trim())}</span>`;
    });

    return processed;
  }

  function formatMarkdown(text) {
    if (!text) return '';
    let renderedHtml = '';

    if (typeof marked !== 'undefined' && typeof marked.parse === 'function') {
      try {
        renderedHtml = marked.parse(text);
      } catch (err) {
        renderedHtml = `<p>${text.replace(/\n/g, '<br>')}</p>`;
      }
    } else {
      renderedHtml = `<p>${text.replace(/\n/g, '<br>')}</p>`;
    }

    renderedHtml = renderMathInHtml(renderedHtml);

    // Style Code Blocks
    renderedHtml = renderedHtml.replace(/<pre><code class="language-([a-zA-Z0-9_-]+)">([\s\S]*?)<\/code><\/pre>/g, (match, lang, code) => {
      const codeId = 'code_' + Math.random().toString(36).substr(2, 9);
      return `
        <div class="relative my-4 rounded-2xl overflow-hidden border border-slate-700 bg-slate-900 text-slate-100 shadow-md">
          <div class="flex items-center justify-between px-4 py-2 bg-slate-800 border-b border-slate-700 text-xs font-mono text-slate-400">
            <span class="font-bold text-blue-400 uppercase tracking-wider">${lang}</span>
            <button onclick="BrainwareAITutor.copyCodeBlock('${codeId}', this)" class="hover:text-white transition flex items-center gap-1 text-[11px] bg-slate-700/80 px-2 py-0.5 rounded">
              <i data-lucide="copy" class="w-3 h-3"></i> <span>Copy</span>
            </button>
          </div>
          <pre class="p-4 text-xs font-mono overflow-x-auto leading-relaxed"><code id="${codeId}">${code}</code></pre>
        </div>
      `;
    });

    renderedHtml = renderedHtml.replace(/<pre><code>([\s\S]*?)<\/code><\/pre>/g, (match, code) => {
      const codeId = 'code_' + Math.random().toString(36).substr(2, 9);
      return `
        <div class="relative my-4 rounded-2xl overflow-hidden border border-slate-700 bg-slate-900 text-slate-100 shadow-md">
          <div class="flex items-center justify-between px-4 py-2 bg-slate-800 border-b border-slate-700 text-xs font-mono text-slate-400">
            <span class="font-bold text-slate-300">CODE</span>
            <button onclick="BrainwareAITutor.copyCodeBlock('${codeId}', this)" class="hover:text-white transition flex items-center gap-1 text-[11px] bg-slate-700/80 px-2 py-0.5 rounded">
              <i data-lucide="copy" class="w-3 h-3"></i> <span>Copy</span>
            </button>
          </div>
          <pre class="p-4 text-xs font-mono overflow-x-auto leading-relaxed"><code id="${codeId}">${code}</code></pre>
        </div>
      `;
    });

    // Style Tables
    renderedHtml = renderedHtml.replace(/<table>([\s\S]*?)<\/table>/g, (match, tableInner) => {
      return `
        <div class="my-4 overflow-x-auto rounded-2xl border border-slate-200 shadow-sm bg-white">
          <table class="min-w-full text-xs text-left border-collapse">${tableInner}</table>
        </div>
      `;
    });

    renderedHtml = renderedHtml.replace(/<thead>/g, '<thead class="bg-slate-100 font-black text-slate-800 border-b border-slate-200">');
    renderedHtml = renderedHtml.replace(/<th>/g, '<th class="px-4 py-3 border-r border-slate-200 last:border-r-0">');
    renderedHtml = renderedHtml.replace(/<td>/g, '<td class="px-4 py-2.5 border-t border-slate-100 border-r border-slate-200 last:border-r-0 text-slate-700 font-medium">');

    return renderedHtml;
  }

  // =========================================================================
  // 10. STREAMING TYPEWRITER EFFECT
  // =========================================================================
  async function streamTypewriter(fullText, contentContainer, scrollContainer) {
    if (!contentContainer) return;
    const chunkSize = 6;
    let currentLen = 0;

    while (currentLen < fullText.length) {
      if (!isGenerating) break;
      currentLen = Math.min(fullText.length, currentLen + chunkSize);
      const partial = fullText.substring(0, currentLen);
      contentContainer.innerHTML = formatMarkdown(partial);

      if (scrollContainer) {
        scrollContainer.scrollTop = scrollContainer.scrollHeight;
      }
      await new Promise(resolve => setTimeout(resolve, 14));
    }

    contentContainer.innerHTML = formatMarkdown(fullText);
    if (scrollContainer) {
      scrollContainer.scrollTop = scrollContainer.scrollHeight;
    }
  }

  // =========================================================================
  // 11. UI BUBBLES & ACTIONS
  // =========================================================================
  function createAssistantStreamingBubble(msgId, threadId) {
    const thread = document.getElementById(threadId);
    if (!thread) return { bubble: null, contentContainer: null };

    const bubble = document.createElement('div');
    bubble.id = msgId;
    bubble.className = 'flex gap-3 mb-5 animate-fade-in group';

    bubble.innerHTML = `
      <div class="w-9 h-9 rounded-2xl bg-gradient-to-tr from-blue-600 to-indigo-600 text-white flex items-center justify-center font-black flex-shrink-0 shadow-md">
        <i data-lucide="bot" class="w-5 h-5"></i>
      </div>
      <div class="flex-1 bg-white border border-slate-200/80 rounded-3xl rounded-tl-sm p-4 sm:p-5 shadow-sm overflow-hidden text-xs sm:text-sm text-slate-800 leading-relaxed">
        <div class="ai-content-body prose max-w-none prose-sm font-normal text-slate-800">
          <div class="flex items-center gap-2 text-slate-400 text-xs italic py-1">
            <span class="inline-block w-2 h-2 rounded-full bg-blue-500 animate-pulse"></span>
            Thinking and generating academic response...
          </div>
        </div>
        <div class="ai-msg-actions hidden mt-3 pt-3 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500">
          <div class="flex items-center gap-2">
            <button onclick="BrainwareAITutor.copyMessageText('${msgId}', this)" class="hover:text-blue-600 px-2 py-1 rounded-lg hover:bg-slate-50 transition flex items-center gap-1 font-semibold text-[11px]" title="Copy full response">
              <i data-lucide="copy" class="w-3.5 h-3.5"></i> <span>Copy</span>
            </button>
            <button onclick="BrainwareAITutor.readAloud('${msgId}', this)" class="hover:text-blue-600 px-2 py-1 rounded-lg hover:bg-slate-50 transition flex items-center gap-1 font-semibold text-[11px]" title="Read response aloud">
              <i data-lucide="volume-2" class="w-3.5 h-3.5"></i> <span>Read</span>
            </button>
          </div>
          <span class="text-[10px] text-slate-400 font-mono">Brainware Academic AI</span>
        </div>
      </div>
    `;

    thread.appendChild(bubble);
    thread.scrollTop = thread.scrollHeight;

    if (window.lucide && typeof window.lucide.createIcons === 'function') {
      window.lucide.createIcons();
    }

    const contentContainer = bubble.querySelector('.ai-content-body');
    return { bubble, contentContainer };
  }

  function appendUserBubble(text, threadId, fileSnapshot = null) {
    const thread = document.getElementById(threadId);
    if (!thread) return;

    const bubble = document.createElement('div');
    bubble.className = 'flex justify-end mb-4 animate-fade-in';

    let fileBadgeHtml = '';
    if (fileSnapshot) {
      fileBadgeHtml = `
        <div class="mb-2 pb-2 border-b border-white/20 flex items-center gap-2 text-xs">
          <span class="inline-flex items-center gap-1.5 px-3 py-1 bg-white/20 text-white rounded-xl font-medium shadow-sm">
            <i data-lucide="paperclip" class="w-3.5 h-3.5"></i>
            <b class="truncate max-w-[200px]">${escapeHtml(fileSnapshot.name)}</b>
            <span class="text-[10px] opacity-80 font-mono">(${fileSnapshot.size})</span>
          </span>
        </div>
      `;
    }

    bubble.innerHTML = `
      <div class="max-w-[85%] bg-blue-600 text-white rounded-3xl rounded-tr-sm px-5 py-3 text-xs sm:text-sm font-medium leading-relaxed shadow-sm">
        ${fileBadgeHtml}
        <div>${escapeHtml(text)}</div>
      </div>
    `;

    thread.appendChild(bubble);
    thread.scrollTop = thread.scrollHeight;

    if (window.lucide && typeof window.lucide.createIcons === 'function') {
      window.lucide.createIcons();
    }
  }

  function finishMessage(msgId, fullText, threadId) {
    const bubble = document.getElementById(msgId);
    if (!bubble) return;

    const actions = bubble.querySelector('.ai-msg-actions');
    if (actions) {
      actions.classList.remove('hidden');
    }

    const chipsContainer = document.createElement('div');
    chipsContainer.className = 'flex flex-wrap gap-1.5 mt-3 pt-2';

    const followUps = generateFollowUpSuggestions(lastUserQuery, fullText);
    followUps.forEach(prompt => {
      const chip = document.createElement('button');
      chip.type = 'button';
      chip.className = 'text-[11px] bg-slate-100 hover:bg-blue-50 hover:text-blue-600 text-slate-700 px-3 py-1 rounded-xl transition border border-slate-200 font-semibold';
      chip.innerText = prompt;
      chip.onclick = () => {
        const isPopup = threadId === 'fbChatMessages';
        const inputId = isPopup ? 'fbChatInput' : 'aiTutorInput';
        const input = document.getElementById(inputId);
        if (input) input.value = prompt;
        window.BrainwareAITutor.askQuestion(prompt, isPopup);
      };
      chipsContainer.appendChild(chip);
    });

    const body = bubble.querySelector('.flex-1');
    if (body) {
      body.appendChild(chipsContainer);
    }

    if (window.lucide && typeof window.lucide.createIcons === 'function') {
      window.lucide.createIcons();
    }

    const container = document.getElementById(threadId);
    if (container) container.scrollTop = container.scrollHeight;
  }

  function generateFollowUpSuggestions(userQ, aiResp) {
    const q = (userQ || '').toLowerCase();
    if (q.includes('java') || q.includes('c++') || q.includes('python')) {
      return ['What are its main advantages?', 'Provide an advanced code example', 'Compare with alternative languages'];
    }
    if (q.includes('newton') || q.includes('physics')) {
      return ['Solve a numerical problem using this law', 'Explain Newton’s Third Law', 'Relate momentum to impulse'];
    }
    if (q.includes('differentiation') || q.includes('calculus')) {
      return ['Explain Integration as inverse differentiation', 'Show step-by-step example using Chain Rule', 'What is partial differentiation?'];
    }
    if (q.includes('photosynthesis') || q.includes('biology')) {
      return ['Explain the Light-Dependent reactions', 'What is the role of Chlorophyll?', 'Detail the Calvin Cycle'];
    }
    if (q.includes('inflation') || q.includes('economics')) {
      return ['What are the key causes of demand-pull inflation?', 'How do Central Banks control inflation?', 'What is stagflation?'];
    }
    if (q.includes('transformer') || q.includes('engineering')) {
      return ['What are the main losses in a transformer?', 'Derive the EMF equation of a transformer', 'Explain Step-Up vs Step-Down'];
    }
    return ['Provide a detailed step-by-step example', 'What are the main advantages and disadvantages?', 'Explain with a comparison table'];
  }

  function escapeHtml(str) {
    if (!str) return '';
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#039;');
  }

  // =========================================================================
  // 12. PUBLIC CONTROLLER API: window.BrainwareAITutor
  // =========================================================================
  window.BrainwareAITutor = {
    handleFileUpload: async function(event, isPopup = false) {
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
        prevEl.style.removeProperty('display');
        prevEl.style.display = 'flex';
      }

      try {
        attachedFileData = await parseUploadedDocument(file);
        if (sizeEl && attachedFileData) {
          sizeEl.innerText = `(${attachedFileData.size})`;
        }
        if (window.lucide && typeof window.lucide.createIcons === 'function') {
          window.lucide.createIcons();
        }
      } catch (err) {
        console.error('File parsing notice:', err);
        if (sizeEl) sizeEl.innerText = `(${ (file.size/1024).toFixed(1) } KB)`;
      }
    },

    removeAttachedFile: function(isPopup = false) {
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

    askQuestion: async function(customQuery = null, isPopup = false) {
      if (isGenerating) return;

      const inputId = isPopup ? 'fbChatInput' : 'aiTutorInput';
      const input = document.getElementById(inputId);
      let query = customQuery || (input ? input.value : '');

      if ((!query || !query.trim()) && attachedFileData) {
        query = `Please analyze the attached document "${attachedFileData.name}" (${attachedFileData.type}) in detail. Provide an executive summary, list key concepts, explain all formulas and data tables, and highlight critical insights.`;
      }

      if (!query || !query.trim()) return;

      lastUserQuery = query.trim();
      if (input && !customQuery) {
        input.value = '';
      }

      const fileSnapshot = attachedFileData;
      this.removeAttachedFile(isPopup);

      const threadId = isPopup ? 'fbChatMessages' : 'aiChatThread';
      const threadContainer = document.getElementById(threadId);

      appendUserBubble(query, threadId, fileSnapshot);

      const msgId = 'aimsg_' + (++messageCounter);
      const { contentContainer } = createAssistantStreamingBubble(msgId, threadId);

      isGenerating = true;
      this.updateSendButton(true, isPopup);

      try {
        const result = await generateAcademicResponse(query, fileSnapshot);
        const responseText = result && result.text ? result.text : 'I could not generate an answer at this time. Please try again.';

        chatHistory.push({ role: 'user', text: query });
        chatHistory.push({ role: 'model', text: responseText });

        await streamTypewriter(responseText, contentContainer, threadContainer);
        finishMessage(msgId, responseText, threadId);
      } catch (err) {
        console.error('AI generation error:', err);
        if (contentContainer) {
          contentContainer.innerHTML = `<div class="p-3 bg-red-50 text-red-700 rounded-xl border border-red-200 font-semibold text-xs">Error: ${escapeHtml(err.message || 'Failed to process question. Please check network connection.')}</div>`;
        }
      } finally {
        isGenerating = false;
        this.updateSendButton(false, isPopup);
      }
    },

    stopGeneration: function() {
      isGenerating = false;
      if (abortController) {
        abortController.abort();
      }
      this.updateSendButton(false, false);
      this.updateSendButton(false, true);
    },

    updateSendButton: function(generating, isPopup = false) {
      const btnId = isPopup ? 'fbSendBtn' : 'aiSendBtn';
      const btn = document.getElementById(btnId);
      if (!btn) return;

      if (generating) {
        btn.innerHTML = `
          <svg class="w-4 h-4 animate-spin text-white" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
        `;
        btn.setAttribute('title', 'Stop Generating');
        btn.onclick = (e) => {
          e.preventDefault();
          this.stopGeneration();
        };
      } else {
        btn.innerHTML = `<i data-lucide="send" class="w-4 h-4"></i>`;
        btn.setAttribute('title', 'Send Message');
        btn.onclick = null;
        if (window.lucide && typeof window.lucide.createIcons === 'function') {
          window.lucide.createIcons();
        }
      }
    },

    copyCodeBlock: function(codeId, btn) {
      const codeEl = document.getElementById(codeId);
      if (!codeEl) return;
      const text = codeEl.textContent || codeEl.innerText;
      navigator.clipboard.writeText(text).then(() => {
        const span = btn.querySelector('span');
        if (span) {
          const oldText = span.textContent;
          span.textContent = 'Copied!';
          setTimeout(() => { span.textContent = oldText; }, 2000);
        }
      });
    },

    copyMessageText: function(msgId, btn) {
      const bubble = document.getElementById(msgId);
      if (!bubble) return;
      const body = bubble.querySelector('.ai-content-body');
      if (!body) return;
      const text = body.textContent || body.innerText;
      navigator.clipboard.writeText(text.trim()).then(() => {
        const span = btn.querySelector('span');
        if (span) {
          const oldText = span.textContent;
          span.textContent = 'Copied!';
          setTimeout(() => { span.textContent = oldText; }, 2000);
        }
      });
    },

    readAloud: function(msgId, btn) {
      if (!window.speechSynthesis) {
        alert('Text-to-Speech is not supported in this browser.');
        return;
      }

      if (isSpeaking) {
        window.speechSynthesis.cancel();
        isSpeaking = false;
        const span = btn.querySelector('span');
        if (span) span.textContent = 'Read';
        return;
      }

      const bubble = document.getElementById(msgId);
      if (!bubble) return;
      const body = bubble.querySelector('.ai-content-body');
      if (!body) return;

      const text = (body.textContent || body.innerText).replace(/[`$#*_~|]/g, ' ').replace(/\s+/g, ' ').trim();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.0;
      utterance.pitch = 1.0;

      const span = btn.querySelector('span');
      if (span) span.textContent = 'Stop';
      isSpeaking = true;

      utterance.onend = () => {
        isSpeaking = false;
        if (span) span.textContent = 'Read';
      };
      utterance.onerror = () => {
        isSpeaking = false;
        if (span) span.textContent = 'Read';
      };

      window.speechSynthesis.speak(utterance);
    },

    configureApiKey: function() {
      const current = localStorage.getItem('gemini_api_key') || '';
      const key = prompt('Enter your Google Gemini API Key (e.g. AIzaSy...)\n\nTip: You can obtain a 100% free key at https://aistudio.google.com/app/apikey\nLeave blank to clear and use backend server.', current);
      if (key !== null) {
        if (key.trim()) {
          localStorage.setItem('gemini_api_key', key.trim());
          alert('✓ Google Gemini 1.5 Flash API Key successfully configured and activated!');
        } else {
          localStorage.removeItem('gemini_api_key');
          alert('✓ Client Gemini API Key cleared. Backend server / Universal Academic Engine is active.');
        }
      }
    },

    clearChat: function() {
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
              <p class="text-xs text-slate-600">Conversation cleared. How can I help you today?</p>
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
              <p class="text-[11px] text-slate-600">Ask any question across all academic subjects.</p>
            </div>
          </div>
        `;
      }
      if (window.lucide && typeof window.lucide.createIcons === 'function') {
        window.lucide.createIcons();
      }
    },

    toggleChatPopup: function() {
      const popup = document.getElementById('floatingChatPopup');
      if (!popup) return;
      if (popup.classList.contains('hidden') || popup.style.display === 'none') {
        popup.classList.remove('hidden');
        popup.style.display = 'flex';
      } else {
        popup.classList.add('hidden');
        popup.style.display = 'none';
      }
    }
  };

  // Drag and Drop Zone
  document.addEventListener('DOMContentLoaded', () => {
    const dropZones = [document.getElementById('aiChatContainer'), document.getElementById('floatingChatPopup')].filter(Boolean);
    dropZones.forEach(zone => {
      zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('ring-2', 'ring-blue-500');
      });
      zone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        zone.classList.remove('ring-2', 'ring-blue-500');
      });
      zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('ring-2', 'ring-blue-500');
        const files = e.dataTransfer.files;
        if (files && files[0]) {
          const isPopup = zone.id === 'floatingChatPopup';
          window.BrainwareAITutor.handleFileUpload({ target: { files: [files[0]] } }, isPopup);
        }
      });
    });

    if (window.lucide && typeof window.lucide.createIcons === 'function') {
      window.lucide.createIcons();
    }
  });

})();
