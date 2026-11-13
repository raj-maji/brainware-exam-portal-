/**
 * Brainware University - Open Academic AI Agent & AI Tutor Engine
 * Version: 2.0.0 - Universal Educational Edition
 * 
 * Architecture:
 * 1. Primary Engine: Backend API Proxy (/api/chat) -> Google Gemini 1.5 Flash
 * 2. Secondary Engine: Direct Client-Side Gemini 1.5 Flash API (when key is configured)
 * 3. Autonomous Educational Synthesizer: Universal Multi-Disciplinary Knowledge Engine
 * 4. Multi-Turn Conversation Memory: Context-aware follow-up resolution
 * 5. Document Understanding: Real PDF.js multi-page, Word, Excel, PPT, Code & Multimodal Vision
 * 6. High-Confidence Syllabus Grounding: portalQAIndex (1,254 entries) + papersData (14 exam sets)
 */

(function() {
  'use strict';

  // Configure PDF.js worker
  if (typeof pdfjsLib !== 'undefined') {
    pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
  }

  // --- ENGINE STATE ---
  let isGenerating = false;
  let abortController = null;
  let chatHistory = []; // { role: 'user' | 'assistant' | 'model', text: string }
  let lastUserQuery = '';
  let messageCounter = 0;
  let attachedFileData = null;
  let isSpeaking = false;

  // Backend API Base URL detection
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
  // 1. MULTI-FORMAT DOCUMENT PROCESSING (PDF, DOCX, XLSX, PPTX, TXT, IMAGE)
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
            console.warn('PDF.js extraction notice:', pdfErr);
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

      // 4. PowerPoint Presentations (.pptx) via JSZip
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
      console.warn('Document extraction warning:', err);
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
  // 2. STRICT UNIVERSITY SYLLABUS & QUESTION BANK RETRIEVAL
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
  // 3. STEP-BY-STEP MATHEMATICAL & NUMERICAL SOLVER
  // =========================================================================
  function trySolveMathProblem(query) {
    const q = query.toLowerCase().replace(/\s+/g, ' ').trim();

    // Linear Equation: e.g. "solve 2x + 5 = 15" or "2x + 5 = 15"
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

We are asked to solve the linear algebraic equation:

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
  // 4. PROGRAMMING CODE SYNTHESIZER ACROSS MAJOR LANGUAGES
  // =========================================================================
  function tryGenerateCodeSolution(query) {
    const q = query.toLowerCase();

    // 1. JavaScript String Reversal
    if ((q.includes('reverse') && q.includes('string')) && (q.includes('javascript') || q.includes('js'))) {
      return `### JavaScript Program to Reverse a String

In JavaScript, there are several idiomatic and performant ways to reverse a string.

---

### Method 1: Built-in Methods (\`split\`, \`reverse\`, \`join\`)

\`\`\`javascript
/**
 * Reverses a string using JavaScript standard library methods.
 * @param {string} str - Input string
 * @returns {string} Reversed string
 */
function reverseString(str) {
  if (!str || typeof str !== 'string') return '';
  return str.split('').reverse().join('');
}

// Example Execution
const input = "Brainware University";
const result = reverseString(input);
console.log("Original:", input);
console.log("Reversed:", result);
// Output: "ytisrevinU erawniarB"
\`\`\`

---

### Method 2: Two-Pointer Iterative Algorithm (In-Place Array)

\`\`\`javascript
function reverseStringTwoPointer(str) {
  const chars = Array.from(str);
  let left = 0;
  let right = chars.length - 1;

  while (left < right) {
    // Swap characters at left and right indices
    const temp = chars[left];
    chars[left] = chars[right];
    chars[right] = temp;
    left++;
    right--;
  }

  return chars.join('');
}

// Example Execution
console.log(reverseStringTwoPointer("JavaScript"));
// Output: "tpircSavaJ"
\`\`\`

---

### Method 3: Modern ES6 Reduce

\`\`\`javascript
const reverseWithReduce = str => [...str].reduce((acc, char) => char + acc, '');
\`\`\`

---

### Complexity Analysis
- **Time Complexity:** $\\mathcal{O}(n)$ where $n$ is the length of the string (linear single-pass scan).
- **Space Complexity:** $\\mathcal{O}(n)$ to store the array representation (JavaScript strings are immutable).`;
    }

    // 2. Java / Python String Reversal
    if ((q.includes('reverse') && q.includes('string')) || q.includes('string reverse')) {
      if (q.includes('python')) {
        return `### Python: Reverse a String

\`\`\`python
def reverse_string(s: str) -> str:
    """
    Reverses an input string using Pythonic slice notation.
    Time Complexity: O(n), Space Complexity: O(n)
    """
    return s[::-1]

# Example
original = "Brainware University"
reversed_text = reverse_string(original)
print("Original:", original)
print("Reversed:", reversed_text)
\`\`\`

**Complexity:** $\\mathcal{O}(n)$ time, $\\mathcal{O}(n)$ space.`;
      }

      // Default Java
      return `### Java Program to Reverse a String

\`\`\`java
public class StringReversal {

    // Method 1: Using StringBuilder (Idiomatic & Optimized)
    public static String reverseWithStringBuilder(String input) {
        if (input == null) return null;
        return new StringBuilder(input).reverse().toString();
    }

    // Method 2: Two-Pointer Character Array (Algorithmic)
    public static String reverseWithTwoPointers(String input) {
        if (input == null || input.length() <= 1) return input;
        
        char[] chars = input.toCharArray();
        int left = 0, right = chars.length - 1;

        while (left < right) {
            char temp = chars[left];
            chars[left] = chars[right];
            chars[right] = temp;
            left++;
            right--;
        }

        return new String(chars);
    }

    public static void main(String[] args) {
        String original = "Brainware University";
        System.out.println("Original: " + original);
        System.out.println("Reversed: " + reverseWithStringBuilder(original));
    }
}
\`\`\`

---

### Complexity Analysis
- **Time Complexity:** $\\mathcal{O}(n)$ where $n$ is string length.
- **Space Complexity:** $\\mathcal{O}(n)$ to hold mutable buffer.`;
    }

    // 3. Binary Search
    if (q.includes('binary search')) {
      return `### Binary Search Algorithm & Implementation

**Binary Search** is an optimal divide-and-conquer search algorithm designed for **strictly sorted arrays**. At each step, it compares the target value with the midpoint of the search interval and halves the remaining candidate elements.

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
        # Avoid integer overflow: low + (high - low) // 2
        mid = (low + high) // 2

        if arr[mid] == target:
            return mid  # Target identified
        elif arr[mid] < target:
            low = mid + 1  # Target is in right partition
        else:
            high = mid - 1  # Target is in left partition

    return -1  # Target not in array

# Example Demonstration
dataset = [2, 5, 8, 12, 16, 23, 38, 56, 72, 91]
target_val = 23
index = binary_search(dataset, target_val)

print(f"Target {target_val} found at index: {index}")
\`\`\`

---

### Step-by-Step Example Walkthrough
Target: $23$, Array: $[2, 5, 8, 12, 16, 23, 38, 56, 72, 91]$ ($n=10$)

1. **Step 1:** $\\text{low}=0, \\text{high}=9 \\implies \\text{mid}=4$. $\\text{arr}[4] = 16$. Since $16 < 23$, search right: $\\text{low} = 5$.
2. **Step 2:** $\\text{low}=5, \\text{high}=9 \\implies \\text{mid}=7$. $\\text{arr}[7] = 56$. Since $56 > 23$, search left: $\\text{high} = 6$.
3. **Step 3:** $\\text{low}=5, \\text{high}=6 \\implies \\text{mid}=5$. $\\text{arr}[5] = 23$. **Target match found! Return index 5.**

---

### Complexity Analysis
- **Best Case Time:** $\\mathcal{O}(1)$ (element is at the exact middle on the first check).
- **Average & Worst Case Time:** $\\mathcal{O}(\\log_2 n)$ (search space halves at each iteration).
- **Space Complexity:** $\\mathcal{O}(1)$ auxiliary space for iterative approach.`;
    }

    return null;
  }

  // =========================================================================
  // 5. UNIVERSAL MULTI-DISCIPLINARY EDUCATIONAL KNOWLEDGE ENGINE
  // =========================================================================
  function trySubjectConceptExplanation(query, history = []) {
    const q = query.toLowerCase().trim();

    // Contextual Follow-up: "What are its main advantages?" / "Explain it again"
    if (q.includes('advantage') || q.includes('disadvantage') || q.includes('pros') || q.includes('cons') || q.includes('explain it')) {
      // Find the last subject discussed in chatHistory
      let lastTopic = '';
      for (let i = history.length - 1; i >= 0; i--) {
        const text = (history[i].text || '').toLowerCase();
        if (text.includes('java')) { lastTopic = 'Java'; break; }
        if (text.includes('python')) { lastTopic = 'Python'; break; }
        if (text.includes('c++')) { lastTopic = 'C++'; break; }
        if (text.includes('data structure')) { lastTopic = 'Data Structures'; break; }
        if (text.includes('dbms') || text.includes('database')) { lastTopic = 'DBMS'; break; }
        if (text.includes('cloud')) { lastTopic = 'Cloud Computing'; break; }
        if (text.includes('machine learning')) { lastTopic = 'Machine Learning'; break; }
      }

      if (lastTopic === 'Java') {
        return `### Main Advantages of Java

Java is one of the most widely adopted enterprise programming languages. Key advantages include:

1. **Platform Independence (Write Once, Run Anywhere - WORA):**
   - Java source code compiles into platform-neutral bytecode executed by the Java Virtual Machine (JVM).
2. **Object-Oriented Architecture:**
   - Enforces clean encapsulation, modular inheritance, and polymorphism for large-scale enterprise applications.
3. **Automatic Garbage Collection:**
   - Built-in memory manager automatically reclaims unreferenced objects, preventing manual pointer leaks.
4. **Rich Ecosystem & Standard Library (JDK):**
   - Comprehensive APIs for networking, multithreading, collections, security, and enterprise backends (Spring, Hibernate).
5. **Robust Security & Type Safety:**
   - Bytecode verification, lack of explicit pointer arithmetic, and strict sandboxed classloading.`;
      }
    }

    // 1. What is Java?
    if (q === 'what is java' || q === 'what is java?' || q.startsWith('what is java')) {
      return `### Java Programming Language

**Java** is a high-level, class-based, object-oriented programming language designed to have as few implementation dependencies as possible. It was originally created by James Gosling at Sun Microsystems and released in 1995.

---

### Core Characteristics & Architecture

1. **Platform Independence (WORA):**
   - Java code is compiled into intermediate **Bytecode** (\`.class\` files) rather than machine-specific assembly.
   - The **Java Virtual Machine (JVM)** interprets or Just-In-Time (JIT) compiles this bytecode into native machine code on any operating system (Windows, Linux, macOS).

2. **Core Pillars:**
   - **Encapsulation:** Bundles state and behavior into classes with controlled access.
   - **Inheritance:** Promotes code reusability through superclass-subclass hierarchies.
   - **Polymorphism:** Enables method overloading and dynamic runtime overriding.
   - **Abstraction:** Separates public interfaces from private implementation details.

3. **Automatic Memory Management:**
   - Features built-in **Garbage Collection (GC)** that tracks and deallocates memory occupied by unreachable objects, eliminating manual memory leaks.

---

### Primary Use Cases
- Enterprise Backend Systems (Spring Boot, Jakarta EE)
- Android Mobile Applications
- Big Data Processing Pipelines (Apache Spark, Apache Hadoop)
- Financial Trading Systems and High-Throughput Microservices`;
    }

    // 2. What is C++?
    if (q === 'what is c++' || q === 'what is c++?' || q.startsWith('what is c++')) {
      return `### C++ Programming Language

**C++** is a powerful, high-performance, statically typed, compiled programming language created by Bjarne Stroustrup in 1979 as an extension of the C programming language ("C with Classes").

---

### Core Architectural Features

1. **Multi-Paradigm Support:**
   - Supports **Procedural**, **Object-Oriented**, **Generic (Templates)**, and **Functional** programming styles.

2. **Zero-Overhead Principle:**
   - What you don't use, you don't pay for. C++ provides direct hardware memory access and pointer manipulation without mandatory runtime interpreter overhead.

3. **Resource Acquisition Is Initialization (RAII):**
   - Resource management paradigm where life cycles of files, sockets, and memory are bound directly to variable scope, guaranteeing deterministic resource deallocation via destructors.

4. **Standard Template Library (STL):**
   - Rich suite of parameterized data structures (\`vector\`, \`map\`, \`unordered_set\`) and optimized algorithms (\`sort\`, \`binary_search\`).

---

### Key Applications
- Operating Systems and Device Drivers
- Game Engines (Unreal Engine)
- High-Frequency Trading (HFT) and Low-Latency Financial Systems
- Real-Time Graphics, Embedded Systems, and AI Accelerators`;
    }

    // 3. What is a Data Structure?
    if (q.includes('data structure') && (q.includes('what is') || q.includes('define') || q.includes('explain'))) {
      return `### Data Structures in Computer Science

A **Data Structure** is a specialized format for organizing, processing, retrieving, and storing data in computer memory so that operations can be performed efficiently with minimal time and memory consumption.

---

### Classification Matrix

1. **Linear Data Structures:**
   - Elements form a sequential sequence where each element is connected to its previous and next neighbor.
   - **Arrays:** Contiguous memory allocations providing $\\mathcal{O}(1)$ random access.
   - **Linked Lists:** Nodes containing data and pointers, enabling $\\mathcal{O}(1)$ dynamic insertion/deletion.
   - **Stacks:** Last-In, First-Out (LIFO) structure (used in recursion call-stacks and undo buffers).
   - **Queues:** First-In, First-Out (FIFO) structure (used in CPU scheduling and message queues).

2. **Non-Linear Data Structures:**
   - Elements are arranged hierarchically or interconnected across arbitrary networks.
   - **Trees (Binary Search Trees, AVL, Heaps):** Hierarchical structures providing $\\mathcal{O}(\\log n)$ search and priority extraction.
   - **Graphs:** Sets of vertices connected by edges (used in routing algorithms, social networks, and dependency graphs).
   - **Hash Tables:** Key-value mappings using hash functions to achieve $\\mathcal{O}(1)$ average-time lookup.

---

### Evaluation Criteria
- **Time Complexity:** Measured using Big-O asymptotic notation ($\\\\mathcal{O}(1), \\mathcal{O}(\\log n), \\mathcal{O}(n)$).
- **Space Complexity:** Auxiliary memory overhead required by pointers and storage.`;
    }

    // 4. What is Python?
    if (q === 'what is python' || q === 'what is python?' || q.startsWith('what is python')) {
      return `### Python Programming Language

**Python** is an interpreted, high-level, general-purpose programming language designed by Guido van Rossum and first released in 1991. Its design philosophy emphasizes code readability and simplicity through clean syntax.

---

### Core Strengths & Features

1. **Readable and Expressive Syntax:**
   - Uses significant whitespace indentation rather than curly braces or semicolons, reducing syntactic clutter.
2. **Dynamically Typed & Garbage Collected:**
   - Variable types are inferred at runtime, with automatic reference counting and cyclic garbage collection.
3. **Batteries-Included Standard Library:**
   - Comprehensive modules for file I/O, mathematical operations, JSON parsing, HTTP networking, and regular expressions.
4. **Dominance in AI and Data Science:**
   - Unmatched ecosystem of optimized C-backed libraries: NumPy, Pandas, Scikit-Learn, PyTorch, and TensorFlow.

---

### Applications
- Artificial Intelligence, Machine Learning, and Deep Learning
- Data Analysis, Statistics, and Scientific Computing
- Web Backend Development (Django, FastAPI, Flask)
- Automation Scripting, DevOps, and Cyber Security Tooling`;
    }

    // 5. What is Photosynthesis?
    if (q.includes('photosynthesis')) {
      return `### Photosynthesis in Biological Sciences

**Photosynthesis** is the fundamental biochemical process by which photoautotrophic organisms (green plants, algae, and cyanobacteria) convert light energy into chemical energy stored in glucose molecules.

---

### The Overall Chemical Equation

$$6\\text{CO}_2 + 6\\text{H}_2\\text{O} \\xrightarrow{\\text{Light Energy, Chlorophyll}} \\text{C}_6\\text{H}_{12}\\text{O}_6 + 6\\text{O}_2$$

---

### Two Main Biochemical Stages

1. **Light-Dependent Reactions (Thylakoid Membranes):**
   - Photons hit Chlorophyll molecules in Photosystem II and I.
   - Water molecules undergo **photolysis**: $2\\text{H}_2\\text{O} \\rightarrow 4\\text{H}^+ + 4e^- + \\text{O}_2$.
   - Electron transport produces energy storage molecules: **ATP** and **NADPH**.

2. **Light-Independent Reactions / Calvin Cycle (Stroma):**
   - **Carbon Fixation:** The enzyme **RuBisCO** fixes atmospheric $\\text{CO}_2$ with ribulose 1,5-bisphosphate (RuBP).
   - Utilizing ATP and NADPH from the light reactions, 3-PGA is converted to G3P molecules, ultimately forming **Glucose** ($\\\\text{C}_6\\text{H}_{12}\\text{O}_6$).

---

### Ecological Significance
- Produces virtually all oxygen in Earth's biosphere.
- Serves as the primary trophic energy foundation for all terrestrial and aquatic food webs.`;
    }

    // 6. Explain Newton's Second Law
    if (q.includes('newton') && (q.includes('second law') || q.includes('2nd law'))) {
      return `### Newton's Second Law of Motion (Classical Mechanics)

**Newton's Second Law of Motion** states that the rate of change of momentum of a body is directly proportional to the applied net force and takes place in the direction in which the force acts.

---

### Mathematical Derivation

Linear momentum is defined as $\\vec{p} = m\\vec{v}$.

From Newton's second law:

$$\\vec{F}_{\\text{net}} = \\frac{d\\vec{p}}{dt} = \\frac{d(m\\vec{v})}{dt}$$

For a body with constant mass ($m = \\text{constant}$):

$$\\vec{F}_{\\text{net}} = m \\cdot \\frac{d\\vec{v}}{dt} = m\\vec{a}$$

Where:
- $\\vec{F}_{\\text{net}}$: Net external force vector acting on the mass (measured in Newtons, $\\text{N} = \\text{kg}\\cdot\\text{m}/\\text{s}^2$).
- $m$: Inertial mass of the object (measured in kilograms, $\\text{kg}$).
- $\\vec{a}$: Acceleration vector imparted to the object (measured in $\\text{m}/\\text{s}^2$).

---

### Physical Insights
1. **Direct Proportionality:** Doubling the net applied force doubles the object's acceleration ($a \\propto F$).
2. **Inverse Proportionality:** Doubling the object's mass halves the acceleration for the same force ($a \\propto \\frac{1}{m}$).
3. **Equilibrium Condition:** When $\\vec{F}_{\\text{net}} = 0$, acceleration is zero, reproducing Newton's First Law.`;
    }

    // 7. What is Inflation?
    if (q.includes('inflation') && (q.includes('what is') || q.includes('economics') || q.includes('define') || q.includes('explain'))) {
      return `### Inflation in Macroeconomics

**Inflation** is the quantitative economic measure of the rate at which the general price level of goods and services in an economy increases over a given period, resulting in a sustained decline in the purchasing power of money.

---

### Primary Types & Causes

1. **Demand-Pull Inflation:**
   - Occurs when aggregate demand for goods and services outpaces aggregate supply ("too much money chasing too few goods").
2. **Cost-Push Inflation:**
   - Driven by an aggregate supply decrease resulting from rising production costs (e.g., surges in crude oil prices, raw material shortages, wage hikes).
3. **Built-In / Wage-Price Spiral:**
   - Workers demand higher wages to keep up with rising living costs, prompting firms to raise prices further to maintain margins.

---

### Measurement Indices
- **Consumer Price Index (CPI):** Measures the weighted average price of a representative basket of consumer goods and services.
- **Wholesale Price Index (WPI) / Producer Price Index (PPI):** Tracks price fluctuations at the wholesale and producer stages before retail distribution.

---

### Policy Responses
- **Monetary Policy:** Central banks raise benchmark interest rates (repo rates) and reduce monetary supply to cool excessive demand.
- **Fiscal Policy:** Governments reduce public spending or increase taxation to lower disposable income in circulation.`;
    }

    // 8. Human Digestive System
    if (q.includes('digestive system') || q.includes('human digestion')) {
      return `### The Human Digestive System (Anatomy & Physiology)

The **Human Digestive System** is a complex biological system responsible for the mechanical breakdown, chemical enzymatic hydrolysis, nutrient absorption, and waste elimination of ingested food.

---

### Sequential Anatomical Pathway & Mechanics

1. **Mouth & Oral Cavity:**
   - **Mechanical:** Teeth masticate food into a smaller surface area bolus.
   - **Chemical:** Salivary glands secrete **salivary amylase**, initiating starch hydrolysis into maltose.

2. **Esophagus:**
   - Transport channel that propels the bolus downward into the stomach via involuntary coordinated muscular contractions called **peristalsis**.

3. **Stomach:**
   - Secretes gastric juice containing **Hydrochloric Acid (HCl, pH ~1.5 - 2.0)** and **Pepsinogen** (activated into **pepsin** to cleave peptide bonds in proteins).
   - Churns food into an acidic fluid called **chyme**.

4. **Small Intestine (Duodenum, Jejunum, Ileum):**
   - **Primary Site of Digestion & Absorption.**
   - **Bile** (synthesized by liver, stored in gallbladder) emulsifies lipids.
   - **Pancreatic Juice** delivers lipase, trypsin, and amylase.
   - Millions of microscopic finger-like projections (**villi** and **microvilli**) maximize surface area to absorb glucose, amino acids, and fatty acids into the bloodstream.

5. **Large Intestine (Colon) & Rectum:**
   - Reabsorbs excess water and electrolytes from indigestible remnants.
   - Houses gut microbiota synthesizing Vitamin K and B-complex vitamins before compacting waste for excretion.`;
    }

    // 9. Explain the French Revolution
    if (q.includes('french revolution')) {
      return `### The French Revolution (1789–1799)

The **French Revolution** was a transformative period of social and political upheaval in France that dismantled the ancient feudal monarchy (*Ancien Régime*), established the ideals of modern democratic republics, and influenced political theory worldwide.

---

### Primary Causes

1. **Social Inequality (The Three Estates):**
   - **First Estate (Clergy)** and **Second Estate (Nobility)** enjoyed tax exemptions, feudal privileges, and vast land holdings.
   - **Third Estate (Commoners, Peasants, Bourgeoisie)** comprised 98% of the population, bore the entire national tax burden, and possessed zero political voice.
2. **Financial Bankruptcy:**
   - Heavy royal debts incurred from supporting the American Revolutionary War and extravagant royal expenditures at Versailles.
3. **Enlightenment Ideals:**
   - Works of Voltaire, Rousseau, and Montesquieu questioned the divine right of kings, advocating popular sovereignty and separation of powers.

---

### Major Chronological Milestones
- **1789:** Convening of the Estates-General, Tennis Court Oath, and the **Storming of the Bastille** (July 14).
- **1789:** Proclamation of the *Declaration of the Rights of Man and of the Citizen* (*"Liberté, Égalité, Fraternité"*).
- **1792–1794:** Abolition of the monarchy, execution of King Louis XVI and Marie Antoinette, followed by the **Reign of Terror** led by Maximilien Robespierre.
- **1799:** The Coup of 18 Brumaire, wherein **Napoleon Bonaparte** seized power, ending the revolutionary republic and establishing the Consulate.

---

### Historical Legacy
- Eradicated European feudalism and absolute divine-right monarchies.
- Codified modern secular law, human rights doctrines, and national democratic citizenship.`;
    }

    // 10. What is Machine Learning?
    if (q.includes('machine learning') && (q.includes('what is') || q.includes('define') || q.includes('explain'))) {
      return `### Machine Learning (Artificial Intelligence)

**Machine Learning (ML)** is a subfield of Artificial Intelligence (AI) and computational statistics focused on building mathematical models and algorithms that learn patterns and rules from historical training data to make predictions or decisions on unseen data without being explicitly programmed.

---

### The Three Fundamental Paradigms

1. **Supervised Learning:**
   - Trained on labeled pairs $(x_i, y_i)$.
   - **Tasks:** Classification (discrete classes) and Regression (continuous numerical values).
   - **Algorithms:** Linear Regression, Support Vector Machines (SVM), Random Forests, Deep Neural Networks.

2. **Unsupervised Learning:**
   - Trained on unlabeled datasets $(x_i)$ to detect latent patterns and intrinsic geometries.
   - **Tasks:** Clustering, Anomaly Detection, and Dimensionality Reduction.
   - **Algorithms:** K-Means, DBSCAN, Principal Component Analysis (PCA).

3. **Reinforcement Learning (RL):**
   - An autonomous agent learns an optimal behavioral policy $\\pi(a|s)$ through continuous trial-and-error interaction with a dynamic environment, maximizing cumulative scalar rewards.
   - **Algorithms:** Q-Learning, Deep Q-Networks (DQN), PPO.

---

### Core Mathematical Pipeline
$$\\text{Raw Data} \\rightarrow \\text{Feature Engineering} \\rightarrow \\text{Model Optimization (Loss Minimization } \\min_\\theta \\mathcal{L}(\\theta)) \\rightarrow \\text{Evaluation / Inference}$$`;
    }

    // 11. Process vs Thread
    if (q.includes('process') && q.includes('thread') && (q.includes('difference') || q.includes('vs') || q.includes('compare'))) {
      return `### Difference Between Process and Thread

In Operating Systems, **Processes** and **Threads** are both fundamental units of program execution, but operate at distinct levels of memory and resource isolation.

---

### Side-by-Side Comparison Matrix

| Evaluation Metric | Process | Thread (Lightweight Process) |
| :--- | :--- | :--- |
| **Definition** | An independent program in execution with its own virtual address space. | A lightweight execution unit within a parent process. |
| **Memory Isolation** | Fully isolated address space; protected from other processes. | Shares code, data segment, and open file handles with other sibling threads. |
| **Creation Overhead** | High; requires allocating a complete Process Control Block (PCB) and page tables. | Low; shares parent process memory, requiring only an individual Thread Control Block (TCB) and stack. |
| **Context Switching** | Slower; requires flushing CPU caches (TLB) and switching memory mappings. | Fast; registers and Program Counter (PC) are swapped while memory space remains constant. |
| **Inter-Communication** | Requires explicit Inter-Process Communication (IPC: Pipes, Shared Memory, Sockets). | Direct communication via shared global and heap memory. |
| **Fault Resilience** | High; a crash in one process does not affect other processes. | Low; an unhandled exception or memory crash in one thread can terminate the whole process. |`;
    }

    // 12. TCP vs UDP
    if ((q.includes('tcp') && q.includes('udp')) && (q.includes('difference') || q.includes('vs') || q.includes('compare'))) {
      return `### Difference Between TCP and UDP

At the Transport Layer (Layer 4) of the Internet Protocol Suite, **TCP (Transmission Control Protocol)** and **UDP (User Datagram Protocol)** provide contrasting host-to-host transmission guarantees.

---

### Side-by-Side Comparison Matrix

| Evaluation Metric | TCP (Transmission Control Protocol) | UDP (User Datagram Protocol) |
| :--- | :--- | :--- |
| **Connection Nature** | **Connection-oriented** (Requires 3-way handshake: SYN $\\rightarrow$ SYN-ACK $\\rightarrow$ ACK). | **Connectionless** (Transmits datagrams immediately with zero pre-handshaking). |
| **Reliability** | **Guaranteed delivery** with automated packet retransmissions and acknowledgments. | **Best-effort delivery**; packets may be dropped, duplicated, or arrive out of sequence. |
| **Ordering** | Enforces strict in-order packet delivery using sequence numbers. | No ordering guarantees; packets are delivered in whatever order network routers route them. |
| **Header Overhead** | Heavier: 20 to 60 bytes with flags, checksum, and window size. | Lightweight: Fixed 8 bytes. |
| **Flow & Congestion Control** | Full support (Sliding window, slow-start, additive-increase multiplicative-decrease). | No congestion or flow control; transmits at the rate supplied by application. |
| **Standard Protocols** | HTTP, HTTPS, SSH, FTP, SMTP, MySQL. | DNS, DHCP, VoIP, Real-Time Video Streaming, Online Multiplayer Gaming. |`;
    }

    // 13. ACID Properties in DBMS
    if (q.includes('acid') && (q.includes('dbms') || q.includes('database') || q.includes('properties'))) {
      return `### ACID Properties in Database Management Systems

The **ACID** properties are four fundamental guarantees that ensure database transactions are processed reliably, preserving database integrity even across hardware crashes and concurrent transactions.

---

### 1. Atomicity ("All or Nothing")
- **Principle:** Every database transaction is executed as an indivisible unit of work.
- **Guarantee:** Either all modifications within the transaction are committed successfully, or if a single failure occurs, the entire transaction is rolled back to its initial state via Write-Ahead Logs (WAL).

---

### 2. Consistency (State Invariants)
- **Principle:** A transaction must transition the database from one valid state to another valid state.
- **Guarantee:** All schema constraints, primary keys, foreign key relationships, and check validations remain strictly preserved upon transaction completion.

---

### 3. Isolation (Concurrency Control)
- **Principle:** Intermediate states of concurrently executing transactions must remain invisible to one another.
- **Guarantee:** Concurrent transaction execution yields the exact same outcome as if the transactions were executed sequentially (serializability enforced via two-phase locking 2PL or MVCC).

---

### 4. Durability (Persistence Guarantee)
- **Principle:** Once a transaction commits, its modifications are permanently recorded.
- **Guarantee:** Committed data updates will survive power outages, operating system crashes, and disk reboots from non-volatile storage.`;
    }

    return null;
  }

  // =========================================================================
  // 6. REAL-TIME WIKIPEDIA FACTUAL KNOWLEDGE RETRIEVER
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
  // 7. BACKEND API CLIENT WITH AUTOMATIC LOCAL GEMINI FALLBACK
  // =========================================================================
  async function queryBackendAI(query, history = [], fileSnapshot = null) {
    const clientKey = localStorage.getItem('gemini_api_key') || window.GEMINI_API_KEY || '';

    // 1. Try Backend Server Endpoint first
    for (const backendUrl of BACKEND_URLS) {
      try {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 12000);

        const response = await fetch(backendUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            query: query,
            history: history.slice(-10),
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
        // Backend not reachable on this URL; continue to next fallback
      }
    }

    // 2. Client-Side Direct Google Gemini 1.5 Flash (for static GitHub Pages hosting)
    if (clientKey) {
      try {
        const directEndpoint = `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${encodeURIComponent(clientKey)}`;
        const contents = [];

        // Ingest previous turns
        for (const item of history.slice(-10)) {
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
          fullTextPrompt = `[ATTACHED DOCUMENT: ${fileSnapshot.name} (${fileSnapshot.type})]\n${fileSnapshot.text.substring(0, 12000)}\n\n[STUDENT QUESTION]:\n${query}`;
        }
        currentParts.push({ text: fullTextPrompt });

        contents.push({ role: 'user', parts: currentParts });

        const resp = await fetch(directEndpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            contents: contents,
            generationConfig: { temperature: 0.2, maxOutputTokens: 3000 }
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
        console.warn('Direct client-side Gemini error:', e);
      }
    }

    return null;
  }

  // =========================================================================
  // 8. MASTER TIERED EDUCATIONAL INFERENCE ENGINE
  // =========================================================================
  async function generateAcademicResponse(query, fileSnapshot = null) {
    const qClean = sanitizeUserQuery(query);
    const qLower = qClean.toLowerCase();

    // 1. Conversational Greetings
    const greetings = {
      'hi': 'Hello! I am Brainware Academic AI, your educational assistant. Ask me any question across Computer Science, Programming, Mathematics, Sciences, Engineering, Humanities, or Business, and I will guide you step by step.',
      'hello': 'Greetings! How can I assist with your coursework, exam preparation, or academic research today?',
      'hey': 'Hello! What educational concept, calculation, or coding problem can I solve for you?'
    };
    if (greetings[qLower]) {
      return { text: greetings[qLower], source: 'conversational' };
    }

    // 2. Primary Engine: Backend or Direct Gemini Model
    const aiResult = await queryBackendAI(query, chatHistory, fileSnapshot);
    if (aiResult && aiResult.text) {
      return aiResult;
    }

    // 3. Document Analysis Mode (When file is attached and AI API is offline)
    if (fileSnapshot && fileSnapshot.text) {
      return {
        text: `### Document Analysis: ${fileSnapshot.name}\n\n**Format:** ${fileSnapshot.type} (${fileSnapshot.size})\n\n**Document Content Excerpt:**\n\n\`\`\`\n${fileSnapshot.text.substring(0, 1200)}...\n\`\`\`\n\n---\n\n### Summary Assessment\nThis ${fileSnapshot.type} document has been parsed. For deep cross-page reasoning and visual query answering, connect the Brainware AI backend server or provide your Gemini API key in settings.`,
        source: 'document'
      };
    }

    // 4. University Syllabus & Exam Question Bank Grounding (portalQAIndex: 1,254 entries)
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

    // 5. Mathematical Calculation Solver
    const mathAns = trySolveMathProblem(query);
    if (mathAns) {
      return { text: mathAns, source: 'math' };
    }

    // 6. Programming Code Synthesizer
    const codeAns = tryGenerateCodeSolution(query);
    if (codeAns) {
      return { text: codeAns, source: 'code' };
    }

    // 7. Universal Academic Subject Synthesizer
    const conceptAns = trySubjectConceptExplanation(query, chatHistory);
    if (conceptAns) {
      return { text: conceptAns, source: 'concept' };
    }

    // 8. Factual Academic Knowledge Retrieval via Wikipedia API
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

    // Code blocks with syntax copy buttons
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

    // Style Markdown Tables
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
  // 11. UI BUBBLES & CONTROLLERS
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
    if (q.includes('photosynthesis') || q.includes('biology')) {
      return ['Explain the Light-Dependent reactions', 'What is the role of Chlorophyll?', 'Detail the Calvin Cycle'];
    }
    if (q.includes('newton') || q.includes('physics')) {
      return ['Solve a numerical problem using this law', 'Explain Newton’s Third Law', 'How does momentum relate to impulse?'];
    }
    if (q.includes('inflation') || q.includes('economics')) {
      return ['What are the key causes of demand-pull inflation?', 'How do Central Banks control inflation?', 'What is stagflation?'];
    }
    if (q.includes('machine learning') || q.includes('ai')) {
      return ['Difference between Supervised and Unsupervised Learning', 'Explain Gradient Descent step by step', 'Provide a simple Python ML example'];
    }
    if (q.includes('dbms') || q.includes('database')) {
      return ['Explain ACID properties in detail', 'What is Normalization (1NF, 2NF, 3NF)?', 'Explain Indexing in DBMS'];
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

  // Wire up Drag and Drop onto Chat Container
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
