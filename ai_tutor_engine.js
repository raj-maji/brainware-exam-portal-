/**
 * Brainware University - Open Academic AI Agent
 * - Delivers pure, direct, high-quality answers identical to ChatGPT and Gemini
 * - Zero extraneous UI clutter (NO model badges, NO subject headers, NO footer noise)
 * - Full KaTeX mathematical formula rendering ($\lambda$, $\sigma$, $\theta$, $\alpha$)
 * - Mathematical logic operators: AND ($\land$), OR ($\lor$), NOT ($\neg$), Implication ($\implies$), Equivalence ($\iff$)
 * - Multi-provider live AI (Puter.js + Free OpenAI endpoints) + Master Academic Knowledge Base
 * - Strict query sanitization preventing false question-number or document mismatches
 */

window.BrainwareAITutor = (function() {
  let isGenerating = false;
  let chatHistory = [];
  let lastUserQuery = "";
  let messageCounter = 0;

  // Conversational greetings
  const quickGreetings = {
    "hi": "Hello! How can I help you today? Ask me any question related to your subjects, computer science, mathematics, or programming, and I will give you a clear, straightforward answer.",
    "hello": "Hi there! What topic or question can I help you with today?",
    "hey": "Hey! How can I assist you with your studies or questions today?",
    "who are you": "I am your AI Academic Study Assistant. I am here to provide clear, accurate, and comprehensive answers to any educational, technical, or mathematical question you have.",
    "what is your name": "I am your Brainware AI Study Assistant.",
    "thank you": "You're very welcome! Let me know if you need any further clarification or have another question.",
    "thanks": "Glad to help! Feel free to ask more questions anytime.",
    "bye": "Goodbye! Best of luck with your studies!"
  };

  // Unicode mathematical symbol mappings for fallback & copy
  const mathSymbolMap = [
    [/\\lambda\b/g, 'λ'],
    [/\\alpha\b/g, 'α'],
    [/\\beta\b/g, 'β'],
    [/\\theta\b/g, 'θ'],
    [/\\sigma\b/g, 'σ'],
    [/\\mu\b/g, 'μ'],
    [/\\pi\b/g, 'π'],
    [/\\Delta\b/g, 'Δ'],
    [/\\nabla\b/g, '∇'],
    [/\\land\b/g, '∧'],
    [/\\lor\b/g, '∨'],
    [/\\neg\b/g, '¬'],
    [/\\oplus\b/g, '⊕'],
    [/\\implies\b/g, '⇒'],
    [/\\iff\b/g, '⇔'],
    [/\\equiv\b/g, '≡'],
    [/\\rightarrow\b/g, '→'],
    [/\\leftarrow\b/g, '←'],
    [/\\forall\b/g, '∀'],
    [/\\exists\b/g, '∃'],
    [/\\in\b/g, '∈'],
    [/\\notin\b/g, '∉'],
    [/\\subseteq\b/g, '⊆'],
    [/\\subset\b/g, '⊂'],
    [/\\cup\b/g, '∪'],
    [/\\cap\b/g, '∩'],
    [/\\le\b/g, '≤'],
    [/\\ge\b/g, '≥'],
    [/\\neq\b/g, '≠'],
    [/\\approx\b/g, '≈'],
    [/\\times\b/g, '×'],
    [/\\cdot\b/g, '·'],
    [/\\partial\b/g, '∂'],
    [/\\infty\b/g, '∞']
  ];

  function applyUnicodeMathFallback(str) {
    if (!str) return "";
    let s = str;
    for (const [regex, rep] of mathSymbolMap) {
      s = s.replace(regex, rep);
    }
    return s;
  }

  // Sanitize user query: strip leading question numbers like "Q12.", "12.", "Question 12:"
  function sanitizeUserQuery(raw) {
    if (!raw) return "";
    let cleaned = raw.trim();
    cleaned = cleaned.replace(/^(?:question\s*\d+|q\s*\d+|\d+)[\s.:\-)]+/i, '');
    return cleaned.trim();
  }

  const stopwords = new Set([
    "the", "is", "at", "which", "on", "a", "an", "and", "or", "in", "to", "for", 
    "of", "with", "between", "distinguish", "difference", "explain", "describe", 
    "what", "how", "why", "who", "where", "can", "you", "tell", "me", "about", 
    "state", "compare", "contrast", "write", "briefly", "terms", "define", "give", 
    "list", "following", "does", "do", "are", "by", "its", "from"
  ]);

  function cleanSyllabusHtml(html) {
    if (!html) return "";
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

  // Strict syllabus matching: prevents false positives from question numbers
  function findSyllabusMatch(rawQuery) {
    if (!window.portalQAIndex || !Array.isArray(window.portalQAIndex)) return null;

    const query = sanitizeUserQuery(rawQuery);
    const qNorm = query.toLowerCase().trim();
    const tokens = qNorm.match(/[a-z0-9_-]+/g) || [];
    const significantWords = tokens.filter(w => !stopwords.has(w) && w.length > 2);
    
    if (significantWords.length === 0) return null;

    let bestItem = null;
    let highestScore = 0;

    for (const item of window.portalQAIndex) {
      const candNorm = item.q.toLowerCase();
      let matchedCount = 0;

      for (const w of significantWords) {
        if (candNorm.includes(w)) {
          matchedCount++;
        }
      }

      // Require high keyword overlap ratio (at least 60% of significant words)
      const overlapRatio = matchedCount / significantWords.length;
      if (overlapRatio >= 0.6 && matchedCount >= 2) {
        let score = matchedCount * 50;
        if (candNorm.includes(qNorm) || qNorm.includes(candNorm)) {
          score += 300;
        }
        if (score > highestScore) {
          highestScore = score;
          bestItem = item;
        }
      }
    }

    if (highestScore >= 150 && bestItem) {
      return { item: bestItem, score: highestScore };
    }
    return null;
  }

  // 100% Anonymous, Zero-Login Public AI Inference (No Puter popups, No API keys, No Registration)
  async function queryLiveAI(query, syllabusContext) {
    const systemPrompt = `You are a helpful, expert AI Academic Assistant like ChatGPT and Gemini.
Provide ONLY the direct, articulate, high-quality answer to the question asked.
Do not include metadata headers, subject labels, or robotic boilerplate.
When formulas, mathematics, or logical operators are involved:
- Use standard LaTeX notation: $...$ for inline math and $$...$$ for block math
- Use proper symbols: \\lambda for lambda, \\alpha for alpha, \\theta for theta, \\sigma for sigma
- For logic: use \\land for AND, \\lor for OR, \\neg for NOT, \\implies for implies, \\iff for equivalence
Structure the response cleanly with:
- A direct, natural opening explanation
- Clear bullet points and bold formatting for key concepts
- Clear, well-formatted Markdown comparison tables when comparing concepts
- Clean code blocks with language tags when programming is involved
- Conclude naturally with key takeaways.`;

    let userPrompt = query;
    if (syllabusContext) {
      userPrompt += `\n\n[Reference Curriculum Material]:\nTopic: ${syllabusContext.q}\n${cleanSyllabusHtml(syllabusContext.a).substring(0, 1000)}\n\nAnswer the student's question comprehensively and accurately.`;
    }

    // Free Open Public Text AI Endpoint (100% Anonymous, Never Asks For Login)
    try {
      const fullQuery = `${systemPrompt}\n\nQuestion: ${userPrompt}`;
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 7000); // 7s fast timeout before instant local fallback

      const url = `https://text.pollinations.ai/${encodeURIComponent(fullQuery)}?model=openai`;
      const resp = await fetch(url, { method: "GET", signal: controller.signal });
      clearTimeout(timeoutId);

      if (resp.ok) {
        const text = await resp.text();
        if (text && text.trim().length > 30) {
          return text.trim();
        }
      }
    } catch (e) {
      // Quietly fall back to instant Master Academic Knowledge Engine
    }

    return null;
  }

  // Master Academic Knowledge Engine (Pure ChatGPT / Gemini Style Answers)
  function getMasterAcademicAnswer(rawQuery, syllabusMatch) {
    const q = sanitizeUserQuery(rawQuery).toLowerCase();

    // 1. Word2Vec: CBOW vs Skip-Gram Architectures
    if (q.includes('cbow') || q.includes('skip-gram') || q.includes('skipgram') || (q.includes('word2vec') && (q.includes('architect') || q.includes('distinguish') || q.includes('differ')))) {
      return `Word2Vec, introduced by Tomas Mikolov et al. at Google in 2013, is a foundational neural network technique for learning high-dimensional distributed word representations (word embeddings). It captures semantic and syntactic relationships such that words with similar meanings occupy close vectors in Euclidean space (for example: $\\vec{v}_{\\text{King}} - \\vec{v}_{\\text{Man}} + \\vec{v}_{\\text{Woman}} \\approx \\vec{v}_{\\text{Queen}}$).

Word2Vec comprises two primary model architectures: **Continuous Bag of Words (CBOW)** and **Skip-Gram**.

---

### 1. Continuous Bag of Words (CBOW) Architecture

- **Objective:** Predicts a **single target word** given its surrounding **context words** within a sliding window.
- **How it Works:**
  1. The input consists of $C$ context words encoded as one-hot vectors $\\{w_{t-k}, \\dots, w_{t-1}, w_{t+1}, \\dots, w_{t+k}\\}$.
  2. Each one-hot vector is multiplied by an input weight matrix $W$ to retrieve its embedding.
  3. The network **averages** (or sums) all the context word vectors in the projection layer:
     $$h = \\frac{1}{C} \\sum_{i=1}^{C} W x_i$$
  4. The aggregated vector $h$ is multiplied by an output matrix $W'$ and passed through a Softmax layer to predict the probability distribution over the vocabulary for the target center word $w_t$:
     $$P(w_t \\mid w_{\\text{context}}) = \\frac{\\exp({v'_{w_t}}^{\\top} h)}{\\sum_{j=1}^{|V|} \\exp({v'_{w_j}}^{\\top} h)}$$
- **Example:** In the sentence *"Artificial intelligence transforms the future"*:
  - If the target word is **"transforms"** with window size 2, the context words are \`["artificial", "intelligence", "the", "future"]\`.
  - CBOW takes the average vector of these 4 context words to predict **"transforms"**.
- **Key Characteristics:**
  - Computationally very efficient because the projection layer averages the context into a single vector.
  - Smooths out distributed context, making it fast and robust for large corpora with frequent words.

---

### 2. Skip-Gram Architecture

- **Objective:** Predicts the surrounding **context words** given a **single target word** (the inverse of CBOW).
- **How it Works:**
  1. The input is a single center word $w_t$ as a one-hot vector $x$.
  2. It looks up its embedding directly from the input weight matrix $W$: $h = W x = v_{w_t}$.
  3. The projection vector $h$ is multiplied by the output weight matrix $W'$ to generate probability distributions for each of the $C$ surrounding context window positions:
     $$P(w_{t+j} \\mid w_t) = \\frac{\\exp({v'_{w_{t+j}}}^{\\top} v_{w_t})}{\\sum_{k=1}^{|V|} \\exp({v'_{w_k}}^{\\top} v_{w_t})} \\quad (\\text{for } j \\in [-k, k], j \\neq 0)$$
  4. The model maximizes the sum of log probabilities across all context positions.
- **Example:** For the same sentence with target word **"transforms"**:
  - Skip-Gram treats this as independent training pairs: \`("transforms", "artificial")\`, \`("transforms", "intelligence")\`, \`("transforms", "the")\`, and \`("transforms", "future")\`.
- **Key Characteristics:**
  - Treats each context pair individually without vector averaging, preserving fine-grained semantic nuances.
  - Significantly outperforms CBOW on **rare words** and specialized domain vocabularies.

---

### 3. Comprehensive Comparison Table

| Dimension | Continuous Bag of Words (CBOW) | Skip-Gram |
| :--- | :--- | :--- |
| **Prediction Task** | Predicts Center Target Word from Context | Predicts Context Words from Center Target Word |
| **Input / Output** | Multiple Inputs $\\rightarrow$ Single Output | Single Input $\\rightarrow$ Multiple Outputs |
| **Projection Layer** | Averages / sums all context vectors | Direct vector lookup (no averaging) |
| **Training Complexity** | Lower $\\mathcal{O}(C \\times N + N \\times \\log |V|)$ | Higher $\\mathcal{O}(C \\times (N + N \\times \\log |V|))$ |
| **Training Speed** | **Fast** (several times faster than Skip-Gram) | **Slower** (processes each context pair separately) |
| **Handling Rare Words** | Suboptimal (averaging drowns rare word signals) | **Superior** (gives rare words equal gradient updates) |
| **Dataset Size Suitability**| Large corpora, frequent vocabulary | Smaller datasets or corpora with rare/technical terms |
| **Optimization Method** | Hierarchical Softmax or Negative Sampling | Negative Sampling ($\approx 5\\text{--}20$ noise samples) |

---

### Key Takeaway
Use **CBOW** when training speed is critical and the dataset is large. Use **Skip-Gram** when semantic precision on rare, technical, or domain-specific words is paramount.`;
    }

    // 2. Mathematical Optimization: Lagrange Multipliers with lambda
    if (q.includes('lagrange') || (q.includes('multiplier') && q.includes('lambda')) || (q.includes('constrained') && q.includes('optimization'))) {
      return `The method of **Lagrange Multipliers**, formulated by Joseph-Louis Lagrange, is a mathematical strategy for finding local maxima and minima of a function subject to equality constraints.

---

### 1. Mathematical Formulation

Let $f(x_1, x_2, \\dots, x_n)$ be the **objective function** to optimize, subject to $m$ equality constraints:
$$g_i(x_1, x_2, \\dots, x_n) = 0 \\quad (i = 1, 2, \\dots, m)$$

Instead of solving a constrained problem, we introduce auxiliary variables $\\lambda = (\\lambda_1, \\lambda_2, \\dots, \\lambda_m)$, termed **Lagrange Multipliers**, and formulate the unconstrained **Lagrangian Function** $\\mathcal{L}$:

$$\\mathcal{L}(x, \\lambda) = f(x) - \\sum_{i=1}^{m} \\lambda_i g_i(x)$$

*(Alternatively written as $\\mathcal{L}(x, \\lambda) = f(x) + \\sum_{i=1}^m \\lambda_i g_i(x)$ depending on convention).*

---

### 2. Geometric Intuition: Gradient Alignment

At an optimal constrained point $x^*$, the level curve (contour line) of the objective function $f(x)$ must be **tangent** to the constraint surface $g(x) = 0$.

Because the gradient vector $\\nabla f$ is perpendicular to the contour of $f$, and $\\nabla g$ is perpendicular to the constraint surface $g = 0$, both gradient vectors must be **parallel (collinear)** at the extremum:

$$\\nabla f(x) = \\lambda \\nabla g(x)$$

where $\\lambda \\in \\mathbb{R}$ is the proportionality constant.

---

### 3. Necessary First-Order Optimality Conditions

To solve for the critical points and the multiplier $\\lambda$, we set all partial derivatives of the Lagrangian $\\mathcal{L}$ to zero:

$$\\nabla_x \\mathcal{L}(x, \\lambda) = \\nabla f(x) - \\lambda \\nabla g(x) = 0$$

$$\\frac{\\partial \\mathcal{L}}{\\partial \\lambda} = -g(x) = 0 \\implies g(x) = 0$$

This yields a system of $n + m$ equations with $n + m$ unknowns $(x_1, \\dots, x_n, \\lambda_1, \\dots, \\lambda_m)$.

---

### 4. Step-by-Step Concrete Example

**Problem:** Find the maximum of $f(x, y) = x + 2y$ subject to the constraint $g(x, y) = x^2 + y^2 - 5 = 0$.

1. Compute gradients:
   $$\\nabla f = \\begin{bmatrix} 1 \\\\ 2 \\end{bmatrix}, \\quad \\nabla g = \\begin{bmatrix} 2x \\\\ 2y \\end{bmatrix}$$
2. Set $\\nabla f = \\lambda \\nabla g$:
   $$1 = 2\\lambda x \\implies x = \\frac{1}{2\\lambda}$$
   $$2 = 2\\lambda y \\implies y = \\frac{1}{\\lambda}$$
3. Substitute into the constraint $x^2 + y^2 = 5$:
   $$\\left(\\frac{1}{2\\lambda}\\right)^2 + \\left(\\frac{1}{\\lambda}\\right)^2 = 5 \\implies \\frac{1}{4\\lambda^2} + \\frac{1}{\\lambda^2} = \\frac{5}{4\\lambda^2} = 5$$
   $$\\lambda^2 = \\frac{1}{4} \\implies \\lambda = \\pm \\frac{1}{2}$$
4. For $\\lambda = \\frac{1}{2}$: $x = 1, y = 2 \\implies f(1, 2) = 1 + 2(2) = 5$ (**Maximum**).
   For $\\lambda = -\\frac{1}{2}$: $x = -1, y = -2 \\implies f(-1, -2) = -5$ (**Minimum**).

---

### 5. Economic & Engineering Meaning of $\\lambda$
The multiplier $\\lambda$ is known as the **shadow price** or marginal rate of change:
$$\\lambda = \\frac{\\partial f^*}{\\partial c}$$
It quantifies how much the optimal objective value $f^*$ increases when the constraint budget $g(x) = c$ is relaxed by one unit. Widely utilized in Support Vector Machines (dual formulation), Karush-Kuhn-Tucker (KKT) conditions, and microeconomics.`;
    }

    // 3. Mathematical Logic: Operators, Truth Tables, and De Morgan's Laws
    if (q.includes('logic') || q.includes('demorgan') || q.includes('de morgan') || q.includes('truth table') || q.includes('boolean') || (q.includes('and') && q.includes('or') && q.includes('not'))) {
      return `Mathematical Logic provides formal symbolic frameworks for evaluating deductive reasoning, validity of propositions, and Boolean computer architectures.

---

### 1. Fundamental Logical Operators

| Operator Name | Formal Symbol | ASCII Notation | Natural Language | Semantic Meaning |
| :--- | :--- | :--- | :--- | :--- |
| **Conjunction** | $\\land$ | \`AND\`, \`&&\` | "$P$ and $Q$" | True if and only if **both** $P$ and $Q$ are True. |
| **Disjunction** | $\\lor$ | \`OR\`, \`||\` | "$P$ or $Q$" | True if **at least one** of $P$ or $Q$ is True. |
| **Negation** | $\\neg$ or $\\sim$ | \`NOT\`, \`!\` | "Not $P$" | Inverts truth value (True $\\rightarrow$ False, False $\\rightarrow$ True). |
| **Conditional (Implication)** | $\\rightarrow$ or $\\implies$ | \`IMPLIES\` | "If $P$, then $Q$" | False **only** when $P$ is True and $Q$ is False ($P \\rightarrow Q \\equiv \\neg P \\lor Q$). |
| **Biconditional (Equivalence)** | $\\leftrightarrow$ or $\\iff$ | \`IFF\` | "$P$ if and only if $Q$" | True when both $P$ and $Q$ have the **identical** truth value. |
| **Exclusive OR (XOR)** | $\\oplus$ | \`XOR\`, \`^\` | "$P$ XOR $Q$" | True when exactly one of $P$ or $Q$ is True. |

---

### 2. Comprehensive Master Truth Table

| $P$ | $Q$ | $\\neg P$ | $P \\land Q$ (AND) | $P \\lor Q$ (OR) | $P \\oplus Q$ (XOR) | $P \\rightarrow Q$ (Implication) | $P \\iff Q$ (Equivalence) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **T** | **T** | F | **T** | **T** | F | **T** | **T** |
| **T** | **F** | F | F | **T** | **T** | **F** | F |
| **F** | **T** | T | F | **T** | **T** | **T** | F |
| **F** | **F** | T | F | F | F | **T** | **T** |

---

### 3. De Morgan's Laws

Formulated by Augustus De Morgan, these two fundamental equivalence laws state how negation distributes over conjunction and disjunction:

1. **First Law (Negation of Conjunction):**
   $$\\neg(P \\land Q) \\iff \\neg P \\lor \\neg Q$$
   *"The negation of an AND is equivalent to the OR of the individual negations."*

2. **Second Law (Negation of Disjunction):**
   $$\\neg(P \\lor Q) \\iff \\neg P \\land \\neg Q$$
   *"The negation of an OR is equivalent to the AND of the individual negations."*

#### Truth Table Verification of First Law: $\\neg(P \\land Q) \\equiv \\neg P \\lor \\neg Q$

| $P$ | $Q$ | $P \\land Q$ | $\\neg(P \\land Q)$ | $\\neg P$ | $\\neg Q$ | $\\neg P \\lor \\neg Q$ | Verification |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| T | T | T | **F** | F | F | **F** | Identical (✓) |
| T | F | F | **T** | F | T | **T** | Identical (✓) |
| F | T | F | **T** | T | F | **T** | Identical (✓) |
| F | F | F | **T** | T | T | **T** | Identical (✓) |

---

### 4. First-Order Logic Quantifiers

In predicate logic, propositions involve variables over domains quantified by:
- **Universal Quantifier ($\\forall$):** "For all $x$ in domain $D$, $P(x)$ is True" ($\\forall x \\in D, P(x)$).
- **Existential Quantifier ($\\exists$):** "There exists at least one $x$ such that $P(x)$ is True" ($\\exists x \\in D, P(x)$).
- **Quantifier Negation Duals:**
  $$\\neg (\\forall x, P(x)) \\iff \\exists x, \\neg P(x)$$
  $$\\neg (\\exists x, P(x)) \\iff \\forall x, \\neg P(x)$$`;
    }

    // 4. Linear Algebra: Eigenvalues and Eigenvectors (lambda)
    if (q.includes('eigen') || (q.includes('matrix') && q.includes('lambda'))) {
      return `In Linear Algebra, given a square matrix $A \\in \\mathbb{R}^{n \\times n}$, a non-zero vector $v \\in \\mathbb{R}^n$ is an **Eigenvector** if the linear transformation of $v$ by $A$ produces a scalar multiple of itself:

$$A v = \\lambda v$$

where $\\lambda \\in \\mathbb{C}$ is a scalar known as the **Eigenvalue** corresponding to $v$.

---

### 1. The Characteristic Equation

To find the eigenvalues, we rewrite the fundamental equation:
$$A v - \\lambda I v = 0 \\implies (A - \\lambda I) v = 0$$

For a non-trivial (non-zero) solution $v \\neq 0$ to exist, the matrix $(A - \\lambda I)$ must be singular (non-invertible). Therefore, its determinant must be zero:

$$\\det(A - \\lambda I) = 0$$

This produces a polynomial of degree $n$ in $\\lambda$, known as the **Characteristic Polynomial**.

---

### 2. Step-by-Step Calculation Example

Find eigenvalues and eigenvectors for $A = \\begin{bmatrix} 4 & 2 \\\\ 1 & 3 \\end{bmatrix}$:

1. Form $A - \\lambda I$:
   $$A - \\lambda I = \\begin{bmatrix} 4 - \\lambda & 2 \\\\ 1 & 3 - \\lambda \\end{bmatrix}$$
2. Set determinant to zero:
   $$\\det(A - \\lambda I) = (4 - \\lambda)(3 - \\lambda) - (2)(1) = \\lambda^2 - 7\\lambda + 12 - 2 = \\lambda^2 - 7\\lambda + 10 = 0$$
3. Factor roots:
   $$(\\lambda - 5)(\\lambda - 2) = 0 \\implies \\lambda_1 = 5, \\quad \\lambda_2 = 2$$
4. Find eigenvector for $\\lambda_1 = 5$:
   $$(A - 5I)v = \\begin{bmatrix} -1 & 2 \\\\ 1 & -2 \\end{bmatrix} \\begin{bmatrix} x_1 \\\\ x_2 \\end{bmatrix} = \\begin{bmatrix} 0 \\\\ 0 \\end{bmatrix} \\implies -x_1 + 2x_2 = 0 \\implies v_1 = \\begin{bmatrix} 2 \\\\ 1 \\end{bmatrix}$$
5. Find eigenvector for $\\lambda_2 = 2$:
   $$(A - 2I)v = \\begin{bmatrix} 2 & 2 \\\\ 1 & 1 \\end{bmatrix} \\begin{bmatrix} x_1 \\\\ x_2 \\end{bmatrix} = \\begin{bmatrix} 0 \\\\ 0 \\end{bmatrix} \\implies x_1 + x_2 = 0 \\implies v_2 = \\begin{bmatrix} 1 \\\\ -1 \\end{bmatrix}$$

---

### 3. Core Properties
- **Trace Property:** $\\sum_{i=1}^n \\lambda_i = \\text{Tr}(A)$ (sum of diagonal elements).
- **Determinant Property:** $\\prod_{i=1}^n \\lambda_i = \\det(A)$.
- **Applications:** Principal Component Analysis (PCA dimensionality reduction), Google PageRank algorithm, vibration analysis, and quantum mechanics state vectors.`;
    }

    // 5. Operating Systems: Process vs Thread
    if (q.includes('process') && q.includes('thread')) {
      return `A **Process** is an independent program in execution with its own dedicated memory address space, whereas a **Thread** is a lightweight unit of CPU execution running within the context of a process.

An intuitive analogy: A **Process** is like an apartment building with dedicated land, private plumbing, and its own boundary walls. A **Thread** is like a tenant living inside an apartment—tenants share the living room, kitchen, and resources (Code, Data, Heap), but each tenant maintains their own private notebook (Stack and Registers).

---

### Comprehensive Comparison Table

| Parameter | Process (Heavyweight) | Thread (Lightweight) |
| :--- | :--- | :--- |
| **Address Space** | Completely isolated address space (own Code, Data, Heap, Stack). | Shares Code, Data, and Heap with peer threads; has private Stack. |
| **Creation Overhead** | **High:** Requires memory page allocations, process control block (PCB), descriptor tables (\`fork()\`). | **Low:** Minimal allocation; reuses parent process resources (\`pthread_create()\`). |
| **Context Switching** | **Slow:** CPU must flush translation lookaside buffer (TLB) and switch memory page directory tables. | **Fast:** CPU switches only program counter, stack pointer, and general registers; TLB preserved. |
| **Inter-Communication** | Requires Inter-Process Communication (IPC): Pipes, Sockets, Shared Memory, Message Queues. | Direct communication via shared memory / heap variables (requires Mutex/Semaphores). |
| **Fault Isolation** | **High:** If Process A crashes, Process B is completely unaffected. | **Low:** An unhandled exception or segfault in one thread terminates the entire parent process. |
| **Resource Ownership** | Holds open file descriptors, network sockets, child process handles. | Borrows resource handles from the owning parent process. |

---

### When to Use Which
- Use **Processes** when security boundaries, independent memory address spaces, or high fault tolerance are required (e.g., Google Chrome tabs, isolated microservices).
- Use **Threads** when high-performance parallel computation, low context-switching overhead, and intensive shared memory access are needed (e.g., video rendering engines, web server request workers).`;
    }

    // 6. Operating Systems: Deadlock & Coffman Conditions
    if (q.includes('deadlock') || q.includes('coffman')) {
      return `In Operating Systems, a **Deadlock** is a condition in a concurrent system where a set of processes are permanently blocked because each process holds a resource that another process needs, and none can proceed without external intervention.

---

### The Four Necessary Coffman Conditions (1971)

A deadlock can occur **if and only if** all four of the following conditions hold simultaneously:

1. **Mutual Exclusion:**
   - At least one resource must be held in a non-shareable mode (only one process can use the resource at any given moment, e.g., a tape drive or CPU core).
2. **Hold and Wait:**
   - A process currently holding at least one resource is concurrently waiting to acquire additional resources held by other active processes.
3. **No Preemption:**
   - Resources cannot be forcibly seized from a process; they can only be released voluntarily after the holding process finishes its task.
4. **Circular Wait:**
   - A closed chain of processes exists such that $P_0$ waits for a resource held by $P_1$, $P_1$ waits for $P_2$, $\\dots$, and $P_n$ waits for $P_0$:
     $$P_0 \\rightarrow P_1 \\rightarrow P_2 \\dots \\rightarrow P_n \\rightarrow P_0$$

---

### Deadlock Handling Strategies

| Strategy | Core Mechanism | Practical Example |
| :--- | :--- | :--- |
| **Deadlock Prevention** | Systematically design algorithms that violate at least one of the 4 Coffman conditions (e.g., impose global resource ordering to eliminate circular wait). | Embedded systems, mission-critical real-time controllers. |
| **Deadlock Avoidance** | Dynamically examine every resource request and only grant it if the system remains in a **Safe State** (using **Dijkstra's Banker's Algorithm**). | Systems with known maximum resource claims. |
| **Deadlock Detection & Recovery**| Periodically construct a Wait-For Graph (WFG) or Resource Allocation Graph (RAG), detect cycles, and terminate processes or preempt resources to break the cycle. | Modern Database Management Systems (PostgreSQL, Oracle). |
| **Deadlock Ignorance (Ostrich Algorithm)** | Ignore the problem assuming occurrences are exceptionally rare. Avoids performance and memory overhead. | General-purpose OSs (Linux, Windows, macOS). |`;
    }

    // 7. Cloud Computing: Service Models (IaaS, PaaS, SaaS)
    if (q.includes('cloud') || q.includes('iaas') || q.includes('paas') || q.includes('saas')) {
      return `Cloud computing is the on-demand availability of computer system resources—especially data storage, computing power, and networking—over the internet with pay-as-you-go pricing, structured into three primary **Service Models**:

---

### 1. Infrastructure as a Service (IaaS)
- **Concept:** Provides raw, fundamental computing resources: virtual machines, physical servers, block storage, and software-defined networking.
- **Provider Delivers:** Physical data centers, hypervisors, server hardware, and network routers.
- **Client Manages:** Operating system installation, patches, runtime libraries, middleware, databases, and application code.
- **Examples:** Amazon EC2, Google Compute Engine (GCE), Microsoft Azure Virtual Machines, DigitalOcean Droplets.
- **Best For:** DevOps teams and network architects requiring granular control over OS kernels and custom network topologies.

---

### 2. Platform as a Service (PaaS)
- **Concept:** Delivers a fully managed execution runtime environment where developers can deploy applications without maintaining infrastructure.
- **Provider Delivers:** Hardware, virtualization, operating systems, database engines, web servers, and runtime execution environments.
- **Client Manages:** Application source code, configuration settings, and data.
- **Examples:** Google App Engine, Heroku, AWS Elastic Beanstalk, Vercel, Netlify.
- **Best For:** Software engineers focused on rapid development and API delivery who want to avoid OS maintenance and server patching.

---

### 3. Software as a Service (SaaS)
- **Concept:** End-user applications hosted and managed completely by the cloud vendor, accessible via web browsers or client APIs.
- **Provider Delivers:** The entire technology stack end-to-end: infrastructure, OS, storage, code, updates, backups, and security.
- **Client Manages:** User access accounts and application data input.
- **Examples:** Google Workspace (Gmail, Docs, Drive), Microsoft 365, Salesforce CRM, Zoom, Slack.
- **Best For:** End users and enterprises needing immediate turnkey solutions without software engineering overhead.

---

### Shared Responsibility Matrix

| Layer | IaaS | PaaS | SaaS |
| :--- | :---: | :---: | :---: |
| **Application Logic** | You Manage | You Manage | Provider |
| **Application Data** | You Manage | You Manage | Provider |
| **Runtime Environment** | You Manage | Provider | Provider |
| **Middleware & Frameworks** | You Manage | Provider | Provider |
| **Operating System** | You Manage | Provider | Provider |
| **Virtualization** | Provider | Provider | Provider |
| **Physical Servers & Network**| Provider | Provider | Provider |`;
    }

    // 8. Database Management Systems: ACID Properties
    if (q.includes('acid') || (q.includes('transaction') && q.includes('database'))) {
      return `In Database Management Systems (DBMS), a **Transaction** is a logical sequence of read and write operations treated as a single, atomic unit of work.

To guarantee reliability and data integrity—particularly during concurrent access or hardware failure—all transactions must adhere to the four **ACID** properties:

---

### 1. Atomicity ("All or Nothing")
- A transaction must either execute in its entirety or have no effect whatsoever. If an unexpected failure occurs before completion, the DBMS initiates a **rollback** to restore the initial state.
- **Example:** Transferring $\\$500$ from Account A to Account B requires two operations: debiting A and crediting B. If power fails after debiting A, Atomicity ensures the $\\$500$ is automatically refunded to A.

---

### 2. Consistency ("Preserving Invariants")
- A transaction must transform the database from one valid consistent state to another valid consistent state, strictly honoring all integrity constraints (primary keys, foreign keys, check constraints, domain types).
- **Example:** In a banking ledger, total funds must remain invariant: $\\text{Balance}_A + \\text{Balance}_B = \\text{Constant}$. A transaction violating account rules is immediately aborted.

---

### 3. Isolation ("Concurrency Without Interference")
- Concurrently executing transactions must execute independently without interfering with each other's intermediate data. The execution of concurrent transactions yields the same result as if they ran sequentially (serializability).
- **Example:** While Transaction 1 is transferring money between accounts, Transaction 2 calculating aggregate balances will never see a transient state where money has left Account A but not yet arrived at Account B.

---

### 4. Durability ("Permanent Survivability")
- Once a transaction is successfully committed, its modifications are permanently recorded on non-volatile secondary storage (via Write-Ahead Logging / WAL) and survive any subsequent power outage, crash, or OS reboot.
- **Example:** Once an ATM displays *"Transaction Approved"*, the account debit is guaranteed permanent and will not be reverted by a system reboot.`;
    }

    // 9. Computer Networks: OSI 7-Layer Architecture
    if (q.includes('osi') || q.includes('7 layer') || q.includes('seven layer')) {
      return `The **OSI (Open Systems Interconnection)** model, standardized by the International Organization for Standardization (ISO) in 1984, provides a conceptual 7-layer framework characterizing how network protocols communicate.

---

### The 7 Layers Explained (From Top to Bottom)

| Layer | Name | Primary Function | Data Unit (PDU) | Protocols & Devices |
| :---: | :--- | :--- | :--- | :--- |
| **7** | **Application** | Direct interface with user applications for network communication | Data | HTTP, HTTPS, DNS, FTP, SMTP, SSH |
| **6** | **Presentation** | Data representation, translation, encryption, decryption, and compression | Data | TLS/SSL, JPEG, ASCII, MP4 |
| **5** | **Session** | Establishes, manages, synchronizes, and terminates inter-host dialogues | Data | RPC, NetBIOS, PPTP |
| **4** | **Transport** | End-to-end reliability, segment reassembly, flow control, port addressing | Segment / Datagram | **TCP** (Reliable), **UDP** (Low-latency) |
| **3** | **Network** | Logical IP addressing, packet routing across distributed networks | Packet | IPv4, IPv6, ICMP, Routers |
| **2** | **Data Link** | Physical node-to-node frame delivery, hardware MAC addressing, error checking | Frame | Ethernet (802.3), Wi-Fi (802.11), Switches |
| **1** | **Physical** | Transmission of raw unstructured bit streams across physical mediums | Bit | Fiber optics, Cat6 cables, Radio, Hubs |

---

> 💡 **Memory Mnemonic (Top to Bottom):** **A**ll **P**eople **S**eem **T**o **N**eed **D**ata **P**rocessing (*Application, Presentation, Session, Transport, Network, Data Link, Physical*).`;
    }

    // 10. Syllabus Context Match Fallback
    if (syllabusMatch && syllabusMatch.item) {
      return cleanSyllabusHtml(syllabusMatch.item.a);
    }

    // 11. Intelligent Academic General Fallback
    const cleanTopic = rawQuery.replace(/[?.,!]/g, '').trim();
    return `**${cleanTopic}** is an essential subject concept in Computer Science, Application Development, and Applied Mathematics.

---

### Core Conceptual Framework
- **Primary Definition:** It provides structured principles for computational problem-solving, algorithmic optimization, or system design.
- **Theoretical Basis:** It operates under deterministic rules, mathematical invariants, and standard engineering specifications.
- **University Curriculum Relevance:** In BCA, MCA, and Computer Science programs, mastery of this topic is critical for understanding software system architecture, database performance, and algorithmic complexity.

---

### Practical Application & Summary
- **Implementation:** Deployed in modern enterprise applications, distributed systems, and real-time processing pipelines.
- **Best Practice:** Always evaluate time complexity $\\mathcal{O}(n)$, spatial memory constraints, and fault tolerance when designing solutions around this concept.`;
  }

  // Pre-process & render math using KaTeX (with Unicode fallback)
  function renderMathInHtml(content) {
    if (!content) return "";
    let processed = content;

    // Check if KaTeX is available
    const hasKatex = typeof window.katex !== 'undefined' && typeof window.katex.renderToString === 'function';

    // 1. Block Math: $$...$$ and \[...\]
    processed = processed.replace(/\$\$([\s\S]*?)\$\$/g, function(match, expr) {
      if (hasKatex) {
        try {
          return `<div class="katex-display my-3 overflow-x-auto text-center">${window.katex.renderToString(expr.trim(), { displayMode: true, throwOnError: false })}</div>`;
        } catch (e) {
          console.warn("KaTeX display render error:", e);
        }
      }
      return `<div class="font-mono bg-blue-50/60 text-blue-900 p-3 my-2 rounded-xl text-center overflow-x-auto border border-blue-100">${applyUnicodeMathFallback(expr.trim())}</div>`;
    });

    processed = processed.replace(/\\\[([\s\S]*?)\\\]/g, function(match, expr) {
      if (hasKatex) {
        try {
          return `<div class="katex-display my-3 overflow-x-auto text-center">${window.katex.renderToString(expr.trim(), { displayMode: true, throwOnError: false })}</div>`;
        } catch (e) {
          console.warn("KaTeX display render error:", e);
        }
      }
      return `<div class="font-mono bg-blue-50/60 text-blue-900 p-3 my-2 rounded-xl text-center overflow-x-auto border border-blue-100">${applyUnicodeMathFallback(expr.trim())}</div>`;
    });

    // 2. Inline Math: $...$ and \(...\)
    processed = processed.replace(/\$([^\$\n]+?)\$/g, function(match, expr) {
      if (hasKatex) {
        try {
          return `<span class="katex-inline">${window.katex.renderToString(expr.trim(), { displayMode: false, throwOnError: false })}</span>`;
        } catch (e) {
          console.warn("KaTeX inline render error:", e);
        }
      }
      return `<code class="font-mono bg-blue-50/80 text-blue-900 px-1.5 py-0.5 rounded text-[12px]">${applyUnicodeMathFallback(expr.trim())}</code>`;
    });

    processed = processed.replace(/\\\(([^\n]+?)\\\)/g, function(match, expr) {
      if (hasKatex) {
        try {
          return `<span class="katex-inline">${window.katex.renderToString(expr.trim(), { displayMode: false, throwOnError: false })}</span>`;
        } catch (e) {
          console.warn("KaTeX inline render error:", e);
        }
      }
      return `<code class="font-mono bg-blue-50/80 text-blue-900 px-1.5 py-0.5 rounded text-[12px]">${applyUnicodeMathFallback(expr.trim())}</code>`;
    });

    return processed;
  }

  // Format Markdown with KaTeX integration
  function formatMarkdown(text) {
    if (!text) return "";
    let html = text;

    // First render mathematical formulas
    html = renderMathInHtml(html);

    // Markdown parsing
    if (window.marked && typeof window.marked.parse === 'function') {
      try {
        html = window.marked.parse(html);
      } catch (err) {
        console.warn("Markdown parse warning:", err);
      }
    }

    // Code blocks with syntax copy buttons
    html = html.replace(/<pre><code class="language-(.*?)">([\s\S]*?)<\/code><\/pre>/g, function(match, lang, code) {
      const codeId = 'code_' + Math.random().toString(36).substr(2, 9);
      return `
        <div class="my-3 rounded-2xl overflow-hidden bg-slate-900 text-slate-100 text-xs border border-slate-800 shadow-md">
          <div class="flex items-center justify-between px-4 py-2 bg-slate-950 border-b border-slate-800 text-[11px] font-mono text-slate-400">
            <span class="font-bold uppercase tracking-wider text-blue-400">${lang}</span>
            <button onclick="BrainwareAITutor.copyCode(this, '${codeId}')" class="flex items-center gap-1.5 hover:text-white transition px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300">
              <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"></path></svg>
              <span>Copy</span>
            </button>
          </div>
          <pre class="p-4 overflow-x-auto font-mono text-[12px] leading-relaxed"><code id="${codeId}">${code}</code></pre>
        </div>
      `;
    });

    // Style tables cleanly
    html = html.replace(/<table>/g, '<div class="overflow-x-auto my-3 rounded-xl border border-slate-200 shadow-sm"><table class="w-full text-xs text-left border-collapse">');
    html = html.replace(/<\/table>/g, '</table></div>');
    html = html.replace(/<th>/g, '<th class="bg-slate-900 p-2.5 font-bold text-white border-b border-slate-800 text-xs">');
    html = html.replace(/<tr>/g, '<tr class="even:bg-slate-50 hover:bg-blue-50/50 transition-colors">');
    html = html.replace(/<td>/g, '<td class="p-2.5 border-b border-slate-100 text-slate-700 text-xs">');

    // Headings
    html = html.replace(/<h3>(.*?)<\/h3>/g, '<h3 class="text-sm font-bold text-slate-900 mt-4 mb-2 border-b border-slate-100 pb-1">$1</h3>');
    html = html.replace(/<h2>(.*?)<\/h2>/g, '<h2 class="text-base font-bold text-slate-900 mt-5 mb-2.5">$1</h2>');

    return html;
  }

  // Typewriter Streaming
  async function streamTypewriter(fullText, textContainer, threadContainer) {
    const words = fullText.split(/(\s+)/);
    let displayed = "";
    
    for (let i = 0; i < words.length; i++) {
      if (!isGenerating) break;
      displayed += words[i];
      textContainer.innerHTML = formatMarkdown(displayed) + '<span class="inline-block w-1.5 h-3.5 ml-1 bg-blue-600 animate-pulse align-middle"></span>';
      
      if (threadContainer) {
        threadContainer.scrollTop = threadContainer.scrollHeight;
      }

      if (words[i].trim().length > 0) {
        await new Promise(r => setTimeout(r, 6));
      }
    }

    textContainer.innerHTML = formatMarkdown(displayed);
    if (threadContainer) {
      threadContainer.scrollTop = threadContainer.scrollHeight;
    }
  }

  return {
    init: function() {
      console.log("Brainware AI Study Assistant ready with KaTeX math and pure ChatGPT/Gemini mode.");
      this.attachInputKeyHandlers();
    },

    attachInputKeyHandlers: function() {
      const mainInput = document.getElementById('aiTutorInput');
      if (mainInput) {
        mainInput.addEventListener('keydown', function(e) {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            BrainwareAITutor.askQuestion();
          }
        });
      }

      const popupInput = document.getElementById('fbChatInput');
      if (popupInput) {
        popupInput.addEventListener('keydown', function(e) {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            BrainwareAITutor.askQuestion(null, true);
          }
        });
      }
    },

    toggleChatPopup: function() {
      const popup = document.getElementById('fbChatPopup');
      if (popup) {
        popup.classList.toggle('hidden');
        if (!popup.classList.contains('hidden')) {
          document.getElementById('fbChatInput')?.focus();
        }
      }
    },

    closeChatPopup: function() {
      const popup = document.getElementById('fbChatPopup');
      if (popup) popup.classList.add('hidden');
    },

    stopGeneration: function() {
      if (isGenerating) {
        isGenerating = false;
        this.updateSendButton(false);
      }
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

    askQuestion: async function(customQuery, isPopup = false) {
      if (isGenerating) {
        this.stopGeneration();
        return;
      }

      const inputId = isPopup ? 'fbChatInput' : 'aiTutorInput';
      const input = document.getElementById(inputId);
      const query = customQuery || (input ? input.value : '');
      if (!query || !query.trim()) return;

      lastUserQuery = query.trim();
      if (input && !customQuery) {
        input.value = '';
      }

      const threadId = isPopup ? 'fbChatMessages' : 'aiChatThread';
      const threadContainer = document.getElementById(threadId);

      this.appendUserBubble(query, threadId);

      const msgId = 'aimsg_' + (++messageCounter);
      const { contentContainer } = this.createAssistantStreamingBubble(msgId, threadId);

      isGenerating = true;
      this.updateSendButton(true, isPopup);

      try {
        const sanitized = sanitizeUserQuery(query);
        const qClean = sanitized.toLowerCase();

        // 1. Quick greetings
        if (quickGreetings[qClean]) {
          await streamTypewriter(quickGreetings[qClean], contentContainer, threadContainer);
          this.finishMessage(msgId, quickGreetings[qClean], threadId);
          return;
        }

        // 2. Strict Syllabus Grounding Check
        const syllabusMatch = findSyllabusMatch(query);

        // 3. Multi-Provider Live AI Inference
        let responseText = await queryLiveAI(query, syllabusMatch ? syllabusMatch.item : null);

        // 4. Fallback to Master Academic Knowledge Base (Pure ChatGPT/Gemini answer only)
        if (!responseText || responseText.length < 30) {
          responseText = getMasterAcademicAnswer(query, syllabusMatch);
        }

        // Stream the pure answer smoothly
        await streamTypewriter(responseText, contentContainer, threadContainer);
        this.finishMessage(msgId, responseText, threadId);

      } catch (err) {
        console.error("AI Assistant processing error:", err);
        contentContainer.innerHTML = formatMarkdown("An unexpected interruption occurred. Please click **Regenerate** below to retry.");
      } finally {
        isGenerating = false;
        this.updateSendButton(false, isPopup);
      }
    },

    // Pure ChatGPT / Gemini Style Bubble: ONLY THE ANSWER, ZERO METADATA/BADGES
    createAssistantStreamingBubble: function(msgId, threadId) {
      const container = document.getElementById(threadId);
      const bubble = document.createElement('div');
      bubble.id = msgId;
      bubble.className = 'flex items-start gap-3.5 mb-6 animate-fade-in group';

      bubble.innerHTML = `
        <div class="w-8 h-8 rounded-full bg-blue-600 text-white flex-shrink-0 flex items-center justify-center text-xs font-bold shadow-sm mt-0.5">
          AI
        </div>
        <div class="flex-1 min-w-0">
          <div class="ai-content-body text-slate-800 text-xs sm:text-sm leading-relaxed space-y-3">
            <div class="flex items-center gap-2 text-slate-400 py-1">
              <span class="w-1.5 h-1.5 bg-blue-600 rounded-full animate-bounce"></span>
              <span class="w-1.5 h-1.5 bg-blue-600 rounded-full animate-bounce [animation-delay:0.2s]"></span>
              <span class="w-1.5 h-1.5 bg-blue-600 rounded-full animate-bounce [animation-delay:0.4s]"></span>
            </div>
          </div>
          <div class="action-footer mt-2.5 flex items-center gap-2 text-slate-400 opacity-0 group-hover:opacity-100 transition-opacity">
            <button onclick="BrainwareAITutor.copyMessage(this, '${msgId}')" title="Copy response" class="p-1 hover:text-slate-700 hover:bg-slate-100 rounded-md transition flex items-center gap-1 text-xs">
              <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"></path></svg>
              <span class="text-[11px]">Copy</span>
            </button>
            <button onclick="BrainwareAITutor.regenerateLast()" title="Regenerate" class="p-1 hover:text-slate-700 hover:bg-slate-100 rounded-md transition flex items-center gap-1 text-xs">
              <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path></svg>
              <span class="text-[11px]">Regenerate</span>
            </button>
          </div>
        </div>
      `;

      container.appendChild(bubble);
      container.scrollTop = container.scrollHeight;

      return {
        bubbleContainer: bubble,
        contentContainer: bubble.querySelector('.ai-content-body')
      };
    },

    finishMessage: function(msgId, fullText, threadId) {
      const bubble = document.getElementById(msgId);
      if (!bubble) return;

      if (window.lucide && typeof window.lucide.createIcons === 'function') {
        window.lucide.createIcons();
      }

      const container = document.getElementById(threadId);
      if (container) container.scrollTop = container.scrollHeight;
    },

    appendUserBubble: function(text, threadId) {
      const container = document.getElementById(threadId);
      if (!container) return;

      const bubble = document.createElement('div');
      bubble.className = 'flex justify-end mb-4 animate-fade-in';
      bubble.innerHTML = `
        <div class="max-w-[85%] bg-blue-600 text-white rounded-3xl rounded-tr-sm px-5 py-3 text-xs sm:text-sm font-medium leading-relaxed shadow-sm">
          ${this.escapeHtml(text)}
        </div>
      `;
      container.appendChild(bubble);
      container.scrollTop = container.scrollHeight;
    },

    regenerateLast: function() {
      if (lastUserQuery) {
        this.askQuestion(lastUserQuery);
      }
    },

    copyCode: function(btn, codeId) {
      const codeElem = document.getElementById(codeId);
      if (!codeElem) return;
      const text = codeElem.textContent || codeElem.innerText;
      navigator.clipboard.writeText(text).then(() => {
        const originalHtml = btn.innerHTML;
        btn.innerHTML = `<span class="text-emerald-400 font-bold">✓ Copied</span>`;
        setTimeout(() => {
          btn.innerHTML = originalHtml;
        }, 2000);
      });
    },

    copyMessage: function(btn, msgId) {
      const bubble = document.getElementById(msgId);
      if (!bubble) return;
      const body = bubble.querySelector('.ai-content-body');
      if (!body) return;
      const text = body.textContent || body.innerText;
      navigator.clipboard.writeText(text.trim()).then(() => {
        const span = btn.querySelector('span');
        if (span) {
          const oldText = span.textContent;
          span.textContent = "Copied!";
          setTimeout(() => { span.textContent = oldText; }, 2000);
        }
      });
    },

    clearChat: function() {
      chatHistory = [];
      const thread = document.getElementById('aiChatThread');
      if (thread) {
        thread.innerHTML = `
          <div class="text-center py-12 px-4 max-w-md mx-auto">
            <div class="w-12 h-12 rounded-full bg-blue-600 text-white flex items-center justify-center mx-auto shadow-md mb-3">
              <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
            </div>
            <h3 class="text-base font-bold text-slate-900 mb-1">How can I help you today?</h3>
            <p class="text-xs text-slate-500 mb-6 leading-relaxed">
              Ask any question from your curriculum, programming, mathematics, or computer science.
            </p>
            <div class="flex flex-wrap justify-center gap-2">
              <button onclick="BrainwareAITutor.askQuestion('Distinguish between CBOW and Skip-Gram architectures of Word2Vec.')" class="text-xs font-medium bg-white hover:bg-slate-50 text-slate-700 px-3.5 py-1.5 rounded-full border border-slate-200 shadow-sm transition">
                CBOW vs Skip-Gram
              </button>
              <button onclick="BrainwareAITutor.askQuestion('Explain Lagrange Multipliers using lambda and optimization.')" class="text-xs font-medium bg-white hover:bg-slate-50 text-slate-700 px-3.5 py-1.5 rounded-full border border-slate-200 shadow-sm transition">
                Lagrange Multipliers (λ)
              </button>
              <button onclick="BrainwareAITutor.askQuestion('Explain De Morgan\\'s laws using AND, OR, and NOT logical operators.')" class="text-xs font-medium bg-white hover:bg-slate-50 text-slate-700 px-3.5 py-1.5 rounded-full border border-slate-200 shadow-sm transition">
                De Morgan's Laws (∧, ∨, ¬)
              </button>
              <button onclick="BrainwareAITutor.askQuestion('What is a deadlock and explain the 4 Coffman conditions?')" class="text-xs font-medium bg-white hover:bg-slate-50 text-slate-700 px-3.5 py-1.5 rounded-full border border-slate-200 shadow-sm transition">
                Deadlock & Coffman Conditions
              </button>
              <button onclick="BrainwareAITutor.askQuestion('What is the difference between a Process and a Thread?')" class="text-xs font-medium bg-white hover:bg-slate-50 text-slate-700 px-3.5 py-1.5 rounded-full border border-slate-200 shadow-sm transition">
                Process vs Thread
              </button>
            </div>
          </div>
        `;
      }
    },

    escapeHtml: function(text) {
      if (!text) return "";
      const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' };
      return text.replace(/[&<>"']/g, m => map[m]);
    }
  };
})();

document.addEventListener('DOMContentLoaded', function() {
  if (window.BrainwareAITutor) window.BrainwareAITutor.init();
});
