"""
Brainware Academic AI - Enterprise Universal Educational Intelligence Model Engine
File: academic_model.py
Version: 6.0.0 - Full Multi-Provider LLM Intelligence (Gemini & OpenAI) with Dynamic Vector RAG & Web Search

Architecture:
1. EXAM_SYSTEM_PROMPT: Universal Educational Master Instruction
   - Covers all academic disciplines worldwide (Computer Science, Math, Science, Medicine, Business, Humanities, Law, Languages, Research).
   - Understands education systems globally (India CBSE/ICSE/UGC, UK GCSE/A-level, IB, Cambridge, US AP/Undergrad, European frameworks).
   - Adapts to every academic level (Primary to Doctoral/Professional).
   - Handles every question format (Definitions, Derivations, Code, Comparison Tables, MCQs, Essays).
   - Respects user-specified marks (2M, 5M, 10M, 15M) and word counts.
   - Context-aware technical term disambiguation (Java, DS, C++, AI, DBMS, ML).
   - Native multilingual answering (Bengali, Hindi, English, Spanish, French, etc.).
2. GeminiNeuralGateway: Official Google Gemini Foundation Model gateway (gemini-1.5-flash, gemini-2.0-flash-exp, gemini-1.5-pro).
3. OpenAINeuralGateway: Official OpenAI Python SDK client (gpt-4o-mini, gpt-4o, gpt-3.5-turbo) with custom OPENAI_BASE_URL support (Groq, OpenRouter, Ollama, vLLM).
4. WebSearchEngine: Real-time web retrieval for current technological and academic developments.
5. VectorRAGModel: NumPy & SciPy-accelerated vector space grounding over 1,254 university syllabus questions & uploaded documents.
6. DynamicAcademicAIModel: Master engine implementing OpenAI standard (/v1/chat/completions) & Web Portal (/api/chat).
"""

import os
import sys
import re
import json
import time
import math
import urllib.parse
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import scipy.spatial.distance

# Import official OpenAI SDK
try:
    import openai
    from openai import OpenAI
    HAS_OPENAI_SDK = True
except ImportError:
    HAS_OPENAI_SDK = False

# HTTP client library
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    import urllib.request
    import urllib.error
    HAS_REQUESTS = False


# =============================================================================
# 1. UNIVERSAL EDUCATIONAL SYSTEM INSTRUCTION (Worldwide General-Purpose Intelligence)
# =============================================================================
EXAM_SYSTEM_PROMPT = """You are Brainware Academic AI, an expert worldwide educational intelligence, senior university professor, and universal academic examiner.

You are comparable in knowledge, reasoning ability, and pedagogy to the world's most capable general-purpose AI models (Google Gemini and ChatGPT).

Your core mandate is to understand and accurately answer educational questions across virtually every academic and professional discipline, from any country, curriculum, educational board, university, and academic level worldwide.

---

### 1. UNIVERSAL DISCIPLINARY SCOPE
You possess mastery across all academic and professional fields, including but not limited to:
- **Computer Science & IT:** Programming (Java, Python, C, C++, JavaScript, PHP, Rust, Go), Web & Full-Stack (HTML, CSS, React, Node.js, Express, MongoDB, MERN), Data Structures & Algorithms, Operating Systems, DBMS & SQL/NoSQL, Computer Networks, Artificial Intelligence & Machine Learning, Deep Learning, Cybersecurity, Cloud Computing, Distributed Systems, Software Engineering, Computer Architecture.
- **Mathematics & Statistics:** Arithmetic, Algebra, Calculus, Geometry, Trigonometry, Linear Algebra, Probability, Statistics, Discrete Mathematics, Numerical Methods, Optimization.
- **Science & Engineering:** Classical & Modern Physics, Inorganic/Organic/Physical Chemistry, Cell Biology, Molecular Genetics, Biotechnology, Environmental Science, Electrical, Mechanical, Civil Engineering, Robotics.
- **Medical & Health Sciences:** Human Anatomy, Physiology, Biochemistry, Pharmacology, Pathology, Microbiology, Public Health, Nursing, Medical Terminology (educational explanations only).
- **Business, Management & Economics:** Financial Accounting, Corporate Finance, Micro/Macroeconomics, Marketing, Management, Organizational Behavior, Business Analytics, Banking.
- **Humanities & Social Sciences:** World History, Regional History, Geography, Political Science, Sociology, Psychology, Philosophy, Anthropology, International Relations.
- **Law, Literature & Languages:** Constitutional Law, Jurisprudence, English & World Literature, Linguistics, Research Methodology, Multilingual Studies.
- **General Knowledge & Other Subjects:** Open education across all recognized disciplines worldwide.

---

### 2. GLOBAL CURRICULUM & LEVEL ADAPTABILITY
- **Curricula Worldwide:** Adapt seamlessly to the curriculum specified by the user—whether Indian (CBSE, ICSE, ISC, State Boards, UGC, AICTE, NEP 2020 OBE), British (GCSE, IGCSE, A-levels), International (Cambridge, International Baccalaureate - IB), American (K-12, AP, College/Undergraduate, Graduate), Canadian, Australian, European, or others. If unspecified, answer using globally accepted scientific and academic principles.
- **Academic Levels:** Intelligently adjust terminology, explanation depth, and examples to match the audience:
  - *Primary & Middle School:* Intuitive analogies, clear everyday language, foundational principles.
  - *Secondary & Higher Secondary:* Structured concepts, formal definitions, standard equations, worked examples.
  - *Undergraduate & Diploma:* In-depth theoretical rigor, architectural schematics, Big-O complexity, mathematical proofs, real-world case studies.
  - *Postgraduate, Doctoral & Professional:* Advanced literature, edge cases, underlying mechanics, mathematical formalisms, trade-off evaluations.

---

### 3. QUESTION FORMAT & STRUCTURE REQUIREMENTS
Produce the structure best suited to the question's specific intent:
1. **Definitions & Concepts ("What is / Define"):**
   - Open with an unambiguous formal definition in bold, stating key principles and historical/theoretical origins.
2. **Comparative Inquiries ("Difference between X and Y" / "X vs Y"):**
   - ALWAYS include a comprehensive Markdown table comparing parameters (Parameter / Criteria, Concept A, Concept B, Examination Notes).
3. **Mathematical & Physics Problems ("Calculate / Solve / Derive"):**
   - Use LaTeX ($...$ inline, $$...$$ display).
   - Format: Given Parameters $\\to$ Target Variable $\\to$ Governing Formula $\\to$ Step-by-Step Derivation $\\to$ Final Answer with SI Units $\\to$ Verification Check.
4. **Programming & Algorithmic Problems ("Write a program / Code / Algorithm"):**
   - Provide the algorithmic strategy, clean source code with syntax highlighting and comments, line-by-line explanation, sample input/output, and Big-O Time & Space Complexity analysis.
5. **Architectural & Flow Inquiries:**
   - Use clean ASCII / Unicode diagrams to illustrate memory models, network handshakes, pipelines, biological cycles, or system architectures.
6. **Multiple-Choice Questions (MCQs):**
   - Clearly state the correct option, provide an in-depth justification, and explain why each distractor is incorrect.

---

### 4. ACADEMIC CUSTOMIZATION & MARKS RESPECT
Strictly obey explicit user constraints:
- **Marks Guidelines:**
  - *2 Marks:* Concise formal definition and core takeaway (~50-80 words).
  - *5 Marks:* Definition, key mechanics, 3-4 bullet points, and practical example (~150-250 words).
  - *10 Marks:* Thorough explanation, structured headings, comparison table or diagram, code/derivation, and examination summary (~400-600 words).
  - *15 Marks:* Comprehensive university thesis answer: introduction, theoretical foundations, architectural diagrams, deep mathematical/algorithmic analysis, real-world applications, limitations, and conclusion (~800+ words).
- **Word Limits:** Adhere strictly to any requested word count (e.g., "explain in 70 words").

---

### 5. CONTEXTUAL DISAMBIGUATION & CONVERSATION MEMORY
- **Homonym Disambiguation:** Distinguish terms by domain context (e.g., *Java* = programming language vs Indonesian island; *DS* = Data Structures vs Dental Surgery; *C++* = programming language; *ML* = Machine Learning vs Milliliters).
- **Multi-Turn Memory:** Connect seamlessly with previous conversation turns. If the user asks *"What are its advantages?"* after asking about *Java*, immediately recognize that *"it"* refers to Java.

---

### 6. MULTILINGUAL INTELLIGENCE
- Answer in the exact language requested by the student (e.g., Bengali, Hindi, English, Spanish, French, German).
- When answering in Bengali or Hindi, provide pure, natural, grammatically correct prose while preserving standard technical terms in English or clear transliteration.

---

### 7. FACTUAL ACCURACY & EPISTEMIC HONESTY
- Never hallucinate formulas, citations, or facts.
- For numerical questions, double-check all arithmetic.
- If information is genuinely unknown, ambiguous, or unverifiable, clearly state the uncertainty instead of fabricating an answer."""


# =============================================================================
# 2. NUMPY & SCIPY VECTOR RAG (Retrieval-Augmented Generation) MODEL
# =============================================================================
class VectorRAGModel:
    """NumPy and SciPy vector space retrieval model for dynamic syllabus grounding."""

    def __init__(self, corpus_path: Optional[str] = None):
        self.documents: List[Dict[str, Any]] = []
        self.vocabulary: Dict[str, int] = {}
        self.idf: np.ndarray = np.array([])
        self.matrix: Optional[np.ndarray] = None
        if corpus_path and os.path.exists(corpus_path):
            self.build_index(corpus_path)

    def build_index(self, corpus_path: str):
        try:
            with open(corpus_path, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)

            doc_tokens = []
            doc_freq: Dict[str, int] = {}
            for doc in self.documents[:3000]:
                text = f"{doc.get('q', '')} {doc.get('a', '') or doc.get('ans', '')}"
                tokens = re.findall(r'[a-zA-Z0-9]{3,}', text.lower())
                doc_tokens.append(tokens)
                unique_tokens = set(tokens)
                for t in unique_tokens:
                    doc_freq[t] = doc_freq.get(t, 0) + 1

            sorted_vocab = sorted(doc_freq.items(), key=lambda x: x[1], reverse=True)[:5000]
            self.vocabulary = {term: idx for idx, (term, _) in enumerate(sorted_vocab)}
            vocab_size = len(self.vocabulary)
            num_docs = len(doc_tokens)

            if vocab_size == 0 or num_docs == 0:
                return

            self.idf = np.zeros(vocab_size, dtype=np.float32)
            for term, idx in self.vocabulary.items():
                df = doc_freq.get(term, 0)
                self.idf[idx] = math.log((num_docs + 1) / (df + 1)) + 1.0

            self.matrix = np.zeros((num_docs, vocab_size), dtype=np.float32)
            for i, tokens in enumerate(doc_tokens):
                for t in tokens:
                    if t in self.vocabulary:
                        self.matrix[i, self.vocabulary[t]] += 1.0
                self.matrix[i] *= self.idf

            norms = np.linalg.norm(self.matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.matrix = self.matrix / norms

        except Exception as e:
            print(f"Notice: Vector RAG initialization: {e}", file=sys.stderr)

    def retrieve(self, query: str, top_k: int = 2, min_similarity: float = 0.50) -> List[Dict[str, Any]]:
        """Dynamically retrieves top-K relevant knowledge vectors using cosine similarity."""
        if self.matrix is None or len(self.vocabulary) == 0:
            return []

        q_tokens = re.findall(r'[a-zA-Z0-9]{3,}', query.lower())
        if not q_tokens:
            return []

        q_vec = np.zeros(len(self.vocabulary), dtype=np.float32)
        for t in q_tokens:
            if t in self.vocabulary:
                q_vec[self.vocabulary[t]] += 1.0
        q_vec *= self.idf

        norm = np.linalg.norm(q_vec)
        if norm == 0:
            return []
        q_vec /= norm

        similarities = np.dot(self.matrix, q_vec)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= min_similarity:
                doc = dict(self.documents[idx])
                doc['similarity'] = round(score, 4)
                results.append(doc)

        return results


# =============================================================================
# 3. REAL-TIME WEB SEARCH & SCHOLARLY RETRIEVAL ENGINE
# =============================================================================
class WebSearchEngine:
    """
    Real-time web retrieval engine for recent technical releases, current events,
    and updated academic documentation.
    """

    RECENT_TRIGGERS = [
        "latest", "recent", "newest", "current", "2024", "2025", "2026",
        "release date", "update", "current law", "news", "today", "who is the current"
    ]

    @classmethod
    def should_search(cls, query: str) -> bool:
        q_lower = query.lower()
        return any(trig in q_lower for trig in cls.RECENT_TRIGGERS)

    @classmethod
    def search(cls, query: str, max_results: int = 3) -> Optional[str]:
        """Performs live web retrieval via DuckDuckGo API or Wikipedia REST."""
        clean_q = re.sub(r'^(?:what is|who is|explain|tell me about)\s+', '', query, flags=re.I).strip('?.! ')
        if not clean_q:
            return None

        snippets = []

        # 1. Try DuckDuckGo Instant Answer API
        try:
            ddg_url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(clean_q)}&format=json&no_html=1&skip_disambig=1"
            headers = {"User-Agent": "BrainwareAcademicAI/6.0 (Educational Search)"}
            if HAS_REQUESTS:
                resp = requests.get(ddg_url, headers=headers, timeout=4)
                if resp.status_code == 200:
                    data = resp.json()
                    abstract = data.get("AbstractText") or data.get("Abstract")
                    if abstract:
                        snippets.append(f"Source: {data.get('Heading', 'Overview')} - {abstract}")
                    for topic in data.get("RelatedTopics", [])[:2]:
                        if isinstance(topic, dict) and topic.get("Text"):
                            snippets.append(f"Related: {topic.get('Text')}")
        except Exception:
            pass

        # 2. Try Wikipedia Summary API for authoritative encyclopedic grounding
        if not snippets:
            try:
                wiki_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(clean_q)}"
                headers = {"User-Agent": "BrainwareAcademicAI/6.0 (Educational Research)"}
                if HAS_REQUESTS:
                    w_resp = requests.get(wiki_url, headers=headers, timeout=4)
                    if w_resp.status_code == 200:
                        w_data = w_resp.json()
                        extract = w_data.get("extract")
                        if extract and len(extract) > 40:
                            snippets.append(f"Encyclopedic Reference ({w_data.get('title')}): {extract}")
            except Exception:
                pass

        if snippets:
            return "\n\n".join(snippets[:max_results])
        return None


# =============================================================================
# 4. OFFICIAL GOOGLE GEMINI NEURAL GATEWAY
# =============================================================================
class GeminiNeuralGateway:
    """
    Direct official integration with Google Gemini foundation models:
    - gemini-1.5-flash (primary, high-throughput)
    - gemini-2.0-flash-exp (next-generation multimodal intelligence)
    - gemini-1.5-pro (deep academic reasoning)
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = (api_key or os.environ.get("GEMINI_API_KEY", "")).strip()
        self.primary_model = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
        self.fallback_models = ["gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash"]

    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate(
        self,
        contents: List[Dict[str, Any]],
        system_prompt: str,
        temperature: float = 0.2
    ) -> Optional[Dict[str, Any]]:
        if not self.is_available():
            return None

        models_to_try = [self.primary_model] + [m for m in self.fallback_models if m != self.primary_model]
        last_error = None

        for model in models_to_try:
            for attempt in range(1, 3):
                try:
                    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={urllib.parse.quote(self.api_key)}"
                    payload = {
                        "contents": contents,
                        "systemInstruction": {"parts": [{"text": system_prompt}]},
                        "generationConfig": {
                            "temperature": temperature,
                            "topP": 0.95,
                            "maxOutputTokens": 4096
                        }
                    }

                    if HAS_REQUESTS:
                        resp = requests.post(
                            endpoint,
                            headers={"Content-Type": "application/json"},
                            json=payload,
                            timeout=32
                        )
                        status = resp.status_code
                        data = resp.json() if resp.content else {}
                    else:
                        req_data = json.dumps(payload).encode('utf-8')
                        req = urllib.request.Request(
                            endpoint,
                            data=req_data,
                            headers={"Content-Type": "application/json"},
                            method="POST"
                        )
                        with urllib.request.urlopen(req, timeout=32) as r:
                            status = r.status
                            data = json.loads(r.read().decode('utf-8'))

                    if status == 200:
                        candidates = data.get("candidates", [])
                        if candidates:
                            c = candidates[0]
                            parts = c.get("content", {}).get("parts", [])
                            if parts and "text" in parts[0] and parts[0]["text"].strip():
                                text = parts[0]["text"].strip()
                                return {
                                    "text": text,
                                    "model": model,
                                    "source": "google-gemini"
                                }

                    # Rate limited or server busy
                    if status in (429, 503) and attempt < 2:
                        time.sleep(1.5 * attempt)
                        continue

                    # If model not found or forbidden, break immediately to try fallback model
                    if status in (404, 400):
                        break

                except Exception as e:
                    last_error = e
                    if attempt < 2:
                        time.sleep(1.0)
                        continue
                    break

        if last_error:
            print(f"Gemini Gateway notice: {last_error}", file=sys.stderr)
        return None


# =============================================================================
# 5. OFFICIAL OPENAI NEURAL GATEWAY (With Custom Base URL Support)
# =============================================================================
class OpenAINeuralGateway:
    """
    Invokes OpenAI models (gpt-4o, gpt-4o-mini, gpt-3.5-turbo) or any OpenAI-compatible
    endpoint (Groq, OpenRouter, Mistral, Ollama, vLLM) via official OpenAI Python SDK.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini", base_url: Optional[str] = None):
        self.api_key = (api_key or os.environ.get("OPENAI_API_KEY", "")).strip()
        self.model = os.environ.get("OPENAI_MODEL", model)
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "")).strip() or None
        self.client: Optional[OpenAI] = None

        if self.api_key and HAS_OPENAI_SDK:
            try:
                if self.base_url:
                    self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
                else:
                    self.client = OpenAI(api_key=self.api_key)
            except Exception as e:
                print(f"Notice: OpenAI SDK init: {e}", file=sys.stderr)

    def is_available(self) -> bool:
        return bool(self.api_key and self.client is not None)

    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.2) -> Optional[Dict[str, Any]]:
        if not self.is_available():
            return None

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=4096
            )
            content = response.choices[0].message.content
            if content and content.strip():
                return {
                    "text": content.strip(),
                    "model": self.model,
                    "source": "openai-neural"
                }
        except Exception as e:
            print(f"OpenAI Generation Error: {e}", file=sys.stderr)
        return None


# =============================================================================
# 6. UNIVERSAL DYNAMIC ACADEMIC AI MODEL (MASTER SYSTEM ENGINE)
# =============================================================================
class DynamicAcademicAIModel:
    """
    Master Universal Academic AI Model Engine running entirely in Python.
    Orchestrates Google Gemini, OpenAI, Live Web Search, Vector RAG syllabus grounding,
    and multi-format document analysis.
    """

    def __init__(self, base_dir: Optional[str] = None, api_key: Optional[str] = None):
        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))

        # Initialize Neural Gateways
        self.openai_gateway = OpenAINeuralGateway(api_key=os.environ.get("OPENAI_API_KEY"))
        self.gemini_gateway = GeminiNeuralGateway(api_key=api_key or os.environ.get("GEMINI_API_KEY"))

        # Initialize NumPy Vector RAG model
        qa_path = os.path.join(self.base_dir, 'portal_qa_index.json')
        self.rag_model = VectorRAGModel(qa_path)

    def reload_keys(self):
        """Dynamically reloads keys from environment variables."""
        self.openai_gateway = OpenAINeuralGateway(api_key=os.environ.get("OPENAI_API_KEY"))
        self.gemini_gateway = GeminiNeuralGateway(api_key=os.environ.get("GEMINI_API_KEY"))

    def generate_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.2,
        file_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        OpenAI-compatible chat completion method.
        Matches exact schema: POST /v1/chat/completions
        """
        user_query = ""
        for m in reversed(messages):
            if m.get('role') == 'user':
                user_query = m.get('content', '')
                break

        # 1. Live Web Search (if query involves recent developments or current updates)
        web_context = ""
        if WebSearchEngine.should_search(user_query):
            web_results = WebSearchEngine.search(user_query)
            if web_results:
                web_context = f"[REAL-TIME WEB RETRIEVAL CONTEXT]:\n{web_results}"

        # 2. NumPy Vector RAG Grounding (Optional university syllabus reference)
        rag_results = self.rag_model.retrieve(user_query, top_k=2, min_similarity=0.62)
        grounding_context = ""
        if rag_results:
            grounding_chunks = []
            for r in rag_results:
                clean_ans = re.sub(r'<[^>]+>', '', r.get('a') or r.get('ans') or '').strip()
                grounding_chunks.append(f"University Reference Question: {r.get('q')}\nReference Solution:\n{clean_ans}")
            grounding_context = "[OPTIONAL UNIVERSITY QUESTION BANK REFERENCE]:\n" + "\n\n---\n\n".join(grounding_chunks) + "\n*(Use this reference if directly relevant to the question, but answer using your universal general academic knowledge.)*"

        # 3. Assemble Augmented Prompt with System Instruction
        augmented_messages = [{"role": "system", "content": EXAM_SYSTEM_PROMPT}]
        if web_context:
            augmented_messages.append({"role": "system", "content": web_context})
        if grounding_context:
            augmented_messages.append({"role": "system", "content": grounding_context})

        for m in messages:
            if m.get('role') != 'system':
                augmented_messages.append(m)

        # 4. Primary Model Arbitration: OpenAI -> Gemini
        result = None
        used_model = "brainware-academic-ai-v6"

        # Try OpenAI Gateway
        if self.openai_gateway.is_available():
            res = self.openai_gateway.generate(augmented_messages, temperature=temperature)
            if res and res.get("text"):
                result = res["text"]
                used_model = res["model"]

        # Try Google Gemini Gateway
        if not result and self.gemini_gateway.is_available():
            gemini_contents = []
            for m in augmented_messages:
                if m.get("role") == "system":
                    continue
                role = "model" if m.get("role") in ("model", "assistant") else "user"
                parts = [{"text": m.get("content", "")}]

                # Include image data if attached
                if file_info and file_info.get("base64") and file_info["base64"].startswith("data:image/"):
                    try:
                        mime_type = file_info.get("mimeType", "image/jpeg")
                        b64_data = file_info["base64"].split(",")[1]
                        parts.append({"inlineData": {"mimeType": mime_type, "data": b64_data}})
                    except Exception:
                        pass

                gemini_contents.append({"role": role, "parts": parts})

            res = self.gemini_gateway.generate(gemini_contents, system_prompt=EXAM_SYSTEM_PROMPT, temperature=temperature)
            if res and res.get("text"):
                result = res["text"]
                used_model = res["model"]

        # 5. If no API key configured, provide structured instructional guide
        if not result:
            result = self._get_unconfigured_api_response(user_query, rag_results)
            used_model = "brainware-academic-ai-setup-guide"

        prompt_words = len(user_query.split())
        completion_words = len(result.split())

        return {
            "id": f"chatcmpl-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": used_model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": prompt_words * 2,
                "completion_tokens": completion_words * 2,
                "total_tokens": (prompt_words + completion_words) * 2
            }
        }

    def generate_exam_response(
        self,
        query: str,
        history: List[Dict[str, Any]] = None,
        file_info: Optional[Dict[str, Any]] = None,
        client_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Primary handler for /api/chat. Adapts web portal payload to the Universal AI Model.
        """
        history = history or []
        messages = []

        for item in history[-14:]:
            role = "user" if item.get("role") == "user" else "assistant"
            text = item.get("text") or item.get("content") or ""
            if text:
                messages.append({"role": role, "content": text})

        full_query = query
        if file_info and file_info.get("text"):
            full_query = f"[ATTACHED DOCUMENT: {file_info.get('name', 'document')} ({file_info.get('type', 'DOCUMENT')})]\n{file_info.get('text', '')[:12000]}\n\n[STUDENT INQUIRY]:\n{query}"

        messages.append({"role": "user", "content": full_query})

        # Temporary key override if provided
        if client_key and not self.gemini_gateway.is_available() and not self.openai_gateway.is_available():
            if client_key.startswith("sk-"):
                temp_openai = OpenAINeuralGateway(api_key=client_key)
                if temp_openai.is_available():
                    res = temp_openai.generate([{"role": "system", "content": EXAM_SYSTEM_PROMPT}] + messages)
                    if res and res.get("text"):
                        return {"success": True, "text": res["text"], "model": res["model"], "source": "openai-neural"}
            else:
                temp_gemini = GeminiNeuralGateway(api_key=client_key)
                if temp_gemini.is_available():
                    gemini_contents = []
                    for m in messages:
                        role = "model" if m.get("role") == "assistant" else "user"
                        gemini_contents.append({"role": role, "parts": [{"text": m.get("content", "")}]})
                    res = temp_gemini.generate(gemini_contents, system_prompt=EXAM_SYSTEM_PROMPT)
                    if res and res.get("text"):
                        return {"success": True, "text": res["text"], "model": res["model"], "source": "google-gemini"}

        completion = self.generate_chat_completion(messages, file_info=file_info)
        text = completion["choices"][0]["message"]["content"]
        model = completion["model"]

        return {
            "success": True,
            "text": text,
            "model": model,
            "source": "backend-llm" if ("gemini" in model or "gpt" in model) else "system-guide",
            "usage": completion.get("usage")
        }

    def _get_unconfigured_api_response(self, query: str, rag_results: List[Dict[str, Any]]) -> str:
        """Clear guidance when server-side LLM key is not yet set."""
        rag_note = ""
        if rag_results and rag_results[0].get('similarity', 0) >= 0.75:
            doc = rag_results[0]
            clean_ans = re.sub(r'<[^>]+>', '', doc.get('a') or doc.get('ans') or '').strip()
            rag_note = f"""

---

### [University Syllabus Reference Match Found]
*From Brainware University Examination Bank (Similarity: {doc.get('similarity')}):*

**Question:** {doc.get('q')}

{clean_ans}"""

        return f"""### Brainware Academic AI — Universal Model Setup

To activate live, general-purpose LLM reasoning across all academic subjects worldwide (like ChatGPT and Google Gemini), add your official API key to the server-side `.env` file:

---

### How to Configure in 60 Seconds:
1. Open the `.env` file in your project directory:
   ```env
   # Option A: Google Gemini (Free tier available)
   GEMINI_API_KEY=your_gemini_api_key_here
   GEMINI_MODEL=gemini-1.5-flash

   # Option B: OpenAI (or compatible providers like Groq, OpenRouter)
   OPENAI_API_KEY=your_openai_api_key_here
   OPENAI_MODEL=gpt-4o-mini
   ```
2. Free Google Gemini API keys can be obtained directly from Google AI Studio: [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey).
3. Restart the server (`python server.py`). The universal AI model will immediately begin answering any question from any academic subject worldwide.{rag_note}"""


# Backwards compatibility alias
AcademicAIModel = DynamicAcademicAIModel
