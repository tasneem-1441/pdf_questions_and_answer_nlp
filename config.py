"""
Configuration settings for the PDF Question Answering RAG system.
Includes chunking parameters, embedding and generator model configurations,
FAISS retrieval settings, and prompt templates.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Suppress Hugging Face Windows symlink warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "cache"

# Ensure cache directory exists
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Load environment variables from .env file
load_dotenv(dotenv_path=BASE_DIR / ".env")

# Generator Backend Configuration
# Options: "gemini" (Google Gemini API) or "local" (Hugging Face)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip() or None
GEMINI_MODEL_NAME = "gemini-2.5-flash"

# Local models (defaulting to lightweight, high-speed 0.5B instruction model for CPU)
LOCAL_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
LOCAL_MODEL_OPTIONS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "google/flan-t5-base"
]
FALLBACK_MODEL_NAME = "google/flan-t5-base"
GENERATOR_MODEL_NAME = LOCAL_MODEL_NAME

# Backend auto-selection: default to "gemini" if API key exists, otherwise "local"
GENERATOR_BACKEND = "gemini" if GEMINI_API_KEY else "local"

# Document Chunking Settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Embedding Model Settings
# sentence-transformers/all-MiniLM-L6-v2 produces 384-dimensional dense vectors
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Vector & Hybrid Search Settings
# Work reduction per answer: top 3 chunks, ~400 char truncation per chunk
TOP_K = 3
CHUNK_CONTEXT_CHAR_LIMIT = 400
MAX_CHAT_HISTORY_TURNS = 2
SIMILARITY_THRESHOLD = 0.22

# Hybrid Retrieval (BM25 + Semantic Cosine Similarity)
USE_HYBRID_SEARCH = True
BM25_WEIGHT = 0.4
SEMANTIC_WEIGHT = 0.6

# Answer Confidence Thresholds (empirically calibrated via calibrate_confidence.py)
CONFIDENCE_HIGH_THRESHOLD = 0.58
CONFIDENCE_MED_THRESHOLD = 0.35

# Key Terms Extraction Settings
TOP_KEY_TERMS_COUNT = 10

# Alternative embedding model for evaluation comparison
ALTERNATIVE_EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Summarization: larger page groups (~6 pages per group) to minimize model calls
PAGES_PER_SUMMARY_GROUP = 6


# Sample Documents
SAMPLE_DOCS_DIR = BASE_DIR / "sample_documents"
SAMPLE_PDF_PATH = SAMPLE_DOCS_DIR / "ai_renewable_energy_report.pdf"

# Generation Execution Settings
MAX_NEW_TOKENS = 200
DO_SAMPLE = False  # Deterministic decoding for faithful Q&A
TEMPERATURE = 0.0


# System Prompt enforcing strict factual grounding and structured 3-6 sentences
SYSTEM_PROMPT = (
    "You are a helpful assistant for question answering over PDF documents.\n"
    "Answer ONLY from the provided context below.\n"
    "Provide your answer in 3-6 clear, informative sentences (use bullet points for lists if appropriate).\n"
    "Do not invent facts, extrapolate, or assume outside information.\n"
    "If the context does not contain the answer, reply exactly: 'Answer not found in the document.'"
)

# User Prompt Template injecting context and question
USER_PROMPT_TEMPLATE = (
    "Context:\n"
    "{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

# Map-Reduce Document Summarization Prompts
MAP_SUMMARY_PROMPT = (
    "Summarize the following document excerpt from {page_range} in 2-3 concise sentences.\n"
    "Rules: Preserve key facts, definitions, names, numbers, and dates exactly. "
    "Do not add anything that is not in the text. Avoid filler phrases like 'this section discusses'.\n\n"
    "Excerpt ({page_range}):\n{context}\n\nGroup Summary:"
)

REDUCE_SUMMARY_DETAILED_PROMPT = (
    "Combine the following section summaries from the document into a structured executive summary of about 150-200 words.\n"
    "Rules:\n"
    "1. Use ONLY the provided information. Do not invent or assume outside facts.\n"
    "2. Keep numbers, names, and technical terms unchanged.\n"
    "3. Do not use filler phrases like 'this document discusses' or 'in conclusion'.\n\n"
    "Format strictly as follows:\n"
    "Overview: [One clear sentence describing what the document is about]\n\n"
    "Key Topics:\n"
    "- [Topic 1 summary] (pp. X-Y)\n"
    "- [Topic 2 summary] (pp. X-Y)\n"
    "- [Topic 3 summary] (pp. X-Y)\n"
    "- [Topic 4 summary] (pp. X-Y)\n"
    "- [Topic 5 summary] (pp. X-Y)\n\n"
    "Key Terms & Figures: [One line listing 4-6 essential terms, metrics, or numbers worth remembering]\n\n"
    "Section Summaries:\n{context}\n\nStructured Summary:"
)

REDUCE_SUMMARY_BRIEF_PROMPT = (
    "Combine the following section summaries from the document into a concise summary of about 80 words.\n"
    "Rules:\n"
    "1. Use ONLY the provided information. Keep numbers and technical terms unchanged.\n"
    "2. No filler phrases.\n\n"
    "Format strictly as follows:\n"
    "Overview: [One concise sentence describing what the document is about]\n\n"
    "Key Takeaways:\n"
    "- [Point 1] (pp. X-Y)\n"
    "- [Point 2] (pp. X-Y)\n"
    "- [Point 3] (pp. X-Y)\n\n"
    "Key Terms: [3-5 essential terms or figures]\n\n"
    "Section Summaries:\n{context}\n\nBrief Summary:"
)

# Suggested question generation prompt per chunk
CHUNK_QUESTION_PROMPT = (
    "Write exactly one clear, specific question that the following passage directly answers.\n"
    "Rules:\n"
    "- The question MUST be directly answerable from the passage.\n"
    "- Prefer 'What', 'How', 'Why', 'Define', or 'Compare'.\n"
    "- Do NOT ask yes/no questions.\n"
    "- Do NOT ask vague questions like 'What is this document about?'.\n"
    "- Output ONLY the question string, nothing else.\n\n"
    "Passage:\n{context}\n\nQuestion:"
)

# Follow-up question rewriting template
REWRITE_PROMPT_TEMPLATE = (
    "Given the conversation history and a follow-up question, rewrite the follow-up question "
    "into a standalone search question that can be understood without the conversation history.\n"
    "Output ONLY the rewritten standalone question with no preamble.\n\n"
    "Conversation History:\n"
    "{history}\n\n"
    "Follow-up Question: {question}\n\n"
    "Standalone Question:"
)

