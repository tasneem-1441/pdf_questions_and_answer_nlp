"""
Clean, modern ChatGPT/Claude-style Gradio interface for PDF Question Answering using Hybrid RAG.
Features:
- FORCED Clean Light Theme (universal light mode even with dark OS/browser settings)
- Custom gr.themes.Soft theme + injected JS + CSS variables locking light colors
- Left sidebar (~320px) in a white rounded card:
    - Document heading + compact PDF upload
    - File info line (name, pages, chunks)
    - 5 full-width, 2-line clamped suggested questions
    - Key terms (top 10 salient keywords/entities)
    - Answer style toggle: Short (1-2 sentences) / Detailed (3-6 sentences)
    - Summarize button + Brief/Detailed dropdown (precomputed in background, single-call for <=15 pages)
    - Clear chat & Download chat (.docx / .txt) via gr.DownloadButton
- Main chat area (~remaining width, 1400px container):
    - 65vh Chatbot with friendly empty state
    - Citations ("Sources: pages X, Y") & Confidence labels ("Confidence: High / Medium / Low")
    - Thumbs up / thumbs down feedback logging to feedback_log.csv
    - Input row: Textbox (80%) + Send (12%) + Stop button (8%)
    - Collapsed Source Page Preview accordion with PyMuPDF yellow evidence highlighting
    - Collapsed Retrieved Chunks accordion
    - Footer with active generator backend label
"""

import logging
import re
import time
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import gradio as gr

from config import (
    TOP_K,
    SIMILARITY_THRESHOLD,
    CACHE_DIR,
    GENERATOR_BACKEND,
    LOCAL_MODEL_NAME,
    GEMINI_MODEL_NAME,
    GEMINI_API_KEY,
    PAGES_PER_SUMMARY_GROUP
)
from loader import PDFLoadError, EmptyPDFError, ScannedPDFError, EncryptedPDFError
from retriever import FAISSRetriever
from embedder import Embedder
from generator import Generator, export_chat_transcript, log_feedback

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("rag_app")

# Global singleton handles
_retriever: Optional[FAISSRetriever] = None
_generator: Optional[Generator] = None
_embedder: Optional[Embedder] = None


def get_retriever() -> FAISSRetriever:
    global _retriever
    if _retriever is None:
        _retriever = FAISSRetriever()
    return _retriever


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder.get_instance()
    return _embedder


def get_generator() -> Generator:
    global _generator
    if _generator is None:
        _generator = Generator.get_instance()
    return _generator


def format_source_chunks(chunks: List[Dict[str, Any]]) -> str:
    """Formats retrieved chunks as plain readable text with page numbers and similarity scores."""
    if not chunks:
        return "No relevant source chunks found for this query."

    formatted = []
    for i, chunk in enumerate(chunks, start=1):
        page_no = chunk.get("page_number", "?")
        score = chunk.get("similarity_score", 0.0)
        h_score = chunk.get("hybrid_score", score)
        text = chunk.get("text", "").strip()
        formatted.append(f"--- Chunk {i} | Page {page_no} (Cosine: {score:.3f}, Hybrid: {h_score:.3f}) ---\n{text}")

    return "\n\n".join(formatted)


def extract_cited_pages(chunks: List[Dict[str, Any]]) -> List[int]:
    """Returns sorted unique page numbers from retrieved chunks."""
    if not chunks:
        return []
    return sorted(list(set(c["page_number"] for c in chunks[:TOP_K] if "page_number" in c)))


def handle_pdf_upload(file_obj):
    """
    Handles PDF upload:
    1. Indexes document into FAISS / BM25 hybrid store.
    2. Extracts top 10 salient key terms.
    3. Generates 5 verified example questions across the document.
    4. BUG 1 FIX: Spawns background thread to precompute Detailed summary right after indexing.
    5. Resets conversation history and populates UI controls.
    """
    retriever = get_retriever()
    embedder = get_embedder()
    generator = get_generator()

    hidden_btns = [
        gr.update(value="", visible=False),
        gr.update(value="", visible=False),
        gr.update(value="", visible=False),
        gr.update(value="", visible=False),
        gr.update(value="", visible=False)
    ]

    if file_obj is None:
        retriever.index = None
        retriever.chunks = []
        return (
            "No PDF uploaded yet.",
            "None",
            [],
            *hidden_btns,
            "*Upload a document to extract key terms.*",
            "",
            gr.update(choices=[], value=None),
            None,
            ""
        )

    file_path = Path(file_obj.name if hasattr(file_obj, "name") else file_obj)

    try:
        t0 = time.time()
        stats = retriever.index_pdf(file_path)
        index_time = time.time() - t0
        status_line = f"**{stats['pdf_name']}** • {stats['total_pages']} pages, {stats['total_chunks']} chunks"

        # Extract top 10 key terms
        key_terms = retriever.extract_key_terms(count=10)
        key_terms_str = ", ".join(key_terms) if key_terms else "No key terms extracted."

        # Generate 5 verified example questions (instant if cached)
        t_q = time.time()
        questions = generator.generate_5_example_questions(retriever, embedder)
        print(f"[TIMING] Indexing: {index_time:.3f}s | Questions ({len(questions)}): {time.time() - t_q:.3f}s")

        # BUG 1 FIX: Start summary precomputation in background thread right after indexing
        generator.precompute_summary_async(retriever, length_option="Detailed")

        btn_updates = []
        for i in range(5):
            if i < len(questions):
                btn_updates.append(gr.update(value=questions[i], visible=True))
            else:
                btn_updates.append(gr.update(value="", visible=False))

        return (
            status_line,
            stats["pdf_name"],
            [],
            *btn_updates,
            key_terms_str,
            "",
            gr.update(choices=[], value=None),
            None,
            ""
        )

    except EmptyPDFError:
        return ("The uploaded PDF contains no readable pages.", "None", [], *hidden_btns, "*None*", "", gr.update(choices=[], value=None), None, "")
    except ScannedPDFError:
        return ("No text could be extracted. The PDF may be scanned or image-only.", "None", [], *hidden_btns, "*None*", "", gr.update(choices=[], value=None), None, "")
    except EncryptedPDFError:
        return ("This PDF is password-protected or encrypted.", "None", [], *hidden_btns, "*None*", "", gr.update(choices=[], value=None), None, "")
    except PDFLoadError as e:
        return (f"Error reading PDF: {str(e)}", "None", [], *hidden_btns, "*None*", "", gr.update(choices=[], value=None), None, "")
    except Exception as e:
        logger.exception("Unexpected error indexing PDF:")
        return (f"Error indexing PDF: {str(e)}", "None", [], *hidden_btns, "*None*", "", gr.update(choices=[], value=None), None, "")


def handle_chat_turn(
    user_message: str,
    history: List[Dict[str, str]],
    answer_style: str = "Detailed (3-6 sentences)"
):
    """
    Handles a user question turn with token-by-token streaming into the chat:
    1. Validates query and document status.
    2. Applies heuristic follow-up rewriting (only on short/pronoun queries).
    3. Retrieves top chunks with hybrid search and similarity filtering.
    4. FEATURE 2: Respects Short / Detailed answer style toggle.
    5. FEATURE 3: Supports Stop button cancellation cleanly.
    6. FEATURE 1: Highlights evidence from top chunk on page preview in yellow.
    """
    clean_msg = (user_message or "").strip()
    history = history or []

    if not clean_msg:
        new_hist = history + [{"role": "assistant", "content": "Please enter a question."}]
        yield "", new_hist, gr.update(interactive=True), gr.update(interactive=False), "", gr.update(), None, ""
        return

    retriever = get_retriever()
    if retriever.index is None or not retriever.chunks:
        new_hist = history + [
            {"role": "user", "content": clean_msg},
            {"role": "assistant", "content": "Please upload a PDF document first."}
        ]
        yield "", new_hist, gr.update(interactive=True), gr.update(interactive=False), "", gr.update(), None, ""
        return

    # Append user question to history and show initial streaming indicator
    history_with_user = history + [{"role": "user", "content": clean_msg}]
    history_with_searching = history_with_user + [{"role": "assistant", "content": "Thinking..."}]
    yield "", history_with_searching, gr.update(interactive=False), gr.update(interactive=True), "", gr.update(), None, ""

    t_start = time.time()
    generator = get_generator()

    # Heuristic follow-up rewriting
    t_rw = time.time()
    standalone_query = generator.rewrite_question_with_history(clean_msg, history)
    t_rewrite = time.time() - t_rw

    # Retrieval (hybrid BM25 + dense semantic cosine similarity)
    t_ret = time.time()
    retrieved_chunks = retriever.retrieve(
        standalone_query,
        top_k=TOP_K,
        similarity_threshold=SIMILARITY_THRESHOLD
    )
    t_retrieval = time.time() - t_ret
    sources_text = format_source_chunks(retrieved_chunks)
    cited_pages = extract_cited_pages(retrieved_chunks)
    top_chunk_text = retrieved_chunks[0]["text"] if retrieved_chunks else ""

    # Token-by-token streaming generation
    t_gen = time.time()
    last_text = ""
    for partial_answer in generator.generate_answer_stream(clean_msg, retrieved_chunks, answer_style=answer_style):
        last_text = partial_answer
        yield "", history_with_user + [{"role": "assistant", "content": partial_answer}], gr.update(interactive=False), gr.update(interactive=True), sources_text, gr.update(), None, top_chunk_text

    t_generation = time.time() - t_gen
    t_total = time.time() - t_start

    print(
        f"[TIMING] Query: '{clean_msg[:35]}' | "
        f"Rewrite: {t_rewrite:.3f}s | Retrieval: {t_retrieval:.3f}s | "
        f"Generation: {t_generation:.3f}s | Total: {t_total:.3f}s | "
        f"Style: {answer_style} | Backend: {generator.backend.upper()}"
    )

    # FEATURE 1: Render first cited page with yellow evidence highlight
    preview_img = None
    page_dropdown_update = gr.update(choices=[], value=None)
    if cited_pages:
        choices = [f"Page {p}" for p in cited_pages]
        first_page = cited_pages[0]
        preview_img = retriever.render_page_preview(first_page, highlight_text=top_chunk_text)
        page_dropdown_update = gr.update(choices=choices, value=f"Page {first_page}")

    final_history = history_with_user + [{"role": "assistant", "content": last_text}]
    yield "", final_history, gr.update(interactive=True), gr.update(interactive=False), sources_text, page_dropdown_update, preview_img, top_chunk_text


def handle_page_preview_change(page_choice: str, top_chunk_text: str = ""):
    """FEATURE 1: Renders the selected PDF page to PNG preview with evidence highlighted in yellow."""
    if not page_choice:
        return None
    match = re.search(r"\d+", str(page_choice))
    if not match:
        return None
    page_num = int(match.group(0))
    retriever = get_retriever()
    img_path = retriever.render_page_preview(page_num, highlight_text=top_chunk_text)
    return img_path


def handle_summarize(history: List[Dict[str, str]], length_option: str):
    """
    BUG 1 FIX: Executes fast faithful document summarization:
    - Precomputed in background right after upload for Detailed
    - Instant return if cached
    - Single-call summarization for <= 15 pages (eliminates slow map-reduce overhead)
    - Streams token by token into chat
    """
    history = history or []
    retriever = get_retriever()

    if retriever.index is None or not retriever.chunks:
        new_hist = history + [{"role": "assistant", "content": "Please upload a PDF document first."}]
        yield new_hist, ""
        return

    t0 = time.time()
    generator = get_generator()
    history_with_searching = history + [{"role": "assistant", "content": f"Preparing summary ({length_option})..."}]
    yield history_with_searching, ""

    last_text = ""
    for partial_summary in generator.generate_summary_stream(retriever, length_option=length_option):
        last_text = partial_summary
        yield history + [{"role": "assistant", "content": f"**Document Summary ({length_option}):**\n\n{partial_summary}"}], partial_summary

    t_sum = time.time() - t0
    print(f"[TIMING] Summary ({length_option}): {t_sum:.2f}s | Pages: {len(retriever.chunks)} chunks | Backend: {generator.backend.upper()}")
    yield history + [{"role": "assistant", "content": f"**Document Summary ({length_option}):**\n\n{last_text}"}], last_text


def handle_stop_generation():
    """FEATURE 3: Stops the active text generation and re-enables Send button."""
    generator = get_generator()
    generator.stop_generation()
    return gr.update(interactive=True), gr.update(interactive=False)


def handle_clear_chat():
    """Resets only conversation history, sources, and preview, retaining loaded PDF."""
    return [], "", "<p style='color: #6B7280; font-size: 14px;'>Retrieved chunks will appear here after asking a question.</p>", gr.update(choices=[], value=None), None, ""


def handle_download_chat(history: List[Dict[str, Any]], doc_name: str, summary_text: str, format_type: str):
    """
    BUG 2 FIX: Reliable export flow using gr.DownloadButton.
    Safely exports history to .docx or .txt with friendly warning for empty chat.
    """
    if not history or len(history) == 0:
        gr.Warning("No conversation history to download. Please ask a question first.")
        return None, gr.update(visible=False)

    fmt = "docx" if "docx" in str(format_type).lower() else "txt"
    file_path = export_chat_transcript(
        history=history,
        doc_name=doc_name or "Document",
        summary_text=summary_text,
        format_type=fmt
    )
    if file_path and Path(file_path).exists():
        return file_path, gr.update(value=file_path, visible=True)

    gr.Warning("Export could not be generated.")
    return None, gr.update(visible=False)


def handle_like_feedback(x: gr.LikeData, history: List[Dict[str, Any]], doc_name: str):
    """Appends thumbs up/down user rating to feedback_log.csv."""
    try:
        idx = x.index[0] if isinstance(x.index, (list, tuple)) else x.index
        rating = "like" if x.liked else "dislike"
        answer = history[idx].get("content", "") if 0 <= idx < len(history) else str(x.value)

        # Preceding user question
        question = ""
        if idx > 0 and history[idx - 1].get("role") == "user":
            question = history[idx - 1].get("content", "")

        # Extract cited pages
        page_matches = re.findall(r"pages?\s+([0-9,\s]+)", answer, re.IGNORECASE)
        pages_str = page_matches[0].strip() if page_matches else ""

        log_feedback(
            doc_name=doc_name or "Document",
            question=question,
            answer=answer,
            pages=pages_str,
            rating=rating
        )
        gr.Info(f"Feedback recorded ({'👍 Liked' if x.liked else '👎 Disliked'}). Thank you!")
    except Exception as e:
        logger.warning("Error logging like feedback: %s", e)


# Pre-warm models into memory at startup
print("[STARTUP] Pre-warming retriever, embedder, and generator models...")
_ = get_embedder()
_gen = get_generator()
active_backend_label = "Gemini 2.5 Flash" if _gen.backend == "gemini" else "Local Qwen 0.5B"
print(f"[STARTUP] System ready! Active Backend: {_gen.backend.upper()} ({active_backend_label})")


# JavaScript strictly stripping dark mode classes on load and observing DOM mutations
light_mode_js = """
function() {
    function enforceLight() {
        document.documentElement.classList.remove('dark');
        document.body.classList.remove('dark');
        document.documentElement.setAttribute('data-theme', 'light');
    }
    function updateTooltips() {
        document.querySelectorAll('.suggestion-btn').forEach(function(btn) {
            const text = (btn.innerText || btn.textContent || '').trim();
            if (text && (!btn.title || btn.title !== text)) {
                btn.title = text;
            }
        });
    }
    enforceLight();
    updateTooltips();
    window.addEventListener('DOMContentLoaded', function() {
        enforceLight();
        updateTooltips();
    });
    document.addEventListener('mouseover', function(e) {
        if (e.target && e.target.closest && e.target.closest('.suggestion-btn')) {
            const btn = e.target.closest('.suggestion-btn');
            const text = (btn.innerText || btn.textContent || '').trim();
            if (text && (!btn.title || btn.title !== text)) {
                btn.title = text;
            }
        }
    });
    const observer = new MutationObserver(function(mutations) {
        if (document.documentElement.classList.contains('dark') || document.body.classList.contains('dark')) {
            enforceLight();
        }
        updateTooltips();
    });
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
    observer.observe(document.body, { attributes: true, childList: true, subtree: true });
}
"""

# Custom CSS ensuring strict Light Theme (#F7F8FA background, #FFFFFF cards, 16px font, no dark elements)
custom_css = """
/* UNIVERSAL LIGHT MODE OVERRIDES - lock all modes to clean light values */
:root, .dark, body, html {
    --body-background-fill: #F7F8FA !important;
    --background-fill-primary: #F7F8FA !important;
    --background-fill-secondary: #FFFFFF !important;
    --block-background-fill: #FFFFFF !important;
    --panel-background-fill: #FFFFFF !important;
    --color-background-primary: #F7F8FA !important;
    --body-text-color: #111827 !important;
    --body-text-color-subdued: #6B7280 !important;
    --border-color-primary: #E5E7EB !important;
    --border-color-accent: #2563EB !important;
    --primary-500: #2563EB !important;
    --primary-600: #1D4ED8 !important;
    color-scheme: light !important;
}

body, html {
    background-color: #F7F8FA !important;
    color: #111827 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    font-size: 16px !important;
    margin: 0 !important;
    padding: 0 !important;
    min-height: 100% !important;
    overflow-y: auto !important;
}

/* Base font 16px everywhere, strictly nothing under 14px */
* {
    font-size: max(14px, inherit);
}

/* Centered container with max width 1400px */
.gradio-container {
    max-width: 1400px !important;
    width: 100% !important;
    margin: 0 auto !important;
    padding: 16px 24px !important;
    background-color: #F7F8FA !important;
    box-sizing: border-box !important;
    overflow: visible !important;
}

/* Header styling */
.header-box {
    text-align: center;
    padding: 8px 0 16px 0;
}
.header-box h1 {
    font-size: 24px !important;
    font-weight: 700 !important;
    color: #111827 !important;
    letter-spacing: -0.02em;
    margin: 0 0 4px 0 !important;
}
.header-box p {
    font-size: 16px !important;
    color: #6B7280 !important;
    margin: 0 !important;
}

/* Main app two-column row */
.main-app-row {
    display: flex !important;
    flex-direction: row !important;
    align-items: flex-start !important;
    gap: 20px !important;
    width: 100% !important;
    overflow: visible !important;
}

/* Sidebar panel (~330px wide, fits all content naturally without cutting off) */
.sidebar-panel {
    width: 330px !important;
    min-width: 310px !important;
    max-width: 350px !important;
    flex-shrink: 0 !important;
    background: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
    padding: 16px !important;
    box-sizing: border-box !important;
    height: auto !important;
    min-height: auto !important;
    max-height: none !important;
    overflow: visible !important;
}
.sidebar-heading {
    font-size: 14.5px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
    color: #4B5563 !important;
    margin: 4px 0 8px 0 !important;
}
.sidebar-heading:first-of-type {
    margin-top: 0 !important;
}
.key-terms-heading {
    margin-top: 20px !important;
}

/* Thin divider line above sections */
.sidebar-divider {
    border-top: 1px solid #E5E7EB !important;
    margin: 12px 0 8px 0 !important;
    width: 100% !important;
}

/* Compact upload box */
.compact-upload {
    min-height: 85px !important;
    max-height: 105px !important;
    background: #FAFAFA !important;
    border: 1px dashed #D1D5DB !important;
    border-radius: 10px !important;
    margin-bottom: 2px !important;
}
.compact-upload > div {
    min-height: 75px !important;
    padding: 6px !important;
}

/* File info line */
.file-info-line {
    font-size: 14px !important;
    color: #374151 !important;
    margin: 4px 0 8px 0 !important;
    line-height: 1.4 !important;
}

/* Suggested questions buttons - auto height, multiline, readable, no clamp, no trailing ... */
.suggestion-btn {
    height: auto !important;
    min-height: 44px !important;
    max-height: none !important;
    white-space: normal !important;
    word-break: break-word !important;
    overflow-wrap: break-word !important;
    text-align: left !important;
    justify-content: flex-start !important;
    align-items: center !important;
    line-height: 1.4 !important;
    padding: 10px 12px !important;
    font-size: 14.5px !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    margin-bottom: 8px !important;
    width: 100% !important;
    background: #F9FAFB !important;
    border: 1px solid #E5E7EB !important;
    color: #1F2937 !important;
    display: block !important;
    -webkit-line-clamp: unset !important;
    -webkit-box-orient: unset !important;
    overflow: visible !important;
    text-overflow: clip !important;
    transition: all 0.15s ease-in-out !important;
    cursor: pointer !important;
    box-sizing: border-box !important;
}
.suggestion-btn:hover {
    background: #EFF6FF !important;
    border-color: #BFDBFE !important;
    color: #1D4ED8 !important;
}
.suggestion-btn span,
.suggestion-btn * {
    white-space: normal !important;
    word-break: break-word !important;
    overflow-wrap: break-word !important;
    text-overflow: clip !important;
    overflow: visible !important;
    line-height: 1.4 !important;
    font-size: 14.5px !important;
    display: inline !important;
    text-align: left !important;
}

/* Key terms plain line */
.key-terms-box {
    font-size: 14.5px !important;
    color: #374151 !important;
    line-height: 1.5 !important;
    background: #F9FAFB !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 8px !important;
    padding: 8px 10px !important;
    margin-bottom: 8px !important;
}

/* Gradio Form wrapper in sidebar: eliminate the grey strip underneath controls */
.sidebar-panel .form,
.sidebar-panel div.form {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
    gap: 12px !important;
}

/* Dropdown outer blocks: completely transparent, NO border, NO shadow, NO background, NO padding */
.sidebar-dropdown,
#answer-style-dd,
#summary-length-dd,
#export-format-dd {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 0 12px 0 !important;
    width: 100% !important;
    min-width: 0 !important;
}

/* Dropdown container: stack label directly ABOVE dropdown */
.sidebar-dropdown > .container,
.sidebar-dropdown .container,
#answer-style-dd .container,
#summary-length-dd .container,
#export-format-dd .container {
    display: flex !important;
    flex-direction: column !important;
    align-items: stretch !important;
    justify-content: flex-start !important;
    gap: 6px !important;
    padding: 0 !important;
    margin: 0 !important;
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    width: 100% !important;
}

/* Dropdown labels: plain text above, 14px muted grey, NO blue background or colored tag */
.sidebar-dropdown label,
.sidebar-dropdown [data-testid="block-info"],
.sidebar-dropdown .block-label,
.sidebar-dropdown span.block-label,
.sidebar-dropdown label > span,
#answer-style-dd label,
#answer-style-dd [data-testid="block-info"],
#summary-length-dd label,
#summary-length-dd [data-testid="block-info"],
#export-format-dd label,
#export-format-dd [data-testid="block-info"] {
    display: block !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    color: #6B7280 !important;
    margin: 0 0 6px 0 !important;
    padding: 0 !important;
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    border-radius: 0 !important;
    line-height: 1.4 !important;
    width: 100% !important;
    text-transform: none !important;
    letter-spacing: normal !important;
}

/* Visible dropdown control: Border on EXACTLY ONE element only (.wrap) */
.sidebar-dropdown .wrap,
#answer-style-dd .wrap,
#summary-length-dd .wrap,
#export-format-dd .wrap {
    border: 1px solid #E5E7EB !important;
    border-radius: 8px !important;
    background: #FFFFFF !important;
    background-color: #FFFFFF !important;
    min-height: 42px !important;
    height: 42px !important;
    width: 100% !important;
    box-sizing: border-box !important;
    padding: 0 12px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    box-shadow: none !important;
    position: relative !important;
    cursor: pointer !important;
    transition: border-color 0.15s ease !important;
}
.sidebar-dropdown .wrap:hover,
.sidebar-dropdown .wrap:focus-within,
#answer-style-dd .wrap:hover,
#summary-length-dd .wrap:hover,
#export-format-dd .wrap:hover {
    border-color: #BFDBFE !important;
}

/* Reset all nested wrappers so NO stacked/nested outlines or backgrounds appear */
.sidebar-dropdown .wrap-inner,
.sidebar-dropdown .secondary-wrap,
.sidebar-dropdown input,
#answer-style-dd .wrap-inner,
#answer-style-dd .secondary-wrap,
#answer-style-dd input,
#summary-length-dd .wrap-inner,
#summary-length-dd .secondary-wrap,
#summary-length-dd input,
#export-format-dd .wrap-inner,
#export-format-dd .secondary-wrap,
#export-format-dd input {
    border: none !important;
    background: transparent !important;
    background-color: transparent !important;
    box-shadow: none !important;
    border-radius: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    outline: none !important;
    box-sizing: border-box !important;
}

/* Dropdown text: selected value clearly visible */
.sidebar-dropdown input,
.sidebar-dropdown .single-select,
.sidebar-dropdown span.single-select,
.sidebar-dropdown .item-text,
.sidebar-dropdown input[role="combobox"],
#answer-style-dd input,
#summary-length-dd input,
#export-format-dd input {
    font-size: 14.5px !important;
    color: #111827 !important;
    font-weight: 500 !important;
    line-height: 1.4 !important;
}

.sidebar-dropdown svg,
.sidebar-dropdown .dropdown-arrow {
    color: #6B7280 !important;
}

/* Dropdown popup menu */
.sidebar-dropdown ul,
.sidebar-dropdown .options,
.sidebar-dropdown ul.options,
#answer-style-dd ul,
#summary-length-dd ul,
#export-format-dd ul,
ul.options {
    background: #FFFFFF !important;
    background-color: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
    z-index: 1000 !important;
    padding: 4px 0 !important;
    margin-top: 4px !important;
}
.sidebar-dropdown ul li,
.sidebar-dropdown .options li,
ul.options li {
    font-size: 14.5px !important;
    color: #111827 !important;
    padding: 8px 12px !important;
    background: #FFFFFF !important;
    cursor: pointer !important;
    transition: background-color 0.1s ease !important;
}
.sidebar-dropdown ul li:hover,
.sidebar-dropdown .options li:hover,
ul.options li:hover {
    background-color: #EFF6FF !important;
    color: #1D4ED8 !important;
}

/* Summarize Document button: full width, 40px tall */
.summarize-btn-40 {
    width: 100% !important;
    height: 40px !important;
    min-height: 40px !important;
    max-height: 40px !important;
    font-size: 14.5px !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    background: #F3F4F6 !important;
    border: 1px solid #D1D5DB !important;
    color: #111827 !important;
    margin-top: 0px !important;
    margin-bottom: 4px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
}
.summarize-btn-40:hover {
    background: #E5E7EB !important;
}

/* Clear and Download row: two equal-width buttons, 40px tall, same size, same font */
.clear-download-row {
    margin-top: 0px !important;
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    gap: 10px !important;
    width: 100% !important;
}
.clear-download-row > * {
    flex: 1 1 0% !important;
    min-width: 0 !important;
    margin: 0 !important;
    width: 50% !important;
}
.action-btn-40 {
    height: 40px !important;
    min-height: 40px !important;
    max-height: 40px !important;
    font-size: 14.5px !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    box-sizing: border-box !important;
    margin: 0 !important;
    padding: 0 8px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 100% !important;
    background: #F3F4F6 !important;
    border: 1px solid #D1D5DB !important;
    color: #111827 !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
}
.action-btn-40:hover {
    background: #E5E7EB !important;
}

/* Main chat card: at least 800px wide, container up to 1400px */
.main-chat-card {
    flex: 1 1 800px !important;
    min-width: 800px !important;
    background: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
    padding: 18px !important;
    box-sizing: border-box !important;
}

.main-chat {
    border-radius: 10px !important;
    border: 1px solid #E5E7EB !important;
    background: #FFFFFF !important;
    height: 65vh !important;
    min-height: 520px !important;
}

/* Centered Welcome Card for Empty Chat */
.welcome-card-box {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
    margin: 40px auto !important;
    max-width: 440px !important;
    padding: 28px 32px !important;
    background: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
}
.welcome-card-icon {
    font-size: 32px !important;
    line-height: 1 !important;
    margin-bottom: 12px !important;
}
.welcome-card-title {
    font-size: 20px !important;
    font-weight: 700 !important;
    color: #111827 !important;
    margin: 0 0 16px 0 !important;
    letter-spacing: -0.01em !important;
}
.welcome-card-steps {
    display: flex !important;
    flex-direction: column !important;
    gap: 8px !important;
    width: 100% !important;
    text-align: left !important;
}
.welcome-card-step {
    font-size: 15px !important;
    color: #374151 !important;
    background: #F9FAFB !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 8px !important;
    padding: 9px 14px !important;
    line-height: 1.45 !important;
}
.welcome-card-step strong {
    color: #2563EB !important;
    font-weight: 600 !important;
    margin-right: 4px !important;
}

/* Chatbot message bubbles */
div[data-testid="user"], .user-message, .message.user, .user-row .message-bubble {
    background-color: #EAF2FF !important;
    color: #111827 !important;
    border: 1px solid #C7D9F8 !important;
    border-radius: 12px !important;
    font-size: 16px !important;
    line-height: 1.55 !important;
}
div[data-testid="bot"], .bot-message, .message.bot, .bot-row .message-bubble {
    background-color: #FFFFFF !important;
    color: #111827 !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
    font-size: 16px !important;
    line-height: 1.55 !important;
}

/* Input row styling: Textbox fills ~80%, Send & Stop share rest, all same height (at least 48px) */
.chat-input-row {
    margin-top: 12px !important;
    display: flex !important;
    flex-direction: row !important;
    align-items: stretch !important;
    gap: 10px !important;
    width: 100% !important;
}
.chat-textbox {
    flex: 8 1 0% !important;
    margin: 0 !important;
}
.chat-textbox > label {
    margin: 0 !important;
    padding: 0 !important;
    height: 100% !important;
}
.chat-textbox textarea {
    height: 50px !important;
    min-height: 50px !important;
    max-height: 50px !important;
    font-size: 16px !important;
    border-radius: 8px !important;
    border: 1px solid #D1D5DB !important;
    background: #FFFFFF !important;
    color: #111827 !important;
    padding: 12px 14px !important;
    box-sizing: border-box !important;
    line-height: 24px !important;
    resize: none !important;
}
.chat-textbox textarea:focus {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.15) !important;
}
.chat-send-btn {
    flex: 1 1 0% !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    height: 50px !important;
    min-height: 50px !important;
    background: #2563EB !important;
    color: #FFFFFF !important;
    border: none !important;
    box-sizing: border-box !important;
    margin: 0 !important;
    cursor: pointer !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: background 0.15s ease !important;
}
.chat-send-btn:hover {
    background: #1D4ED8 !important;
}
.chat-stop-btn {
    flex: 1 1 0% !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    height: 50px !important;
    min-height: 50px !important;
    box-sizing: border-box !important;
    margin: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

/* Thin light grey scrollbar */
*::-webkit-scrollbar {
    width: 6px !important;
    height: 6px !important;
}
*::-webkit-scrollbar-track {
    background: transparent !important;
}
*::-webkit-scrollbar-thumb {
    background-color: #D1D5DB !important;
    border-radius: 4px !important;
}
*::-webkit-scrollbar-thumb:hover {
    background-color: #9CA3AF !important;
}
* {
    scrollbar-width: thin !important;
    scrollbar-color: #D1D5DB transparent !important;
}

/* Accordions */
.accordion {
    border: 1px solid #E5E7EB !important;
    border-radius: 10px !important;
    background: #FFFFFF !important;
    margin-top: 10px !important;
}

/* Footer info */
.footer-text {
    text-align: center;
    color: #6B7280;
    font-size: 14.5px !important;
    margin-top: 14px;
    padding-top: 8px;
    border-top: 1px solid #E5E7EB;
}
"""

# Custom Soft theme with explicit light mode tokens
light_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="#F7F8FA",
    body_background_fill_dark="#F7F8FA",
    block_background_fill="#FFFFFF",
    block_background_fill_dark="#FFFFFF",
    block_border_color="#E5E7EB",
    block_border_color_dark="#E5E7EB",
    body_text_color="#111827",
    body_text_color_dark="#111827",
    body_text_color_subdued="#6B7280",
    body_text_color_subdued_dark="#6B7280",
    button_primary_background_fill="#2563EB",
    button_primary_background_fill_dark="#2563EB",
    button_primary_background_fill_hover="#1D4ED8",
    button_primary_background_fill_hover_dark="#1D4ED8",
    button_primary_text_color="#FFFFFF",
    button_primary_text_color_dark="#FFFFFF",
    button_secondary_background_fill="#F3F4F6",
    button_secondary_background_fill_dark="#F3F4F6",
    button_secondary_text_color="#111827",
    button_secondary_text_color_dark="#111827",
    border_color_primary="#E5E7EB",
    border_color_primary_dark="#E5E7EB",
    input_background_fill="#FFFFFF",
    input_background_fill_dark="#FFFFFF",
    input_border_color="#E5E7EB",
    input_border_color_dark="#E5E7EB"
)

# Centered Welcome Card HTML for empty chat placeholder
welcome_card_html = """
<div class="welcome-card-box">
    <div class="welcome-card-icon">📄</div>
    <div class="welcome-card-title">PDF Question Answering</div>
    <div class="welcome-card-steps">
        <div class="welcome-card-step"><strong>1.</strong> Upload a PDF</div>
        <div class="welcome-card-step"><strong>2.</strong> Click a suggested question or type your own</div>
        <div class="welcome-card-step"><strong>3.</strong> Check the cited page</div>
    </div>
</div>
"""

# App Blocks definition
with gr.Blocks(title="PDF Question Answering System") as demo:
    # State tracking
    doc_name_state = gr.State("")
    summary_state = gr.State("")
    top_chunk_state = gr.State("")

    # Top Header
    gr.HTML(
        """
        <div class="header-box">
            <h1>PDF Question Answering System</h1>
            <p>Upload a PDF, ask questions with hybrid search, inspect source pages, and generate faithful summaries.</p>
        </div>
        """
    )

    with gr.Row(elem_classes=["main-app-row"]):
        # LEFT SIDEBAR (~330px)
        with gr.Column(scale=1, min_width=310, elem_classes=["sidebar-panel"]):
            gr.HTML("<div class='sidebar-heading'>Document</div>")
            pdf_input = gr.File(
                file_types=[".pdf"],
                type="filepath",
                show_label=False,
                elem_classes=["compact-upload"]
            )
            pdf_status = gr.Markdown(value="*No PDF uploaded yet.*", elem_classes=["file-info-line"])

            gr.HTML("<div class='sidebar-heading'>Suggested Questions</div>")
            q_btn1 = gr.Button("", elem_classes=["suggestion-btn"], visible=False)
            q_btn2 = gr.Button("", elem_classes=["suggestion-btn"], visible=False)
            q_btn3 = gr.Button("", elem_classes=["suggestion-btn"], visible=False)
            q_btn4 = gr.Button("", elem_classes=["suggestion-btn"], visible=False)
            q_btn5 = gr.Button("", elem_classes=["suggestion-btn"], visible=False)

            gr.HTML("<div class='sidebar-heading key-terms-heading' style='margin-top: 20px !important;'>Key Terms</div>")
            key_terms_display = gr.Markdown(
                value="*Upload a document to extract key terms.*",
                elem_classes=["key-terms-box"]
            )

            # Options section (stacked vertically, each full width of sidebar)
            gr.HTML("<div class='sidebar-divider'></div><div class='sidebar-heading'>Options</div>")
            answer_style_selector = gr.Dropdown(
                label="Answer style",
                choices=["Detailed", "Short"],
                value="Detailed",
                min_width=0,
                interactive=True,
                elem_classes=["sidebar-dropdown"],
                elem_id="answer-style-dd"
            )
            summary_length = gr.Dropdown(
                label="Summary length",
                choices=["Detailed", "Brief"],
                value="Detailed",
                min_width=0,
                interactive=True,
                elem_classes=["sidebar-dropdown"],
                elem_id="summary-length-dd"
            )
            summarize_btn = gr.Button("Summarize Document", variant="secondary", elem_classes=["summarize-btn-40"])

            # Export section (stacked vertically: Export format dropdown full width, then equal-width Clear and Download row)
            gr.HTML("<div class='sidebar-divider'></div><div class='sidebar-heading'>Export Chat</div>")
            export_fmt = gr.Dropdown(
                label="Export format",
                choices=[".docx", ".txt"],
                value=".docx",
                min_width=0,
                interactive=True,
                elem_classes=["sidebar-dropdown"],
                elem_id="export-format-dd"
            )
            with gr.Row(elem_classes=["clear-download-row"], equal_height=True):
                clear_chat_btn = gr.Button("Clear", variant="secondary", min_width=0, scale=1, elem_classes=["action-btn-40"])
                download_chat_btn = gr.DownloadButton(
                    label="Download",
                    variant="secondary",
                    min_width=0,
                    scale=1,
                    elem_classes=["action-btn-40"]
                )

            download_file = gr.File(label="Download Transcript", visible=False)

        # MAIN CHAT AREA (at least 800px wide, container up to 1400px)
        with gr.Column(scale=3, min_width=800, elem_classes=["main-chat-card"]):
            chatbot = gr.Chatbot(
                show_label=False,
                placeholder=welcome_card_html,
                elem_classes=["main-chat"],
                height=580
            )

            # Input row: question textbox fills ~80% of row, Send and Stop share the rest, all same height (at least 48px). Enter sends.
            with gr.Row(elem_classes=["chat-input-row"]):
                question_input = gr.Textbox(
                    placeholder="Ask a question about the document...",
                    show_label=False,
                    lines=1,
                    max_lines=1,
                    scale=8,
                    min_width=0,
                    elem_classes=["chat-textbox"],
                    autofocus=True
                )
                send_btn = gr.Button("Send", variant="primary", scale=1, min_width=75, elem_classes=["chat-send-btn"])
                stop_btn = gr.Button("Stop", variant="stop", scale=1, min_width=75, interactive=False, elem_classes=["chat-stop-btn"])

            # FEATURE 1: Source Page Preview Accordion with Evidence Highlighting
            with gr.Accordion("Source Page Preview", open=False, elem_classes=["accordion"]):
                with gr.Row():
                    page_selector = gr.Dropdown(
                        label="Cited Page",
                        choices=[],
                        value=None,
                        scale=3,
                        interactive=True
                    )
                    view_page_btn = gr.Button("Render Page", scale=1, variant="secondary")
                page_image = gr.Image(
                    label="Page Image (with yellow highlighted evidence)",
                    type="filepath",
                    interactive=False,
                    height=480
                )

            # Retrieved Chunks Accordion
            with gr.Accordion("Retrieved chunks", open=False, elem_classes=["accordion"]):
                sources_output = gr.Textbox(
                    label="Source chunks used for latest answer",
                    placeholder="Retrieved chunks with page numbers and similarity scores will appear here...",
                    lines=5,
                    interactive=False
                )

    # Footer
    gr.HTML(
        f"<div class='footer-text'>Generator: {active_backend_label} • Hybrid Retrieval: BM25 (0.4) + Semantic (0.6)</div>"
    )

    # --- EVENT BINDINGS ---
    # PDF Upload
    pdf_input.change(
        fn=handle_pdf_upload,
        inputs=[pdf_input],
        outputs=[
            pdf_status,
            doc_name_state,
            chatbot,
            q_btn1,
            q_btn2,
            q_btn3,
            q_btn4,
            q_btn5,
            key_terms_display,
            sources_output,
            page_selector,
            page_image,
            top_chunk_state
        ]
    )

    # Question Submission with Send / Stop buttons
    chat_event = send_btn.click(
        fn=handle_chat_turn,
        inputs=[question_input, chatbot, answer_style_selector],
        outputs=[question_input, chatbot, send_btn, stop_btn, sources_output, page_selector, page_image, top_chunk_state]
    )

    submit_event = question_input.submit(
        fn=handle_chat_turn,
        inputs=[question_input, chatbot, answer_style_selector],
        outputs=[question_input, chatbot, send_btn, stop_btn, sources_output, page_selector, page_image, top_chunk_state]
    )

    # FEATURE 3: Stop Button Handler canceling active generation
    stop_btn.click(
        fn=handle_stop_generation,
        inputs=[],
        outputs=[send_btn, stop_btn],
        cancels=[chat_event, submit_event]
    )

    # Suggested Questions Buttons
    for q_btn in [q_btn1, q_btn2, q_btn3, q_btn4, q_btn5]:
        q_btn.click(
            fn=handle_chat_turn,
            inputs=[q_btn, chatbot, answer_style_selector],
            outputs=[question_input, chatbot, send_btn, stop_btn, sources_output, page_selector, page_image, top_chunk_state]
        )

    # FEATURE 1: Page Preview Dropdown & Button with Highlighted Evidence
    page_selector.change(
        fn=handle_page_preview_change,
        inputs=[page_selector, top_chunk_state],
        outputs=[page_image]
    )
    view_page_btn.click(
        fn=handle_page_preview_change,
        inputs=[page_selector, top_chunk_state],
        outputs=[page_image]
    )

    # BUG 1 FIX: Streaming Summarize Document
    summarize_btn.click(
        fn=handle_summarize,
        inputs=[chatbot, summary_length],
        outputs=[chatbot, summary_state]
    )

    # Clear Chat
    clear_chat_btn.click(
        fn=handle_clear_chat,
        inputs=[],
        outputs=[chatbot, question_input, sources_output, page_selector, page_image, top_chunk_state]
    )

    # BUG 2 FIX: gr.DownloadButton flow
    download_chat_btn.click(
        fn=handle_download_chat,
        inputs=[chatbot, doc_name_state, summary_state, export_fmt],
        outputs=[download_chat_btn, download_file]
    )

    # Thumbs up / down feedback logging
    chatbot.like(
        fn=handle_like_feedback,
        inputs=[chatbot, doc_name_state],
        outputs=[]
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=light_theme,
        css=custom_css,
        js=light_mode_js
    )
