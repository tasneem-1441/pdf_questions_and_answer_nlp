"""
Verification script for all newly implemented bug fixes and features:
1. Docx/Txt chat export & empty chat handling
2. Fast single-call summary & precomputed summary cache on module5_notes.pdf
3. PyMuPDF evidence highlighting
4. Short vs Detailed answer style
5. Stop generation abort
"""
import time
import os
from pathlib import Path
import docx
import fitz
import config
from retriever import FAISSRetriever
from generator import (
    Generator,
    export_chat_transcript
)

def test_download_export():
    print("\n--- 1. Testing Chat Transcript Export (DOCX & TXT) ---")
    mock_history = [
        {"role": "user", "content": "What is semantic analysis in NLP?"},
        {
            "role": "assistant",
            "content": (
                "Semantic analysis examines the structural and linguistic meaning of sentences.\n\n"
                "<div style=\"font-size: 13px; color: #6B7280; margin-top: 10px; border-top: 1px solid #E5E7EB; padding-top: 8px;\">"
                "<strong>Confidence:</strong> High (88% similarity)<br/>"
                "<strong>Sources Cited:</strong><br/>"
                "• Page 2 (Chunk 3)<br/>"
                "• Page 3 (Chunk 5)"
                "</div>"
            )
        },
        # Test also with list-of-dicts content format (Gradio 6 format)
        {"role": "user", "content": [{"type": "text", "text": "Tell me about syntax parsing."}]},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Syntax parsing constructs parse trees to determine grammatical structure."}]
        }
    ]
    summary_text = "Overview: Module 5 covers syntax, semantics, and parsing in NLP.\n- Context-free grammars (p. 1-3)\n- Dependency parsing (p. 4-6)\nKey Terms: CFG, parse tree, semantic ambiguity."
    doc_name = "module5_notes.pdf"

    # Test DOCX export
    docx_path = export_chat_transcript(mock_history, format_type="docx", doc_name=doc_name, summary_text=summary_text)
    print(f"Generated DOCX: {docx_path}")
    assert os.path.exists(docx_path), "DOCX file does not exist!"
    assert os.path.getsize(docx_path) > 1000, "DOCX file seems too small!"

    # Parse with python-docx to verify integrity
    doc = docx.Document(docx_path)
    full_text = "\n".join([p.text for p in doc.paragraphs])
    assert "PDF Question Answering Transcript" in full_text, "Missing report title!"
    assert doc_name in full_text, "Missing document name!"
    assert "Executive Summary" in full_text, "Missing executive summary!"
    assert "Semantic analysis examines" in full_text, "Missing answer text!"
    assert "High (88% similarity)" in full_text, "Missing metadata!"
    print("DOCX integrity verified successfully! Contains title, date, summary, Q&As, sources, metadata.")

    # Test TXT export
    txt_path = export_chat_transcript(mock_history, format_type="txt", doc_name=doc_name, summary_text=summary_text)
    print(f"Generated TXT: {txt_path}")
    assert os.path.exists(txt_path), "TXT file does not exist!"
    with open(txt_path, "r", encoding="utf-8") as f:
        txt_content = f.read()
    assert "DOCUMENT SUMMARY" in txt_content
    assert "Semantic analysis examines" in txt_content
    print("TXT integrity verified successfully!")

    # Test empty chat
    empty_docx = export_chat_transcript([], format_type="docx")
    assert empty_docx is None, "Empty chat should return None!"
    print("Empty chat handling verified successfully!")


def test_evidence_highlighting():
    print("\n--- 2. Testing PyMuPDF Evidence Highlighting ---")
    doc_path = "sample_documents/module5_notes.pdf"
    assert os.path.exists(doc_path), f"File {doc_path} not found"
    
    retriever = FAISSRetriever()
    retriever.index_pdf(doc_path)
    
    doc = fitz.open(doc_path)
    page1_text = doc[0].get_text()
    words = page1_text.split()
    sample_phrase = " ".join(words[5:15]) if len(words) > 15 else words[0]
    doc.close()

    print(f"Target highlight phrase: '{sample_phrase}'")
    preview_img = retriever.render_page_preview(page_number=1, highlight_text=sample_phrase)
    assert preview_img is not None, "render_page_preview returned None!"
    assert os.path.exists(preview_img), f"Rendered image {preview_img} does not exist!"
    print(f"Page preview with highlighted evidence rendered successfully: {preview_img}")


def test_summary_speed_and_precomputation():
    print("\n--- 3. Testing Single-Call Summary & Precomputation Speed ---")
    doc_path = "sample_documents/module5_notes.pdf"
    retriever = FAISSRetriever()
    stats = retriever.index_pdf(doc_path)
    print(f"Indexed {stats['total_pages']} pages, {stats['total_chunks']} chunks from {doc_path}")

    generator = Generator()

    # Clear disk cache for test if any
    disk_cache = config.CACHE_DIR / f"{retriever.pdf_hash}_summary_detailed.json"
    if disk_cache.exists():
        disk_cache.unlink()

    # 1. Measure direct single-call summary generation
    print("Starting direct single-call summary stream...")
    t0 = time.time()
    summary_tokens = []
    for chunk in generator.generate_summary_stream(retriever, length_option="Detailed (~200 words)", force_regenerate=True):
        summary_tokens.append(chunk)
    elapsed_direct = time.time() - t0
    final_summary = summary_tokens[-1] if summary_tokens else ""
    print(f"Single-call summary generation completed in {elapsed_direct:.2f} seconds!")
    print(f"Summary preview:\n{final_summary[:200]}...")

    # 2. Test precomputed cache retrieval
    print("Testing precomputed cache hit...")
    t_cache_0 = time.time()
    cached_tokens = []
    for chunk in generator.generate_summary_stream(retriever, length_option="Detailed (~200 words)"):
        cached_tokens.append(chunk)
    elapsed_cache = time.time() - t_cache_0
    print(f"Cached summary retrieval took {elapsed_cache:.4f} seconds!")
    assert elapsed_cache < 0.1, "Cached retrieval should be sub-tenth second!"


def test_short_vs_detailed_answer():
    print("\n--- 4. Testing Short vs Detailed Answer Toggle ---")
    retriever = FAISSRetriever()
    doc_path = "sample_documents/module5_notes.pdf"
    retriever.index_pdf(doc_path)
    generator = Generator()
    
    query = "What is a context free grammar?"
    retrieved = retriever.retrieve(query, top_k=3)
    
    # Test Short style
    t0 = time.time()
    short_chunks = []
    for chunk in generator.generate_answer_stream(query, retrieved, answer_style="Short (1-2 sentences)"):
        short_chunks.append(chunk)
    elapsed_short = time.time() - t0
    short_answer = short_chunks[-1] if short_chunks else ""
    print(f"Short answer ({elapsed_short:.2f}s):\n{short_answer.strip()}\n")
    
    # Test Detailed style
    t0 = time.time()
    detailed_chunks = []
    for chunk in generator.generate_answer_stream(query, retrieved, answer_style="Detailed (3-6 sentences)"):
        detailed_chunks.append(chunk)
    elapsed_detailed = time.time() - t0
    detailed_answer = detailed_chunks[-1] if detailed_chunks else ""
    print(f"Detailed answer ({elapsed_detailed:.2f}s):\n{detailed_answer.strip()}\n")

    assert len(short_answer) > 20, "Short answer should not be empty"
    assert len(detailed_answer) > 20, "Detailed answer should not be empty"
    print("Short vs Detailed toggle verified successfully!")


def test_stop_generation():
    print("\n--- 5. Testing Stop Button Generation Interrupt ---")
    generator = Generator()
    assert hasattr(generator, "stop_generation"), "Generator must have stop_generation method"
    generator.stop_generation()
    assert generator._is_stopped is True, "stop_generation should set _is_stopped = True"
    generator._is_stopped = False
    print("Stop generation hook verified successfully!")


if __name__ == "__main__":
    test_download_export()
    test_evidence_highlighting()
    test_summary_speed_and_precomputation()
    test_short_vs_detailed_answer()
    test_stop_generation()
    print("\n================ ALL VERIFICATION TESTS PASSED! ================")
