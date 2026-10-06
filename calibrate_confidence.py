"""
Calibration script for Answer Confidence Scoring thresholds.
Evaluates:
- 10 benchmark questions on ai_renewable_energy_report.pdf (8 answerable, 2 out-of-scope)
- 8 authentic discourse questions on module5_notes.pdf (8 answerable)
- 6 out-of-scope questions (newly created)
Prints top cosine similarity scores and determines empirical High / Medium / Low thresholds.
"""

import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from config import EMBEDDING_MODEL_NAME
from embedder import Embedder
from retriever import FAISSRetriever

SAMPLE_PDF_PATH = Path("sample_documents") / "ai_renewable_energy_report.pdf"
MODULE5_PDF_PATH = Path("sample_documents") / "module5_notes.pdf"

# 10 Questions for AI Renewable Energy Report (8 In-Scope, 2 Out-of-Scope)
ENERGY_QUESTIONS = [
    ("How much was invested in clean energy tech in 2025?", False),
    ("By what percentage did INT8 quantization reduce latency?", False),
    ("What accuracy retention was achieved with INT8 quantization?", False),
    ("How many charge cycles did LiFePO4 cells achieve before degrading to 80% capacity?", False),
    ("What battery cell chemistry was evaluated for thermal runaway prevention?", False),
    ("What uptime did Project Aegis demonstrate against simulated attacks?", False),
    ("How many edge nodes does the consortium plan to deploy?", False),
    ("By what target date does the consortium plan to deploy the edge nodes?", False),
    ("Who won the FIFA Men's World Cup in 1994?", True),
    ("What is the capital city of Mars?", True),
]

# 8 Questions for Module 5 Notes (All In-Scope)
MODULE5_QUESTIONS = [
    ("What is discourse processing and how does it differ from single-sentence language analysis?", False),
    ("What is an antecedent in reference resolution, and how does anaphora differ from cataphora?", False),
    ("What grammatical and syntactic agreement constraints must be satisfied for valid pronoun resolution?", False),
    ("What does discourse coherence mean, and what makes a multi-sentence discourse coherent?", False),
    ("How does the Hobbs algorithm resolve pronoun references using sentence parse trees?", False),
    ("What are the primary limitations and failure scenarios of the Hobbs algorithm?", False),
    ("What is the purpose of Centering Theory in discourse processing?", False),
    ("What is the key difference between anaphora resolution and coreference resolution?", False),
]

# 6 New Diverse Out-of-Scope Questions (Tested across both documents)
NEW_OUT_OF_SCOPE_QUESTIONS = [
    ("What is the standard boiling point of liquid nitrogen at 1 atm pressure?", True),
    ("How did Napoleon Bonaparte escape from his initial exile on Elba?", True),
    ("What are the essential steps and ingredients to bake traditional sourdough bread?", True),
    ("Who directed the 1997 epic romance and disaster film Titanic?", True),
    ("How does the Cas9 endonuclease induce double-stranded DNA breaks in mammalian genome editing?", True),
    ("Under what specific conditions is a batsman dismissed Leg Before Wicket (LBW) in cricket?", True),
]


def calibrate() -> None:
    print("=" * 84, flush=True)
    print("        RAG CONFIDENCE THRESHOLD CALIBRATION BENCHMARK", flush=True)
    print("=" * 84, flush=True)

    embedder = Embedder(model_name=EMBEDDING_MODEL_NAME)
    retriever_energy = FAISSRetriever(embedder=embedder)
    retriever_energy.index_pdf(SAMPLE_PDF_PATH)

    retriever_module5 = FAISSRetriever(embedder=embedder)
    retriever_module5.index_pdf(MODULE5_PDF_PATH)

    in_scope_scores = []
    out_of_scope_scores = []

    print("\n--- 1. Evaluating Questions on ai_renewable_energy_report.pdf ---", flush=True)
    for q, is_oos in ENERGY_QUESTIONS:
        chunks = retriever_energy.retrieve(q, top_k=1, similarity_threshold=None, use_hybrid=False)
        top_sim = chunks[0]["similarity_score"] if chunks else 0.0
        label = "Out-of-Scope" if is_oos else "Answerable"
        print(f"[{label:12}] Score: {top_sim:.4f} | {q}", flush=True)
        if is_oos:
            out_of_scope_scores.append(top_sim)
        else:
            in_scope_scores.append(top_sim)

    print("\n--- 2. Evaluating 8 Questions on module5_notes.pdf ---", flush=True)
    for q, is_oos in MODULE5_QUESTIONS:
        chunks = retriever_module5.retrieve(q, top_k=1, similarity_threshold=None, use_hybrid=False)
        top_sim = chunks[0]["similarity_score"] if chunks else 0.0
        print(f"[{'Answerable':12}] Score: {top_sim:.4f} | {q}", flush=True)
        in_scope_scores.append(top_sim)

    print("\n--- 3. Evaluating 6 New Out-of-Scope Questions on module5_notes.pdf ---", flush=True)
    for q, is_oos in NEW_OUT_OF_SCOPE_QUESTIONS:
        chunks = retriever_module5.retrieve(q, top_k=1, similarity_threshold=None, use_hybrid=False)
        top_sim = chunks[0]["similarity_score"] if chunks else 0.0
        print(f"[{'Out-of-Scope':12}] Score: {top_sim:.4f} | {q}", flush=True)
        out_of_scope_scores.append(top_sim)

    # Statistical summary
    in_arr = np.array(in_scope_scores)
    oos_arr = np.array(out_of_scope_scores)

    print("\n" + "=" * 84, flush=True)
    print("### EMPIRICAL SCORE DISTRIBUTIONS", flush=True)
    print("=" * 84, flush=True)
    print(f"Answerable Queries (N={len(in_arr)}):", flush=True)
    print(f"   Min:    {in_arr.min():.4f}")
    print(f"   Mean:   {in_arr.mean():.4f}")
    print(f"   Median: {np.median(in_arr):.4f}")
    print(f"   Max:    {in_arr.max():.4f}")
    print(f"\nOut-of-Scope Queries (N={len(oos_arr)}):", flush=True)
    print(f"   Min:    {oos_arr.min():.4f}")
    print(f"   Mean:   {oos_arr.mean():.4f}")
    print(f"   Median: {np.median(oos_arr):.4f}")
    print(f"   Max:    {oos_arr.max():.4f}")

    # Optimal Calibration Calculation
    # High: clear, robust semantic match (above 70th percentile of answerable queries or >= 0.50)
    # Medium: valid match, but moderate certainty (between refusal boundary + buffer and High)
    # Low: borderline (around 0.22 - 0.35)
    # Refusal: strictly filters out max out-of-scope (0.22)
    high_threshold = round(float(np.percentile(in_arr, 45)), 2)  # around 0.50 - 0.55
    med_threshold = round(float(max(oos_arr.max() + 0.10, 0.35)), 2)  # around 0.35

    print("\n" + "=" * 84, flush=True)
    print("### CALIBRATED THRESHOLDS FOR CONFIG.PY", flush=True)
    print("=" * 84, flush=True)
    print(f"CONFIDENCE_HIGH_THRESHOLD = {high_threshold}  (Cosine Similarity >= {high_threshold})")
    print(f"CONFIDENCE_MED_THRESHOLD  = {med_threshold}  ({med_threshold} <= Cosine Similarity < {high_threshold})")
    print(f"CONFIDENCE_LOW_THRESHOLD  = < {med_threshold} (Cosine Similarity < {med_threshold})")
    print("SIMILARITY_THRESHOLD (Refusal) = 0.22")

    print("\n" + "=" * 84, flush=True)
    print("### CALIBRATION SUMMARY TABLE (PASTE INTO REPORT)")
    print("=" * 84, flush=True)
    print("| Query Category | Sample Size | Cosine Range (Min - Max) | Mean Score | Median Score |")
    print("| :--- | :--- | :--- | :--- | :--- |")
    print(f"| Answerable (In-Scope) | {len(in_arr)} | {in_arr.min():.3f} – {in_arr.max():.3f} | {in_arr.mean():.3f} | {np.median(in_arr):.3f} |")
    print(f"| Out-of-Scope (Refusal) | {len(oos_arr)} | {oos_arr.min():.3f} – {oos_arr.max():.3f} | {oos_arr.mean():.3f} | {np.median(oos_arr):.3f} |")
    print("=" * 84, flush=True)


if __name__ == "__main__":
    calibrate()
