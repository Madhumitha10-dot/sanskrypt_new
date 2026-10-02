"""
SansKrypt Read-Only Evaluation Suite (Dynamic Comprehensive Driver)
Strictly executes existing SansKrypt modules in read-only mode across all valid registered manuscript images:
1. Dataset & image specs
2. OCR single-pass (PSM 6, PSM 4, PSM 3) & preprocessing variants
3. OCR multi-pass & Gemini Vision attempts
4. Calibrated confidence distribution & formula
5. Translation tiers (Gemini, deep-translator, FLAN-T5)
6. Explanation generation
7. Keyword extraction
8. Domain classification
9. End-to-End pipeline stage timings
"""
import os
import sys
import socket

# Set safe stdout encoding and network socket timeout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
socket.setdefaulttimeout(15.0)

# Ensure root workspace directory is in sys.path
BASE_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_PROJECT_DIR not in sys.path:
    sys.path.insert(0, BASE_PROJECT_DIR)

import time
import csv
import json
import sqlite3
import hashlib
import statistics
from typing import Dict, List, Any, Tuple
import numpy as np
from PIL import Image

# Import existing application modules in read-only mode
from preprocessing import (
    read_image_safely,
    generate_preprocessing_variants,
    preprocess_image
)
from ocr import (
    run_single_ocr_pass,
    extract_text,
    extract_with_gemini_vision,
    calculate_calibrated_confidence,
    clean_devanagari_text,
    configure_tesseract_path
)
from translation import (
    translate_text,
    translate_with_gemini,
    clean_sanskrit_for_translation,
    HAS_DEEP_TRANSLATOR,
    HAS_GENAI,
    TranslationPipeline
)
if HAS_DEEP_TRANSLATOR:
    from deep_translator import GoogleTranslator
from explain_classify import (
    generate_explanation,
    generate_with_gemini,
    extract_keywords,
    classify_manuscript,
    CANONICAL_CATEGORIES
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

def get_all_samples() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    """
    Loads all registered manuscript image records from the SQLite database
    in strict read-only mode and verifies their physical files.
    Returns (valid_samples, skipped_records, total_db_count).
    """
    db_path = os.path.join(BASE_PROJECT_DIR, "sanskrypt.db")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id,
               title,
               original_image_path,
               processed_image_path,
               ocr_confidence,
               classification,
               status,
               upload_date
        FROM manuscripts
        ORDER BY id
    """)
    rows = cursor.fetchall()
    conn.close()

    total_db_count = len(rows)
    samples = []
    skipped_records = []

    for r in rows:
        db_id, title, fpath, ppath, conf, clf, status, udate = r
        
        # Check path format
        if not fpath:
            skipped_records.append({
                "db_id": db_id,
                "title": title or f"Manuscript #{db_id}",
                "filepath": str(fpath),
                "reason": "original_image_path is empty/null"
            })
            continue

        # Resolve path
        if not os.path.isabs(fpath):
            abs_fpath = os.path.join(BASE_PROJECT_DIR, fpath)
        else:
            abs_fpath = fpath

        # Verify physical file existence
        if not os.path.exists(abs_fpath):
            skipped_records.append({
                "db_id": db_id,
                "title": title or f"Manuscript #{db_id}",
                "filepath": abs_fpath,
                "reason": "File does not exist on filesystem"
            })
            continue

        # Verify file readability and properties via PIL and SHA256
        try:
            with open(abs_fpath, "rb") as fp:
                file_bytes = fp.read()
                sha256 = hashlib.sha256(file_bytes).hexdigest()
            file_sz = len(file_bytes)

            with Image.open(abs_fpath) as img:
                w, h = img.size
                fmt = img.format
                mode = img.mode
        except Exception as e:
            skipped_records.append({
                "db_id": db_id,
                "title": title or f"Manuscript #{db_id}",
                "filepath": abs_fpath,
                "reason": f"Unreadable image file: {str(e)}"
            })
            continue

        fname = os.path.basename(abs_fpath)
        base_name = fname.split("_", 1)[1] if "_" in fname else fname
        is_hw = "handwritten" in fname.lower() or "manu" in fname.lower() or "whatsapp" in fname.lower()

        samples.append({
            "db_id": db_id,
            "title": title or f"Manuscript #{db_id}",
            "id": fname,
            "base_name": base_name,
            "filepath": abs_fpath,
            "processed_path": ppath,
            "file_size": file_sz,
            "sha256": sha256,
            "width": w,
            "height": h,
            "pixels": w * h,
            "format": fmt,
            "mode": mode,
            "db_confidence": conf,
            "db_classification": clf,
            "db_status": status,
            "upload_date": str(udate),
            "type": "Handwritten / Historical Manuscript" if is_hw else "Printed / Modern Typeset"
        })

    return samples, skipped_records, total_db_count

def run_ocr_evaluations(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    n = len(samples)
    print(f"\n--- Running OCR Evaluations on {n} Samples ---", flush=True)
    configure_tesseract_path()
    ocr_records = []
    
    for idx, s in enumerate(samples, 1):
        fpath = s["filepath"]
        sid = s["id"]
        print(f"[{idx:02d}/{n}] Evaluating OCR on: {sid} (DB ID: {s['db_id']})...", flush=True)
        bgr = read_image_safely(fpath)
        variants = generate_preprocessing_variants(bgr)
        
        # 1. Gemini Vision attempt
        t0 = time.perf_counter()
        gemini_out = None
        gemini_conf = 0.0
        gemini_success = False
        gemini_err = "None"
        try:
            gem_res = extract_with_gemini_vision(bgr)
            t_gemini = time.perf_counter() - t0
            if gem_res:
                gemini_out, gemini_conf = gem_res
                gemini_success = True
            else:
                gemini_err = "Model returned None / Fallback triggered"
        except Exception as e:
            t_gemini = time.perf_counter() - t0
            gemini_err = str(e)
            
        # 2. Multi-Pass Tesseract Ensemble (SansKrypt default pipeline)
        t0 = time.perf_counter()
        ens_text, ens_conf = extract_text(bgr)
        t_ensemble = time.perf_counter() - t0
        
        # 3. Single-pass configurations (PSM 6, PSM 4, PSM 3) across variants
        single_pass_res = {}
        for var_name, var_img in [("adaptive_binary", variants["adaptive_binary"]),
                                  ("enhanced_grayscale", variants["enhanced_grayscale"]),
                                  ("otsu_binary", variants["otsu_binary"])]:
            for psm in [6, 4, 3]:
                t0 = time.perf_counter()
                try:
                    text, conf, words = run_single_ocr_pass(var_img, lang="san", psm=psm)
                    dt = time.perf_counter() - t0
                    single_pass_res[f"{var_name}_psm{psm}"] = {
                        "text": text,
                        "conf": conf,
                        "words": words,
                        "time": dt,
                        "success": len(text.strip()) > 0
                    }
                except Exception as e:
                    dt = time.perf_counter() - t0
                    single_pass_res[f"{var_name}_psm{psm}"] = {
                        "text": "",
                        "conf": 0.0,
                        "words": 0,
                        "time": dt,
                        "success": False,
                        "error": str(e)
                    }

        ocr_records.append({
            "sample_id": sid,
            "db_id": s["db_id"],
            "gemini_vision": {
                "success": gemini_success,
                "text": gemini_out or "",
                "conf": gemini_conf,
                "time": t_gemini,
                "error": gemini_err
            },
            "ensemble_tesseract": {
                "text": ens_text,
                "conf": ens_conf,
                "time": t_ensemble,
                "success": len(ens_text.strip()) > 0
            },
            "single_pass": single_pass_res
        })
    return ocr_records

def run_translation_evaluations(samples: List[Dict[str, Any]], ocr_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    n = len(samples)
    print(f"\n--- Running Translation Evaluations on {n} Samples ---", flush=True)
    trans_records = []
    
    for idx, (s, ocr_rec) in enumerate(zip(samples, ocr_records), 1):
        sid = s["id"]
        sanskrit_text = ocr_rec["ensemble_tesseract"]["text"]
        print(f"[{idx:02d}/{n}] Evaluating Translation on: {sid} (DB ID: {s['db_id']})...", flush=True)
        
        # 1. Tier 1: Gemini Translation
        t0 = time.perf_counter()
        gemini_trans = None
        gemini_success = False
        gemini_err = "None"
        try:
            gem_res = translate_with_gemini(sanskrit_text)
            t_gem = time.perf_counter() - t0
            if gem_res:
                gemini_trans = gem_res
                gemini_success = True
            else:
                gemini_err = "Gemini returned None"
        except Exception as e:
            t_gem = time.perf_counter() - t0
            gemini_err = str(e)

        # 2. Tier 2: deep-translator (Google Translate)
        t0 = time.perf_counter()
        dt_trans = None
        dt_success = False
        dt_err = "None"
        if HAS_DEEP_TRANSLATOR and sanskrit_text.strip():
            try:
                clean_s = clean_sanskrit_for_translation(sanskrit_text)
                translator = GoogleTranslator(source="sa", target="en")
                res = translator.translate(clean_s)
                t_dt = time.perf_counter() - t0
                if res and "Error" not in res and "Server Error" not in res:
                    dt_trans = res.strip()
                    dt_success = True
                else:
                    dt_err = f"Response returned error string: {res}"
            except Exception as e:
                t_dt = time.perf_counter() - t0
                dt_err = str(e)
        else:
            t_dt = 0.0
            dt_err = "deep-translator not available or empty text"

        # 3. Tier 3: FLAN-T5 Local Transformer
        t0 = time.perf_counter()
        flan_trans = None
        flan_success = False
        flan_err = "None"
        try:
            pipeline = TranslationPipeline.get_instance()
            res = pipeline.translate_with_transformer(sanskrit_text)
            t_flan = time.perf_counter() - t0
            if res:
                flan_trans = res.strip()
                flan_success = True
        except Exception as e:
            t_flan = time.perf_counter() - t0
            flan_err = str(e)

        # 4. End-to-End Tiered Translation Pipeline
        t0 = time.perf_counter()
        final_trans = translate_text(sanskrit_text)
        t_final = time.perf_counter() - t0
        
        # Determine which tier was activated
        if gemini_success and final_trans == gemini_trans:
            tier_used = "Tier 1: Google Gemini"
        elif dt_success and final_trans == dt_trans:
            tier_used = "Tier 2: deep-translator (Google Translate)"
        elif flan_success and final_trans == flan_trans:
            tier_used = "Tier 3: FLAN-T5 Seq2Seq"
        else:
            tier_used = "Rule-based synthesis / Fallback"

        trans_records.append({
            "sample_id": sid,
            "db_id": s["db_id"],
            "sanskrit_text": sanskrit_text,
            "gemini": {
                "success": gemini_success,
                "text": gemini_trans or "",
                "time": t_gem,
                "error": gemini_err
            },
            "deep_translator": {
                "success": dt_success,
                "text": dt_trans or "",
                "time": t_dt,
                "error": dt_err
            },
            "flan_t5": {
                "success": flan_success,
                "text": flan_trans or "",
                "time": t_flan,
                "error": flan_err
            },
            "tiered_pipeline": {
                "text": final_trans,
                "time": t_final,
                "tier_used": tier_used,
                "success": bool(final_trans and len(final_trans.strip()) > 0)
            }
        })
    return trans_records

def run_explanation_and_nlp_evaluations(
    samples: List[Dict[str, Any]],
    ocr_records: List[Dict[str, Any]],
    trans_records: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    n = len(samples)
    print(f"\n--- Running Explanation, Keyword & Classification Evaluations on {n} Samples ---", flush=True)
    nlp_records = []
    
    for idx, (s, ocr_rec, trans_rec) in enumerate(zip(samples, ocr_records, trans_records), 1):
        sid = s["id"]
        s_text = ocr_rec["ensemble_tesseract"]["text"]
        t_text = trans_rec["tiered_pipeline"]["text"]
        print(f"[{idx:02d}/{n}] Evaluating NLP on: {sid} (DB ID: {s['db_id']})...", flush=True)
        
        # 1. Explanation
        t0 = time.perf_counter()
        exp = generate_explanation(s_text, t_text)
        t_exp = time.perf_counter() - t0
        
        # 2. Keywords
        t0 = time.perf_counter()
        kw_str = extract_keywords(s_text, t_text)
        t_kw = time.perf_counter() - t0
        kws = [k.strip() for k in kw_str.split(",") if k.strip()]
        
        # 3. Domain Classification
        t0 = time.perf_counter()
        domain = classify_manuscript(s_text, t_text)
        t_class = time.perf_counter() - t0
        
        nlp_records.append({
            "sample_id": sid,
            "db_id": s["db_id"],
            "explanation": exp,
            "explanation_time": t_exp,
            "keywords": kw_str,
            "keyword_list": kws,
            "keyword_count": len(kws),
            "keyword_time": t_kw,
            "domain": domain,
            "classification_time": t_class
        })
    return nlp_records

def run_end_to_end_pipeline(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    n = len(samples)
    print(f"\n--- Running Full End-to-End Pipeline Timing Benchmark on {n} Samples ---", flush=True)
    pipeline_records = []
    
    for idx, s in enumerate(samples, 1):
        fpath = s["filepath"]
        sid = s["id"]
        print(f"[{idx:02d}/{n}] Benchmarking End-to-End on: {sid} (DB ID: {s['db_id']})...", flush=True)
        
        # Stage 1: Preprocessing
        t0 = time.perf_counter()
        bgr = read_image_safely(fpath)
        variants = generate_preprocessing_variants(bgr)
        processed_img = variants["adaptive_binary"]
        t_pre = time.perf_counter() - t0
        
        # Stage 2: OCR & Confidence
        t0 = time.perf_counter()
        ocr_text, ocr_conf = extract_text(bgr)
        t_ocr = time.perf_counter() - t0
        
        # Stage 3: Translation
        t0 = time.perf_counter()
        trans_text = translate_text(ocr_text)
        t_trans = time.perf_counter() - t0
        
        # Stage 4: Explanation
        t0 = time.perf_counter()
        exp_text = generate_explanation(ocr_text, trans_text)
        t_exp = time.perf_counter() - t0
        
        # Stage 5: Keywords
        t0 = time.perf_counter()
        kw_text = extract_keywords(ocr_text, trans_text)
        t_kw = time.perf_counter() - t0
        
        # Stage 6: Classification
        t0 = time.perf_counter()
        domain = classify_manuscript(ocr_text, trans_text)
        t_class = time.perf_counter() - t0
        
        total_time = t_pre + t_ocr + t_trans + t_exp + t_kw + t_class
        stage_times = {
            "Preprocessing": t_pre,
            "OCR": t_ocr,
            "Translation": t_trans,
            "Explanation": t_exp,
            "Keywords": t_kw,
            "Classification": t_class
        }
        slowest_stage = max(stage_times, key=stage_times.get)
        
        status = "completed" if ocr_conf >= 40.0 else "low_confidence"
        if not ocr_text.strip():
            status = "failed"
            
        pipeline_records.append({
            "sample_id": sid,
            "db_id": s["db_id"],
            "image_type": s["type"],
            "resolution": f"{s['width']}x{s['height']}",
            "file_size": s["file_size"],
            "sha256": s["sha256"],
            "t_preprocessing": t_pre,
            "t_ocr": t_ocr,
            "t_translation": t_trans,
            "t_explanation": t_exp,
            "t_keywords": t_kw,
            "t_classification": t_class,
            "total_time": total_time,
            "slowest_stage": slowest_stage,
            "ocr_confidence": ocr_conf,
            "domain": domain,
            "status": status,
            "ocr_text": ocr_text,
            "translation": trans_text,
            "explanation": exp_text,
            "keywords": kw_text
        })
    return pipeline_records

def generate_summary_text(
    samples: List[Dict[str, Any]],
    skipped_records: List[Dict[str, Any]],
    total_db_count: int,
    ocr_records: List[Dict[str, Any]],
    trans_records: List[Dict[str, Any]],
    nlp_records: List[Dict[str, Any]],
    pipeline_records: List[Dict[str, Any]]
) -> str:
    evaluated_ids = [s["db_id"] for s in samples]
    skipped_ids = [s["db_id"] for s in skipped_records]
    widths = [s["width"] for s in samples]
    heights = [s["height"] for s in samples]
    pixels = [s["pixels"] for s in samples]
    sizes = [s["file_size"] for s in samples]
    formats = set(s["format"] for s in samples)
    fmt_counts = {fmt: sum(1 for s in samples if s["format"] == fmt) for fmt in formats}

    ens_confs = [o["ensemble_tesseract"]["conf"] for o in ocr_records]
    ens_times = [o["ensemble_tesseract"]["time"] for o in ocr_records]
    ens_success = [o["ensemble_tesseract"]["success"] for o in ocr_records]

    dt_succ = [t["deep_translator"]["success"] for t in trans_records]
    dt_times = [t["deep_translator"]["time"] for t in trans_records]
    flan_succ = [t["flan_t5"]["success"] for t in trans_records]
    pipe_trans_succ = [t["tiered_pipeline"]["success"] for t in trans_records]
    pipe_trans_times = [t["tiered_pipeline"]["time"] for t in trans_records]
    tiers_used = [t["tiered_pipeline"]["tier_used"] for t in trans_records]
    tier_dist = {tier: tiers_used.count(tier) for tier in set(tiers_used)}

    exp_times = [n["explanation_time"] for n in nlp_records]
    kw_counts = [n["keyword_count"] for n in nlp_records]
    kw_times = [n["keyword_time"] for n in nlp_records]
    domains = [n["domain"] for n in nlp_records]
    domain_dist = {cat: domains.count(cat) for cat in CANONICAL_CATEGORIES}

    t_pre = [p["t_preprocessing"] for p in pipeline_records]
    t_ocr = [p["t_ocr"] for p in pipeline_records]
    t_trans = [p["t_translation"] for p in pipeline_records]
    t_exp = [p["t_explanation"] for p in pipeline_records]
    t_kw = [p["t_keywords"] for p in pipeline_records]
    t_class = [p["t_classification"] for p in pipeline_records]
    t_tot = [p["total_time"] for p in pipeline_records]
    slowest_counts = {st: sum(1 for p in pipeline_records if p["slowest_stage"] == st) for st in ["Preprocessing", "OCR", "Translation", "Explanation", "Keywords", "Classification"]}

    lines = []
    lines.append("============================================================")
    lines.append("SANSKRYPT EVALUATION SUMMARY")
    lines.append("============================================================")
    lines.append(f"Database records: {total_db_count}")
    lines.append(f"Valid image records: {len(samples)}")
    lines.append(f"Actually evaluated: {len(samples)}")
    lines.append(f"Skipped: {len(skipped_records)}")
    lines.append(f"Evaluated IDs: {evaluated_ids}")
    lines.append(f"Skipped IDs: {skipped_ids}")
    lines.append("")
    lines.append("DATASET SPECIFICATIONS")
    lines.append(f"- Total Evaluated: {len(samples)}")
    lines.append(f"- Format Breakdown: {fmt_counts}")
    lines.append(f"- File Size: min={min(sizes):,} B, max={max(sizes):,} B, mean={statistics.mean(sizes):,.1f} B")
    lines.append(f"- Resolution Range: {min(widths)}x{min(heights)} to {max(widths)}x{max(heights)}")
    lines.append("")
    lines.append("OCR EVALUATION")
    lines.append(f"- Multi-pass Ensemble Success: {sum(ens_success)}/{len(ocr_records)} ({sum(ens_success)/len(ocr_records)*100:.2f}%)")
    lines.append(f"- Multi-pass Ensemble Failures: {len(ocr_records) - sum(ens_success)}")
    lines.append(f"- Confidence Distribution: min={min(ens_confs):.2f}%, max={max(ens_confs):.2f}%, mean={statistics.mean(ens_confs):.2f}%, median={statistics.median(ens_confs):.2f}%, stdev={statistics.stdev(ens_confs):.2f}%")
    lines.append(f"- Confidence < 40% (low_confidence): {sum(1 for c in ens_confs if c < 40.0)}")
    lines.append(f"- Confidence >= 40% (adequate/high): {sum(1 for c in ens_confs if c >= 40.0)}")
    lines.append(f"- CER/WER: NOT AVAILABLE (no ground-truth transcription dataset provided)")
    lines.append("")
    lines.append("TRANSLATION EVALUATION")
    lines.append(f"- Deep-translator Success: {sum(dt_succ)}/{len(trans_records)} ({sum(dt_succ)/len(trans_records)*100:.2f}%)")
    lines.append(f"- FLAN-T5 Success: {sum(flan_succ)}/{len(trans_records)}")
    lines.append(f"- Tiered Pipeline Total Completed: {sum(pipe_trans_succ)}/{len(trans_records)} ({sum(pipe_trans_succ)/len(trans_records)*100:.2f}%)")
    lines.append(f"- Tier Activations: {tier_dist}")
    lines.append(f"- Translation Latency (s): min={min(pipe_trans_times):.4f}s, max={max(pipe_trans_times):.4f}s, mean={statistics.mean(pipe_trans_times):.4f}s, median={statistics.median(pipe_trans_times):.4f}s")
    lines.append(f"- BLEU / chrF / Reference Metrics: NOT AVAILABLE (no ground-truth reference translations provided)")
    lines.append("")
    lines.append("EXPLANATION & KEYWORDS EVALUATION")
    lines.append(f"- Explanations Completed: {len(nlp_records)}/{len(nlp_records)} (100.0%), mean time={statistics.mean(exp_times):.4f}s")
    lines.append(f"- Keywords Completed: {len(nlp_records)}/{len(nlp_records)} (100.0%), mean count={statistics.mean(kw_counts):.2f}, mean time={statistics.mean(kw_times):.4f}s")
    lines.append("")
    lines.append("DOMAIN CLASSIFICATION EVALUATION")
    for cat, cnt in domain_dist.items():
        lines.append(f"- {cat:20s}: {cnt:2d} samples ({(cnt/len(domains))*100:5.2f}%)")
    lines.append(f"- Classification Accuracy / Precision / Recall: NOT AVAILABLE (no ground-truth domain labels provided)")
    lines.append("")
    lines.append("END-TO-END PIPELINE LATENCY")
    lines.append(f"- Total Latency (s): min={min(t_tot):.4f}s, max={max(t_tot):.4f}s, mean={statistics.mean(t_tot):.4f}s, median={statistics.median(t_tot):.4f}s, stdev={statistics.stdev(t_tot):.4f}s")
    lines.append(f"- Preprocessing mean: {statistics.mean(t_pre):.4f}s ({statistics.mean(t_pre)/statistics.mean(t_tot)*100:.2f}%)")
    lines.append(f"- OCR Engine mean:    {statistics.mean(t_ocr):.4f}s ({statistics.mean(t_ocr)/statistics.mean(t_tot)*100:.2f}%)")
    lines.append(f"- Translation mean:   {statistics.mean(t_trans):.4f}s ({statistics.mean(t_trans)/statistics.mean(t_tot)*100:.2f}%)")
    lines.append(f"- Explanation mean:   {statistics.mean(t_exp):.4f}s ({statistics.mean(t_exp)/statistics.mean(t_tot)*100:.2f}%)")
    lines.append(f"- Keyword mean:       {statistics.mean(t_kw):.4f}s ({statistics.mean(t_kw)/statistics.mean(t_tot)*100:.2f}%)")
    lines.append(f"- Classification mean:{statistics.mean(t_class):.4f}s ({statistics.mean(t_class)/statistics.mean(t_tot)*100:.2f}%)")
    lines.append(f"- Slowest Stage Distribution: {slowest_counts}")
    lines.append("============================================================")
    return "\n".join(lines)

def main():
    print("=== STARTING REPRODUCIBLE SANSKRYPT EVALUATION ===", flush=True)
    samples, skipped_records, total_db_count = get_all_samples()
    
    # Required initial report
    print(f"Database records: {total_db_count}", flush=True)
    print(f"Records with valid image files: {len(samples)}", flush=True)
    print(f"Records with missing/unreadable images: {len(skipped_records)}", flush=True)
    print(f"Evaluated IDs: {[s['db_id'] for s in samples]}", flush=True)
    if skipped_records:
        print(f"Skipped IDs: {[s['db_id'] for s in skipped_records]}", flush=True)
        for sk in skipped_records:
            print(f"  Skipped DB ID {sk['db_id']} ({sk['title']}): {sk['filepath']} -> {sk['reason']}", flush=True)
    else:
        print("Skipped IDs: []", flush=True)
    
    ocr_records = run_ocr_evaluations(samples)
    trans_records = run_translation_evaluations(samples, ocr_records)
    nlp_records = run_explanation_and_nlp_evaluations(samples, ocr_records, trans_records)
    pipeline_records = run_end_to_end_pipeline(samples)
    
    # Save CSVs
    # 1. Dataset Results CSV
    dataset_csv_path = os.path.join(RESULTS_DIR, "dataset_results.csv")
    with open(dataset_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Sample_Index", "DB_ID", "Title", "Filename", "Image_Type", "Width", "Height", "Pixels", "Format", "Mode", "File_Size_Bytes", "SHA256", "DB_Status", "DB_Classification", "DB_Confidence"])
        for idx, s in enumerate(samples, 1):
            writer.writerow([
                idx,
                s["db_id"],
                s["title"],
                s["id"],
                s["type"],
                s["width"],
                s["height"],
                s["pixels"],
                s["format"],
                s["mode"],
                s["file_size"],
                s["sha256"],
                s["db_status"],
                s["db_classification"],
                s["db_confidence"]
            ])

    # 2. OCR Results CSV
    ocr_csv_path = os.path.join(RESULTS_DIR, "ocr_results.csv")
    with open(ocr_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Sample_Index", "DB_ID", "Filename", "Image_Type", "Resolution", "File_Size_Bytes", "SHA256_Prefix",
            "Tesseract_Ensemble_Text", "Tesseract_Ensemble_Conf", "Tesseract_Ensemble_Time",
            "Gemini_Vision_Success", "Gemini_Vision_Conf", "Gemini_Vision_Time",
            "Adaptive_PSM6_Conf", "Adaptive_PSM6_Time",
            "Grayscale_PSM6_Conf", "Grayscale_PSM6_Time",
            "Otsu_PSM6_Conf", "Otsu_PSM6_Time",
            "Adaptive_PSM4_Conf", "Adaptive_PSM4_Time",
            "Adaptive_PSM3_Conf", "Adaptive_PSM3_Time"
        ])
        for idx, (s, o) in enumerate(zip(samples, ocr_records), 1):
            writer.writerow([
                idx,
                s["db_id"],
                s["id"],
                s["type"],
                f"{s['width']}x{s['height']}",
                s["file_size"],
                s["sha256"][:12],
                o["ensemble_tesseract"]["text"].replace("\n", " "),
                o["ensemble_tesseract"]["conf"],
                round(o["ensemble_tesseract"]["time"], 4),
                o["gemini_vision"]["success"],
                o["gemini_vision"]["conf"],
                round(o["gemini_vision"]["time"], 4),
                o["single_pass"].get("adaptive_binary_psm6", {}).get("conf", 0.0),
                round(o["single_pass"].get("adaptive_binary_psm6", {}).get("time", 0.0), 4),
                o["single_pass"].get("enhanced_grayscale_psm6", {}).get("conf", 0.0),
                round(o["single_pass"].get("enhanced_grayscale_psm6", {}).get("time", 0.0), 4),
                o["single_pass"].get("otsu_binary_psm6", {}).get("conf", 0.0),
                round(o["single_pass"].get("otsu_binary_psm6", {}).get("time", 0.0), 4),
                o["single_pass"].get("adaptive_binary_psm4", {}).get("conf", 0.0),
                round(o["single_pass"].get("adaptive_binary_psm4", {}).get("time", 0.0), 4),
                o["single_pass"].get("adaptive_binary_psm3", {}).get("conf", 0.0),
                round(o["single_pass"].get("adaptive_binary_psm3", {}).get("time", 0.0), 4)
            ])
            
    # 3. Translation Results CSV
    trans_csv_path = os.path.join(RESULTS_DIR, "translation_results.csv")
    with open(trans_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Sample_Index", "DB_ID", "Filename", "Sanskrit_OCR_Text", "Tier_Used",
            "Final_Translation", "Final_Time",
            "Gemini_Success", "Gemini_Time",
            "DeepTranslator_Success", "DeepTranslator_Translation", "DeepTranslator_Time",
            "FLAN_T5_Success", "FLAN_T5_Translation", "FLAN_T5_Time"
        ])
        for idx, t in enumerate(trans_records, 1):
            writer.writerow([
                idx,
                t["db_id"],
                t["sample_id"],
                t["sanskrit_text"].replace("\n", " "),
                t["tiered_pipeline"]["tier_used"],
                t["tiered_pipeline"]["text"].replace("\n", " "),
                round(t["tiered_pipeline"]["time"], 4),
                t["gemini"]["success"],
                round(t["gemini"]["time"], 4),
                t["deep_translator"]["success"],
                t["deep_translator"]["text"].replace("\n", " "),
                round(t["deep_translator"]["time"], 4),
                t["flan_t5"]["success"],
                t["flan_t5"]["text"].replace("\n", " "),
                round(t["flan_t5"]["time"], 4)
            ])
            
    # 4. Explanation Results CSV
    exp_csv_path = os.path.join(RESULTS_DIR, "explanation_results.csv")
    with open(exp_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Sample_Index", "DB_ID", "Filename", "Generated_Explanation", "Processing_Time"])
        for idx, n in enumerate(nlp_records, 1):
            writer.writerow([idx, n["db_id"], n["sample_id"], n["explanation"].replace("\n", " "), round(n["explanation_time"], 4)])
            
    # 5. Keyword Results CSV
    kw_csv_path = os.path.join(RESULTS_DIR, "keyword_results.csv")
    with open(kw_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Sample_Index", "DB_ID", "Filename", "Keywords", "Keyword_Count", "Processing_Time"])
        for idx, n in enumerate(nlp_records, 1):
            writer.writerow([idx, n["db_id"], n["sample_id"], n["keywords"], n["keyword_count"], round(n["keyword_time"], 4)])
            
    # 6. Classification Results CSV
    class_csv_path = os.path.join(RESULTS_DIR, "classification_results.csv")
    with open(class_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Sample_Index", "DB_ID", "Filename", "Predicted_Domain", "Processing_Time"])
        for idx, n in enumerate(nlp_records, 1):
            writer.writerow([idx, n["db_id"], n["sample_id"], n["domain"], round(n["classification_time"], 4)])
            
    # 7. Pipeline Results CSV
    pipe_csv_path = os.path.join(RESULTS_DIR, "pipeline_results.csv")
    with open(pipe_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Sample_Index", "DB_ID", "Filename", "Image_Type", "Resolution",
            "T_Preprocessing", "T_OCR", "T_Translation", "T_Explanation", "T_Keywords", "T_Classification",
            "Total_Time", "Slowest_Stage", "OCR_Confidence", "Domain", "Status",
            "OCR_Text", "Translation", "Explanation", "Keywords"
        ])
        for idx, p in enumerate(pipeline_records, 1):
            writer.writerow([
                idx,
                p["db_id"],
                p["sample_id"],
                p["image_type"],
                p["resolution"],
                round(p["t_preprocessing"], 4),
                round(p["t_ocr"], 4),
                round(p["t_translation"], 4),
                round(p["t_explanation"], 4),
                round(p["t_keywords"], 4),
                round(p["t_classification"], 4),
                round(p["total_time"], 4),
                p["slowest_stage"],
                p["ocr_confidence"],
                p["domain"],
                p["status"],
                p["ocr_text"].replace("\n", " "),
                p["translation"].replace("\n", " "),
                p["explanation"].replace("\n", " "),
                p["keywords"]
            ])

    # 8. Save complete JSON raw data
    raw_json_path = os.path.join(RAW_DIR, "evaluation_data.json")
    with open(raw_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_db_records": total_db_count,
            "valid_samples_count": len(samples),
            "skipped_records_count": len(skipped_records),
            "skipped_records": skipped_records,
            "samples": samples,
            "ocr_records": ocr_records,
            "trans_records": trans_records,
            "nlp_records": nlp_records,
            "pipeline_records": pipeline_records
        }, f, indent=2, ensure_ascii=False)

    # 9. Save Evaluation Summary TXT
    summary_text = generate_summary_text(
        samples, skipped_records, total_db_count,
        ocr_records, trans_records, nlp_records, pipeline_records
    )
    summary_path = os.path.join(RESULTS_DIR, "evaluation_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print("\n=== EVALUATION RUN COMPLETED SUCCESSFULLY ===", flush=True)
    print(f"Results saved to: {RESULTS_DIR}", flush=True)
    print(f"Raw data saved to: {RAW_DIR}", flush=True)
    print(f"Summary saved to: {summary_path}", flush=True)

if __name__ == "__main__":
    main()
