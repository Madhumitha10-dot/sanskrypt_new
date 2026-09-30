"""
SansKrypt Read-Only Evaluation Suite
Strictly executes existing SansKrypt modules in read-only mode, measuring:
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

# Ensure root workspace directory is in sys.path
BASE_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_PROJECT_DIR not in sys.path:
    sys.path.insert(0, BASE_PROJECT_DIR)

import glob
import time
import csv
import json
import statistics
from typing import Dict, List, Any
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

def get_unique_samples() -> List[Dict[str, Any]]:
    upload_orig = os.path.join("uploads", "original")
    files = sorted(glob.glob(os.path.join(upload_orig, "*")))
    files = [f for f in files if not f.endswith(".gitkeep")]
    
    unique_map = {}
    for fpath in files:
        fname = os.path.basename(fpath)
        base_name = fname.split("_", 1)[1] if "_" in fname else fname
        if base_name not in unique_map:
            unique_map[base_name] = fpath
            
    samples = []
    for base_name, fpath in sorted(unique_map.items()):
        with Image.open(fpath) as img:
            w, h = img.size
            fmt = img.format
            mode = img.mode
        
        # Determine handwritten / printed character based on sample name/image type
        is_hw = "handwritten" in base_name.lower() or "manu" in base_name.lower() or "whatsapp" in base_name.lower()
        samples.append({
            "id": base_name,
            "filepath": fpath,
            "width": w,
            "height": h,
            "pixels": w * h,
            "format": fmt,
            "mode": mode,
            "type": "Handwritten / Historical Manuscript" if is_hw else "Printed / Modern Typeset"
        })
    return samples

def run_ocr_evaluations(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    print("\n--- Running OCR Evaluations ---")
    configure_tesseract_path()
    ocr_records = []
    
    for s in samples:
        fpath = s["filepath"]
        sid = s["id"]
        print(f"Evaluating OCR on: {sid}...")
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
    print("\n--- Running Translation Evaluations ---")
    trans_records = []
    
    for s, ocr_rec in zip(samples, ocr_records):
        sid = s["id"]
        sanskrit_text = ocr_rec["ensemble_tesseract"]["text"]
        print(f"Evaluating Translation on: {sid}...")
        
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
                if res and "Error" not in res:
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
    print("\n--- Running Explanation, Keyword & Classification Evaluations ---")
    nlp_records = []
    
    for s, ocr_rec, trans_rec in zip(samples, ocr_records, trans_records):
        sid = s["id"]
        s_text = ocr_rec["ensemble_tesseract"]["text"]
        t_text = trans_rec["tiered_pipeline"]["text"]
        print(f"Evaluating NLP on: {sid}...")
        
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
    print("\n--- Running Full End-to-End Pipeline Timing Benchmark ---")
    pipeline_records = []
    
    for s in samples:
        fpath = s["filepath"]
        sid = s["id"]
        print(f"Benchmarking End-to-End on: {sid}...")
        
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
            "image_type": s["type"],
            "resolution": f"{s['width']}x{s['height']}",
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

def main():
    print("=== STARTING REPRODUCIBLE SANSKRYPT EVALUATION ===")
    samples = get_unique_samples()
    print(f"Loaded {len(samples)} unique manuscript sample types.")
    
    ocr_records = run_ocr_evaluations(samples)
    trans_records = run_translation_evaluations(samples, ocr_records)
    nlp_records = run_explanation_and_nlp_evaluations(samples, ocr_records, trans_records)
    pipeline_records = run_end_to_end_pipeline(samples)
    
    # Save CSVs
    # 1. OCR Results CSV
    ocr_csv_path = os.path.join(RESULTS_DIR, "ocr_results.csv")
    with open(ocr_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Sample_ID", "Image_Type", "Resolution", "Tesseract_Ensemble_Text", "Tesseract_Ensemble_Conf", "Tesseract_Ensemble_Time", "Gemini_Vision_Success", "Gemini_Vision_Conf", "Gemini_Vision_Time", "Adaptive_PSM6_Conf", "Adaptive_PSM6_Time", "Grayscale_PSM6_Conf", "Grayscale_PSM6_Time", "Otsu_PSM6_Conf", "Otsu_PSM6_Time", "Adaptive_PSM4_Conf", "Adaptive_PSM4_Time"])
        for s, o in zip(samples, ocr_records):
            writer.writerow([
                s["id"],
                s["type"],
                f"{s['width']}x{s['height']}",
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
                round(o["single_pass"].get("adaptive_binary_psm4", {}).get("time", 0.0), 4)
            ])
            
    # 2. Translation Results CSV
    trans_csv_path = os.path.join(RESULTS_DIR, "translation_results.csv")
    with open(trans_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Sample_ID", "Sanskrit_OCR_Text", "Tier_Used", "Final_Translation", "Final_Time", "Gemini_Success", "Gemini_Time", "DeepTranslator_Success", "DeepTranslator_Translation", "DeepTranslator_Time", "FLAN_T5_Success", "FLAN_T5_Translation", "FLAN_T5_Time"])
        for t in trans_records:
            writer.writerow([
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
            
    # 3. Explanation Results CSV
    exp_csv_path = os.path.join(RESULTS_DIR, "explanation_results.csv")
    with open(exp_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Sample_ID", "Generated_Explanation", "Processing_Time"])
        for n in nlp_records:
            writer.writerow([n["sample_id"], n["explanation"].replace("\n", " "), round(n["explanation_time"], 4)])
            
    # 4. Keyword Results CSV
    kw_csv_path = os.path.join(RESULTS_DIR, "keyword_results.csv")
    with open(kw_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Sample_ID", "Keywords", "Keyword_Count", "Processing_Time"])
        for n in nlp_records:
            writer.writerow([n["sample_id"], n["keywords"], n["keyword_count"], round(n["keyword_time"], 4)])
            
    # 5. Classification Results CSV
    class_csv_path = os.path.join(RESULTS_DIR, "classification_results.csv")
    with open(class_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Sample_ID", "Predicted_Domain", "Processing_Time"])
        for n in nlp_records:
            writer.writerow([n["sample_id"], n["domain"], round(n["classification_time"], 4)])
            
    # 6. Pipeline Results CSV
    pipe_csv_path = os.path.join(RESULTS_DIR, "pipeline_results.csv")
    with open(pipe_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Sample_ID", "Image_Type", "Resolution", "T_Preprocessing", "T_OCR", "T_Translation", "T_Explanation", "T_Keywords", "T_Classification", "Total_Time", "Slowest_Stage", "OCR_Confidence", "Domain", "Status", "OCR_Text", "Translation", "Explanation", "Keywords"])
        for p in pipeline_records:
            writer.writerow([
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

    # Save complete JSON raw data
    raw_json_path = os.path.join(RAW_DIR, "evaluation_data.json")
    with open(raw_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "samples": samples,
            "ocr_records": ocr_records,
            "trans_records": trans_records,
            "nlp_records": nlp_records,
            "pipeline_records": pipeline_records
        }, f, indent=2, ensure_ascii=False)

    print("\n=== EVALUATION RUN COMPLETED SUCCESSFULLY ===")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"Raw data saved to: {RAW_DIR}")

if __name__ == "__main__":
    main()
