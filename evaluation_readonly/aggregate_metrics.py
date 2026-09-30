"""
Aggregates all exact numbers and statistical metrics from raw evaluation CSVs.
"""
import os
import csv
import json
import statistics
from typing import Dict, List, Any

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")

CANONICAL_CATEGORIES = [
    "Philosophy",
    "Ayurveda",
    "Astronomy",
    "Literature",
    "Mathematics",
    "Vedic Sciences",
    "Grammar",
    "Yoga & Spirituality",
    "General/Other"
]


def load_json_raw():
    with open(os.path.join(RAW_DIR, "evaluation_data.json"), "r", encoding="utf-8") as f:
        return json.load(f)

def run_stats():
    data = load_json_raw()
    samples = data["samples"]
    ocr = data["ocr_records"]
    trans = data["trans_records"]
    nlp = data["nlp_records"]
    pipe = data["pipeline_records"]

    print("================ DATASET SPECS ================")
    print(f"Total Unique Samples: {len(samples)}")
    widths = [s["width"] for s in samples]
    heights = [s["height"] for s in samples]
    pixels = [s["pixels"] for s in samples]
    formats = set(s["format"] for s in samples)
    print(f"Formats: {formats}")
    print(f"Resolution min: {min(widths)}x{min(heights)} ({min(pixels)} px)")
    print(f"Resolution max: {max(widths)}x{max(heights)} ({max(pixels)} px)")
    print(f"Resolution range: {min(widths)}x{min(heights)} to {max(widths)}x{max(heights)}")

    print("\n================ OCR STATISTICS ================")
    # 1. Multi-pass Tesseract Ensemble
    ens_confs = [o["ensemble_tesseract"]["conf"] for o in ocr]
    ens_times = [o["ensemble_tesseract"]["time"] for o in ocr]
    ens_success = [o["ensemble_tesseract"]["success"] for o in ocr]
    print(f"Tesseract Ensemble Multi-pass:")
    print(f"  Samples: {len(ocr)}, Success: {sum(ens_success)}, Failed: {len(ocr) - sum(ens_success)}")
    print(f"  Confidence: min={min(ens_confs):.2f}%, max={max(ens_confs):.2f}%, mean={statistics.mean(ens_confs):.2f}%, median={statistics.median(ens_confs):.2f}%, stdev={statistics.stdev(ens_confs):.2f}%")
    print(f"  Time (s): min={min(ens_times):.4f}s, max={max(ens_times):.4f}s, mean={statistics.mean(ens_times):.4f}s, median={statistics.median(ens_times):.4f}s")
    
    below_40 = sum(1 for c in ens_confs if c < 40.0)
    above_40 = sum(1 for c in ens_confs if c >= 40.0)
    print(f"  Confidence < 40% (low_confidence): {below_40} ({below_40/len(ens_confs)*100:.2f}%)")
    print(f"  Confidence >= 40% (adequate/high): {above_40} ({above_40/len(ens_confs)*100:.2f}%)")

    # 2. Single-pass configurations
    configs = ["adaptive_binary_psm6", "enhanced_grayscale_psm6", "otsu_binary_psm6", "adaptive_binary_psm4", "adaptive_binary_psm3"]
    for cfg in configs:
        confs = [o["single_pass"].get(cfg, {}).get("conf", 0.0) for o in ocr]
        times = [o["single_pass"].get(cfg, {}).get("time", 0.0) for o in ocr]
        succ = [o["single_pass"].get(cfg, {}).get("success", False) for o in ocr]
        print(f"Single-pass {cfg}:")
        print(f"  Success: {sum(succ)}/{len(ocr)}")
        print(f"  Confidence: min={min(confs):.2f}%, max={max(confs):.2f}%, mean={statistics.mean(confs):.2f}%, median={statistics.median(confs):.2f}%")
        print(f"  Time: mean={statistics.mean(times):.4f}s, min={min(times):.4f}s, max={max(times):.4f}s")

    # 3. Gemini Vision
    gem_succ = [o["gemini_vision"]["success"] for o in ocr]
    gem_times = [o["gemini_vision"]["time"] for o in ocr]
    print(f"\nGemini Vision OCR:")
    print(f"  Tested: {len(ocr)}, Successful: {sum(gem_succ)}, Failed: {len(ocr) - sum(gem_succ)}")
    print(f"  Mean Time: {statistics.mean(gem_times):.4f}s")
    print(f"  Failure reasons: {set(o['gemini_vision']['error'] for o in ocr)}")

    print("\n================ TRANSLATION STATISTICS ================")
    # deep-translator
    dt_succ = [t["deep_translator"]["success"] for t in trans]
    dt_times = [t["deep_translator"]["time"] for t in trans]
    print(f"deep-translator (Google Translate):")
    print(f"  Attempted: {len(trans)}, Successful: {sum(dt_succ)}, Failed: {len(trans) - sum(dt_succ)}")
    print(f"  Time (s): min={min(dt_times):.4f}s, max={max(dt_times):.4f}s, mean={statistics.mean(dt_times):.4f}s, median={statistics.median(dt_times):.4f}s")

    # FLAN-T5
    flan_succ = [t["flan_t5"]["success"] for t in trans]
    flan_times = [t["flan_t5"]["time"] for t in trans]
    print(f"FLAN-T5 Local Model:")
    print(f"  Attempted: {len(trans)}, Successful: {sum(flan_succ)}, Failed: {len(trans) - sum(flan_succ)}")
    print(f"  Time (s): min={min(flan_times):.4f}s, max={max(flan_times):.4f}s, mean={statistics.mean(flan_times):.4f}s, median={statistics.median(flan_times):.4f}s")

    # Tiered pipeline
    pipe_trans_succ = [t["tiered_pipeline"]["success"] for t in trans]
    pipe_trans_times = [t["tiered_pipeline"]["time"] for t in trans]
    tiers_used = [t["tiered_pipeline"]["tier_used"] for t in trans]
    print(f"Tiered Translation Pipeline:")
    print(f"  Attempted: {len(trans)}, Successful: {sum(pipe_trans_succ)}, Failed: {len(trans) - sum(pipe_trans_succ)}")
    print(f"  Time (s): min={min(pipe_trans_times):.4f}s, max={max(pipe_trans_times):.4f}s, mean={statistics.mean(pipe_trans_times):.4f}s, median={statistics.median(pipe_trans_times):.4f}s")
    tier_dist = {tier: tiers_used.count(tier) for tier in set(tiers_used)}
    print(f"  Tier distribution: {tier_dist}")

    print("\n================ NLP & EXPLANATION STATISTICS ================")
    exp_times = [n["explanation_time"] for n in nlp]
    print(f"Explanation Generation:")
    print(f"  Success: {len(nlp)}/{len(nlp)}")
    print(f"  Time (s): min={min(exp_times):.4f}s, max={max(exp_times):.4f}s, mean={statistics.mean(exp_times):.4f}s, median={statistics.median(exp_times):.4f}s")

    kw_counts = [n["keyword_count"] for n in nlp]
    kw_times = [n["keyword_time"] for n in nlp]
    print(f"Keyword Extraction:")
    print(f"  Success: {len(nlp)}/{len(nlp)}")
    print(f"  Keyword count per sample: min={min(kw_counts)}, max={max(kw_counts)}, mean={statistics.mean(kw_counts):.2f}, median={statistics.median(kw_counts):.2f}")
    print(f"  Time (s): min={min(kw_times):.4f}s, max={max(kw_times):.4f}s, mean={statistics.mean(kw_times):.4f}s, median={statistics.median(kw_times):.4f}s")

    domains = [n["domain"] for n in nlp]
    print(f"Domain Classification Distribution:")
    for cat in CANONICAL_CATEGORIES:
        cnt = domains.count(cat)
        pct = (cnt / len(domains)) * 100
        print(f"  {cat}: {cnt} samples ({pct:.2f}%)")

    print("\n================ END-TO-END PIPELINE TIMING ================")
    t_pre = [p["t_preprocessing"] for p in pipe]
    t_ocr = [p["t_ocr"] for p in pipe]
    t_trans = [p["t_translation"] for p in pipe]
    t_exp = [p["t_explanation"] for p in pipe]
    t_kw = [p["t_keywords"] for p in pipe]
    t_class = [p["t_classification"] for p in pipe]
    t_tot = [p["total_time"] for p in pipe]
    slowest_counts = {st: sum(1 for p in pipe if p["slowest_stage"] == st) for st in ["Preprocessing", "OCR", "Translation", "Explanation", "Keywords", "Classification"]}

    print(f"End-to-End Latency:")
    print(f"  Total Time (s): min={min(t_tot):.4f}s, max={max(t_tot):.4f}s, mean={statistics.mean(t_tot):.4f}s, median={statistics.median(t_tot):.4f}s")
    print(f"  Stage Breakdown (Means):")
    print(f"    Preprocessing: {statistics.mean(t_pre):.4f}s ({statistics.mean(t_pre)/statistics.mean(t_tot)*100:.2f}%)")
    print(f"    OCR Engine:    {statistics.mean(t_ocr):.4f}s ({statistics.mean(t_ocr)/statistics.mean(t_tot)*100:.2f}%)")
    print(f"    Translation:   {statistics.mean(t_trans):.4f}s ({statistics.mean(t_trans)/statistics.mean(t_tot)*100:.2f}%)")
    print(f"    Explanation:   {statistics.mean(t_exp):.4f}s ({statistics.mean(t_exp)/statistics.mean(t_tot)*100:.2f}%)")
    print(f"    Keywords:      {statistics.mean(t_kw):.4f}s ({statistics.mean(t_kw)/statistics.mean(t_tot)*100:.2f}%)")
    print(f"    Classification:{statistics.mean(t_class):.4f}s ({statistics.mean(t_class)/statistics.mean(t_tot)*100:.2f}%)")
    print(f"  Slowest Stage Frequencies: {slowest_counts}")

if __name__ == "__main__":
    run_stats()
