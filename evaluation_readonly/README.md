# SansKrypt Read-Only Experimental Evaluation

This directory contains strict, read-only empirical evaluation scripts and reproducible raw experimental logs conducted on the SansKrypt Sanskrit Digitization & Translation system.

## Directory Structure
* `audit_dataset.py`: Read-only script auditing the manuscript sample files, resolutions, formats, and database tables.
* `run_evaluation.py`: Independent evaluation driver measuring OCR configurations, translation tiers, NLP stages, and end-to-end latency.
* `aggregate_metrics.py`: Statistical computation suite calculating distributions, percentiles, means, and std deviations.
* `results/`:
  * `ocr_results.csv`: Per-sample OCR output, calibrated confidence scores, and pass timings.
  * `translation_results.csv`: Tiered translation outputs, attempt times, and tier activations.
  * `explanation_results.csv`: Generated plain-English explanations and generation latencies.
  * `keyword_results.csv`: Extracted Sanskrit and English conceptual keywords and counts.
  * `classification_results.csv`: Predicted canonical knowledge domain classifications.
  * `pipeline_results.csv`: Complete end-to-end breakdown with per-stage latencies and slowest stage tagging.
  * `environment.txt`: Hardware, OS, Python runtime, and package versions.
* `raw/`:
  * `evaluation_data.json`: Comprehensive raw JSON recording all intermediate and final experimental variables.
* `SANSKRYPT_EVALUATION_REPORT.md`: Comprehensive IEEE technical paper-ready experimental report.

## Reproduction Command
To reproduce all experiments from the project root in read-only mode:
```bash
python evaluation_readonly/run_evaluation.py
python evaluation_readonly/aggregate_metrics.py
```
