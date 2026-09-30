# SansKrypt: AI-Powered Sanskrit Manuscript Knowledge Repository

**SansKrypt** is an intelligent web application and digital humanities platform designed to digitize, translate, and preserve ancient Sanskrit manuscripts. The system processes scanned manuscript images through computer vision preprocessing, extracts Devanagari text using OCR with confidence scoring, generates faithful English translations, produces contextual plain-English explanations without over-interpretation, extracts keywords, and automatically classifies manuscripts into traditional knowledge domains within a searchable repository.

---

## Overview

SansKrypt bridges ancient Indian manuscript heritage and modern artificial intelligence. Scanned manuscript images often suffer from historical degradation, paper grain noise, ink bleed, and fading. SansKrypt addresses these challenges by:
1. Enhancing raw manuscript images using bilateral filtering, CLAHE contrast equalization, and adaptive binarization.
2. Transcribing Sanskrit Devanagari script using multi-engine OCR (Tesseract with `san` language data and Google Gemini Vision) alongside word-level confidence scoring.
3. Translating Sanskrit text into English using a tiered pipeline (Gemini API, Google Translate via deep-translator, and local Hugging Face Seq2Seq models).
4. Generating grounded plain-English explanations based strictly on the translated meaning, extracting Devanagari and English keywords, and categorizing texts into canonical Sanskrit domains.
5. Indexing all transcribed texts, translations, metadata, and confidence scores into a searchable local repository.

---

## Features

### Currently Implemented Features

* **Sanskrit Manuscript Image Upload**: Interactive drag-and-drop file upload supporting PNG, JPG, JPEG, TIFF, and BMP formats with client-side preview and size validation.
* **Image Preprocessing Pipeline**: Automated multi-stage image enhancement via OpenCV (grayscale conversion, bilateral filtering for paper texture denoising, CLAHE contrast enhancement, and Otsu/adaptive thresholding).
* **Sanskrit OCR Transcription**: High-accuracy text recognition supporting Devanagari script using Tesseract OCR with `san.traineddata` and Google Gemini Vision.
* **OCR Confidence Scoring & Quality Warning**: Word-level and document-level confidence metrics calculation. Low-confidence transcriptions (< 40%) are flagged with visible verification alerts.
* **Multi-Engine Sanskrit-to-English Translation**: Multi-tier translation pipeline prioritizing Gemini AI, falling back to Google Translate (`deep-translator`), and local Hugging Face transformer models.
* **Grounded Simple English Explanation**: Contextual explanation derived strictly from the translated verse without introducing unsubstantiated philosophical extrapolations.
* **Bilingual Keyword Extraction**: Identification of significant Sanskrit concepts and corresponding English terms.
* **Domain Classification**: Automatic categorization into canonical Sanskrit knowledge domains (e.g., Philosophy / Darśana, Ayurveda & Medicine, Astronomy & Jyotiṣa, Literature & Kāvya, Mathematics / Gaṇita, Vedic Sciences, Grammar / Vyākaraṇa, Yoga).
* **Searchable Manuscript Repository**: SQLite-backed repository enabling multi-field search across titles, Devanagari text, English translations, keywords, domain classifications, and confidence status filters.

---

## Technology Stack

### Frontend
* **HTML5** & **Jinja2 Templates**: Semantic markup and server-rendered templates.
* **Bootstrap 5**: Responsive layout, cards, badges, modal alerts, and typography.
* **Custom CSS**: Scholarly parchment and dark theme accents.
* **Vanilla JavaScript**: Asynchronous upload handler, file preview, and clipboard utilities.

### Backend
* **Python 3.9+ / 3.11**
* **Flask**: Web framework and RESTful route handling.
* **SQLAlchemy**: ORM for database operations and structured querying.

### Optical Character Recognition (OCR)
* **Tesseract OCR (`pytesseract`)**: Open-source OCR engine.
* **Sanskrit Language Pack (`san.traineddata`)**: Trained Devanagari character models.
* **Google Gemini Vision (`google-generativeai`)**: Multimodal visual transcription for manuscripts.

### Image Processing
* **OpenCV (`opencv-python`)**: Computer vision algorithms for denoising, CLAHE, and thresholding.
* **Pillow (PIL)**: Image loading, format conversion, and manipulation.
* **NumPy**: Matrix and array computations for pixel operations.

### Artificial Intelligence & Machine Learning
* **Google Gemini API (`gemini-1.5-flash` / `gemini-1.5-pro`)**: Multimodal OCR, translation, and grounded explanation.
* **Hugging Face Transformers (`transformers`)**: Local sequence-to-sequence neural models.
* **FLAN-T5 (`google/flan-t5-small` / `base`)**: Local CPU-friendly translation and text generation.
* **PyTorch (`torch`)**: Deep learning tensor runtime.

### Translation Services
* **Google Gemini API**: Context-aware Sanskrit-to-English translation.
* **Google Translate (`deep-translator`)**: Neural machine translation service.
* **FLAN-T5 Seq2Seq Fallback**: Local offline translation model.

### Database
* **SQLite 3**: Embedded relational database.
* **SQLAlchemy**: Data modeling and query abstraction.

### Testing & Quality Assurance
* **pytest**: Unit and integration test framework.
* **pytest-mock**: Test isolation and dependency mocking for external APIs and heavy ML models.

---

## Architecture

```
User / Researcher
       │
       ▼
Flask / Jinja2 Frontend (Bootstrap 5, Dropzone, Viewers)
       │
       ▼
Flask Backend (app.py Routing & Pipeline Controller)
       │
       ├──► Image Preprocessing (OpenCV: Bilateral Filter, CLAHE, Otsu)
       │
       ├──► Sanskrit OCR (Tesseract `san` + Gemini Vision + Confidence Scoring)
       │
       ├──► Translation Pipeline (Gemini ➔ deep-translator ➔ FLAN-T5)
       │
       ├──► Explanation & Classification (Grounded Explanation, Keywords, Domain)
       │
       ▼
SQLite Database (database.py / SQLAlchemy ORM)
       │
       ▼
Searchable Knowledge Repository (search.py Multi-Filter Query Engine)
```

---

## Project Structure

```
Sanskrypt_major/
│
├── app.py                      # Flask application entry point and route handlers
├── config.py                   # Centralized configuration, thresholds, and paths
├── database.py                 # SQLAlchemy database schema and CRUD operations
├── preprocessing.py            # OpenCV manuscript enhancement pipeline
├── ocr.py                      # Tesseract & Gemini OCR transcription engine
├── translation.py              # Sanskrit-to-English multi-tier translation
├── explain_classify.py         # Grounded explanation, keywords, and domain classifier
├── search.py                   # Multi-criteria search and repository filter logic
├── train_translation_model.py  # Script for local fine-tuning of Seq2Seq models
├── requirements.txt            # Python dependencies
├── .env.example                # Template for environment configuration
├── .gitignore                  # Git exclusion rules for secrets, caches, and binaries
├── README.md                   # Project documentation and developer guide
│
├── templates/                  # Jinja2 HTML templates
│   ├── base.html               # Shared layout, navbar, and flash message container
│   ├── index.html              # Manuscript upload page with drag-and-drop zone
│   ├── results.html            # Processing results with side-by-side comparison
│   ├── search.html             # Multi-filter search interface
│   ├── search_results.html     # Search results grid and filter summaries
│   ├── manuscripts.html        # Repository browser with status filters
│   └── error.html              # Standard error pages (404, 500)
│
├── static/                     # Static assets
│   ├── css/
│   │   └── style.css           # Scholarly Indian manuscript and modern UI styles
│   └── js/
│       └── main.js             # Drag-and-drop, image preview, and UI interactivity
│
├── uploads/                    # Local storage for manuscript images (ignored by git)
│   ├── original/               # Stored original uploads (.gitkeep tracked)
│   └── processed/              # Stored preprocessed images (.gitkeep tracked)
│
└── tests/                      # Automated test suite (Pytest)
    ├── conftest.py             # Test fixtures and mock setups
    ├── test_app.py             # Route and integration tests
    ├── test_database.py        # Database CRUD unit tests
    ├── test_explain_classify.py# Explanation and classification unit tests
    ├── test_ocr.py             # OCR transcription and confidence tests
    ├── test_preprocessing.py   # OpenCV image enhancement tests
    ├── test_search.py          # Search query engine tests
    └── test_translation.py     # Translation pipeline tests
```

---

## Installation & Setup Guide

### 1. Prerequisites
* **Python 3.9+** (Tested on Python 3.11).
* **Git** installed on your system.
* **Tesseract OCR** with Sanskrit (`san`) trained data.

---

### 2. Clone the Repository

```bash
git clone https://github.com/your-username/sanskrypt.git
cd sanskrypt
```

---

### 3. Create and Activate a Virtual Environment

* **On Windows (PowerShell / Command Prompt):**
  ```bash
  python -m venv venv
  .\venv\Scripts\activate
  ```

* **On Linux / macOS:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

---

### 4. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 5. Install & Configure Tesseract OCR

Tesseract is an external system binary and must be installed separately from Python packages.

#### Windows:
1. Download the installer from UB-Mannheim: [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki).
2. Run the installer (e.g. `tesseract-ocr-w64-setup-5.x.x.exe`).
3. During installation, on the **"Choose Components"** step, expand **"Additional language data (download)"** and select **"Sanskrit"** (`san`).
4. Alternatively, download [`san.traineddata`](https://github.com/tesseract-ocr/tessdata_fast/raw/main/san.traineddata) manually and place it into your `C:\Program Files\Tesseract-OCR\tessdata\` directory.
5. Add `C:\Program Files\Tesseract-OCR` to your Windows System PATH, or define `TESSERACT_CMD` in your `.env` file.

#### Linux (Ubuntu / Debian):
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-san
```

#### macOS (Homebrew):
```bash
brew install tesseract
brew install tesseract-lang
```

---

### 6. Configure Environment Variables

1. Copy the example configuration file:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and set your configuration variables:

```ini
# Google Gemini API Key (Required for Gemini OCR transcription and translation)
GEMINI_API_KEY=your_gemini_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

# Flask secret key for secure sessions
SECRET_KEY=your_secret_key_here

# Path to Tesseract binary (optional if already on system PATH)
# TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

> **Note:** Never commit the `.env` file or any real API keys to version control.

---

### 7. Run the Application

Start the Flask development server:

```bash
python app.py
```

Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

---

### 8. Run Automated Tests

The test suite uses mocks for heavy neural models and OCR binaries to ensure fast, reproducible execution:

```bash
pytest -v
```

---

## Environment Variables

| Variable | Description | Default / Example |
|---|---|---|
| `GEMINI_API_KEY` | API key for Google Gemini multimodal vision & LLM inference | *None (User provided)* |
| `GOOGLE_API_KEY` | Secondary alias for Google Gemini API key | *None (User provided)* |
| `SECRET_KEY` | Secret key used for Flask session cryptography | `sanskrypt-secret-key-change-in-production` |
| `TESSERACT_CMD` | Explicit filesystem path to Tesseract OCR executable | Auto-detected from PATH |
| `TRANSLATION_MODEL` | Hugging Face model identifier for local translation fallback | `google/flan-t5-small` |
| `LLM_MODEL` | Hugging Face model identifier for local text analysis | `google/flan-t5-small` |

---

## Future Roadmap

The following advanced computational linguistics and AI capabilities are planned for future iterations of SansKrypt:

> [!NOTE]
> The features listed below are **PLANNED / NOT CURRENTLY IMPLEMENTED** in the current release.

* **Grammar-Aware Sanskrit Analysis**: Deep structural analysis based on Pāṇinian grammatical rules (*Aṣṭādhyāyī*). *(Not currently implemented)*
* **Sandhi Splitting & Analysis**: Automated phonetic compound splitting (*Sandhi-viccheda*) and rule identification. *(Not currently implemented)*
* **Morphological Analysis (*Pada-viśleṣaṇa*)**: Detailed parsing of word roots, prefixes, suffixes, and conjugations. *(Not currently implemented)*
* **Vibhakti & Dhātu Identification**: Extraction of noun cases (*Vibhakti*), genders (*Liṅga*), numbers (*Vacana*), verbal roots (*Dhātu*), and tense/mood markers (*Lakāra*). *(Not currently implemented)*
* **Sanskrit Syntax & Dependency Parsing (*Kāraka* Analysis)**: Mapping syntactic roles, actor-action dependencies, and semantic relations. *(Not currently implemented)*
* **Samāsa Analysis**: Compound word decomposition and classification (Tatpuruṣa, Bahuvrīhi, Dvandva, Avyayībhāva, etc.). *(Not currently implemented)*
* **Sanskrit RAG (Retrieval-Augmented Generation)**: Grounding explanations in canonical commentaries and classical Sanskrit corpus embeddings. *(Not currently implemented)*
* **Semantic Vector Search**: Dense vector search across Devanagari embeddings for conceptual retrieval. *(Not currently implemented)*
* **Similar Manuscript Recommendation**: Content-based and embedding-based manuscript clustering and cross-referencing. *(Not currently implemented)*
* **Knowledge Graph Generation**: Automated construction of interconnected entities, philosophical concepts, authors, and lineage graphs. *(Not currently implemented)*

---

## License & Scholarly Purpose

This project is developed for digital humanities research, academic preservation, and educational exploration of classical Sanskrit literature.
