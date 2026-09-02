# SansKrypt: System for Translating Manuscripts

**SansKrypt** is a lightweight, CPU-friendly Flask web application and knowledge repository that digitizes ancient Sanskrit manuscript images into high-fidelity Devanagari text, generates English translations, synthesizes contextual plain-English explanations, extracts conceptual keywords, and automatically classifies manuscripts into knowledge domains (Philosophy, Ayurveda, Astronomy, Literature, Mathematics, Vedic Sciences, Grammar, Yoga, etc.).

---

## Key Features

1. **OpenCV Manuscript Enhancement**: Multi-stage image preprocessing (grayscale conversion, bilateral denoising to eliminate aged paper grain, CLAHE contrast equalization, and adaptive Otsu binarization).
2. **Sanskrit OCR with Word-Level Confidence Scoring**: Uses `pytesseract` with Sanskrit (`san`) trained data. Computes average word confidence across bounding boxes.
3. **Automated Low-Confidence Detection**:
   - `LOW_CONFIDENCE_THRESHOLD = 40` (tunable in `config.py`).
   - If confidence is below 40%, the entire pipeline still completes, but the manuscript is automatically marked `status="low_confidence"`.
   - Displays a visible warning banner on the results page advising manual verification against the original manuscript.
   - Status filters enable researchers to review all low-confidence manuscripts in the repository.
4. **Hugging Face Transformer Translation**: Translates Sanskrit into English using lightweight Seq2Seq transformer models (e.g., FLAN-T5).
5. **Contextual LLM Explanations & Domain Classification**: Explains traditional philosophical and scientific concepts, extracts Devanagari/English keywords, and classifies manuscripts into canonical Sanskrit knowledge domains.
6. **SQLite Knowledge Repository & Multi-Criteria Search**: Stores all artifacts locally via SQLAlchemy with full-text search across translations, original OCR text, keywords, domain classifications, and confidence/status filters.

---

## Architecture & Module Structure

```
Sanskrypt_major/
│
├── app.py                      # Flask routes and pipeline orchestration
├── config.py                   # Central settings, thresholds, and paths
├── preprocessing.py            # OpenCV image enhancement pipeline
├── ocr.py                      # Tesseract OCR & confidence computation
├── translation.py              # Sanskrit -> English translation pipeline
├── explain_classify.py         # LLM explanation, keywords & domain classification
├── database.py                 # SQLite models (Manuscript) & CRUD methods
├── search.py                   # Multi-criteria search and status filtering
├── requirements.txt            # Python dependencies
├── README.md                   # Installation & setup guide
│
├── uploads/                    # Local storage for manuscript images
│   ├── original/               # Original uploaded images
│   └── processed/              # OpenCV preprocessed images
│
├── static/
│   ├── css/style.css           # Modern scholarly Indian dark & parchment styles
│   └── js/main.js              # Drag-and-drop uploads, previews, copy buttons
│
├── templates/                  # Jinja2 templates (Bootstrap 5)
│   ├── base.html               # Base layout with navbar, alerts & footer
│   ├── index.html              # Upload page with dropzone & features
│   ├── results.html            # Detail view with confidence banner & comparison
│   ├── search.html             # Multi-filter search interface
│   ├── search_results.html     # Search results grid & filter summaries
│   ├── manuscripts.html        # Repository browsing with status filters
│   └── error.html              # Error pages (404, 500)
│
└── tests/                      # Pytest unit tests (all modules mocked for speed)
    ├── conftest.py
    ├── test_preprocessing.py
    ├── test_ocr.py
    ├── test_translation.py
    ├── test_explain_classify.py
    ├── test_database.py
    ├── test_search.py
    └── test_app.py
```

---

## Installation & Setup Guide

### 1. Prerequisites
- Python 3.9+ installed on your system (Tested on Python 3.11).
- 8GB RAM minimum (runs locally on CPU, no GPU required).

---

### 2. Install Tesseract OCR & Sanskrit Language Pack

Tesseract is a system binary and **must be installed separately from Python packages**.

#### **Windows:**
1. Download the Tesseract installer from UB-Mannheim:
   - [Tesseract Windows Installer Releases](https://github.com/UB-Mannheim/tesseract/wiki)
2. Run the installer (e.g., `tesseract-ocr-w64-setup-5.x.x.exe`).
3. **Important during installation**: On the **"Choose Components"** screen, expand **"Additional language data"** and check **"Sanskrit"** (`san.traineddata`), or download `san.traineddata` manually.
4. If downloading `san.traineddata` manually:
   - Download `san.traineddata` from [tessdata_fast](https://github.com/tesseract-ocr/tessdata_fast/raw/main/san.traineddata).
   - Place the file inside your Tesseract `tessdata` directory:
     ```
     C:\Program Files\Tesseract-OCR\tessdata\san.traineddata
     ```
5. Add Tesseract to your Windows System PATH (`C:\Program Files\Tesseract-OCR`), or set the environment variable:
   ```cmd
   set TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
   ```

#### **Linux (Ubuntu / Debian):**
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-san
```

#### **macOS (Homebrew):**
```bash
brew install tesseract
brew install tesseract-lang
```

---

### 3. Python Virtual Environment & Dependencies

1. Clone or navigate to the project root directory:
   ```bash
   cd Sanskrypt_major
   ```

2. (Optional but recommended) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux/macOS:
   source venv/bin/activate
   ```

3. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

---

### 4. Running the Application

Start the local Flask development server:

```bash
python app.py
```

Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

---

### 5. Running the Pytest Test Suite

All external ML models and OCR dependencies are mocked in unit tests so the test suite runs fast and reliably without requiring model downloads or external binaries:

```bash
pytest -v
```

---

## Configuration & Tuning

Configuration constants can be adjusted in [config.py](file:///c:/Users/madhu/Desktop/Sanskrypt_major/config.py) or via environment variables:

| Setting | Default | Description |
|---|---|---|
| `LOW_CONFIDENCE_THRESHOLD` | `40.0` | Threshold percentage below which manuscripts are tagged `low_confidence`. |
| `MAX_CONTENT_LENGTH` | `16 MB` | Maximum allowed file upload size. |
| `TRANSLATION_MODEL_NAME` | `"google/flan-t5-small"` | Lightweight CPU-friendly translation Seq2Seq model. |
| `LLM_MODEL_NAME` | `"google/flan-t5-small"` | Model for explanations, keywords, and classification. |
| `SQLALCHEMY_DATABASE_URI` | `"sqlite:///sanskrypt.db"` | SQLite database connection string. |

---

## License & Scholarly Purpose
Created as an open digital humanities and AI knowledge preservation system for Sanskrit cultural manuscripts.
