"""
SansKrypt Configuration Module
Central settings for file uploads, OCR, translation, LLM inference, and database.
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Auto-load .env file if present
env_path = BASE_DIR / ".env"
if env_path.is_file():
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip().strip("\"'")
    except Exception:
        pass

# File storage directories
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads", "original")
PROCESSED_FOLDER = os.path.join(BASE_DIR, "uploads", "processed")

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Upload constraints
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "tiff", "bmp"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max file size

# Database configuration
DATABASE_FILE = os.path.join(BASE_DIR, "sanskrypt.db")
SQLALCHEMY_DATABASE_URI = f"sqlite:///{Path(DATABASE_FILE).as_posix()}"
SQLALCHEMY_TRACK_MODIFICATIONS = False

# API Keys & Cloud Services
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))

# OCR Settings
LOW_CONFIDENCE_THRESHOLD = 40.0  # Tunable threshold (0 - 100)
OCR_LANGUAGE = "san"             # Sanskrit Tesseract traineddata

# Machine Learning / NLP Models (CPU-Friendly)
TRANSLATION_MODEL_NAME = os.getenv("TRANSLATION_MODEL", "google/flan-t5-small")
LLM_MODEL_NAME = os.getenv("LLM_MODEL", "google/flan-t5-small")

# Predefined Knowledge Categories
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

# Flask Secret Key
SECRET_KEY = os.getenv("SECRET_KEY", "sanskrypt-major-secret-key-2026")
