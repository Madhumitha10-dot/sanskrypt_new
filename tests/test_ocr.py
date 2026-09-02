"""
Unit tests for Sanskrit OCR extraction and confidence calculation.
"""
import pytest
from unittest.mock import patch
from ocr import extract_text


def test_extract_text_high_confidence(sample_image_path):
    """Verifies text extraction and average confidence computation when confidence is high."""
    mock_data = {
        "text": ["", "सत्यमेव", "जयते", "नानृतम्", ""],
        "conf": ["-1", "92", "88", "90", "-1"],
        "block_num": [1, 1, 1, 1, 1],
        "par_num": [1, 1, 1, 1, 1],
        "line_num": [1, 1, 1, 1, 1]
    }

    with patch("ocr.extract_with_gemini_vision", return_value=None), \
         patch("pytesseract.image_to_data", return_value=mock_data):
        text, confidence = extract_text(sample_image_path)
        assert "सत्यमेव जयते नानृतम्" in text
        assert confidence >= 80.0


def test_extract_text_calibrated_confidence(sample_image_path):
    """Verifies that recognized Sanskrit text receives calibrated confidence above 50%."""
    mock_data = {
        "text": ["", "अथ", "योग", "अनुशासनम्"],
        "conf": ["-1", "25", "30", "35"],
        "block_num": [1, 1, 1, 1],
        "par_num": [1, 1, 1, 1],
        "line_num": [1, 1, 1, 1]
    }

    with patch("ocr.extract_with_gemini_vision", return_value=None), \
         patch("pytesseract.image_to_data", return_value=mock_data):
        text, confidence = extract_text(sample_image_path)
        assert "अथ योग अनुशासनम्" in text
        assert confidence > 50.0


def test_extract_text_empty_ocr(sample_image_path):
    """Verifies handling when no words are detected."""
    mock_data = {
        "text": ["", ""],
        "conf": ["-1", "-1"],
        "block_num": [1, 1],
        "par_num": [1, 1],
        "line_num": [1, 1]
    }

    with patch("ocr.extract_with_gemini_vision", return_value=None), \
         patch("pytesseract.image_to_data", return_value=mock_data):
        text, confidence = extract_text(sample_image_path)
        assert text == ""
        assert confidence == 0.0


def test_extract_text_file_not_found():
    """Verifies FileNotFoundError on invalid input path."""
    with patch("ocr.extract_with_gemini_vision", return_value=None):
        with pytest.raises(FileNotFoundError):
            extract_text("non_existent_image_file.png")
