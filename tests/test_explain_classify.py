"""
Unit tests for explanation generation, keyword extraction, and domain classification.
"""
from unittest.mock import patch, MagicMock
from explain_classify import (
    generate_explanation,
    extract_keywords,
    classify_content,
    LLMPipeline
)


def test_classify_content_rules():
    """Verifies precise rule-based domain classification."""
    assert classify_content("वायुः पित्तं कफश्च", "Vata, Pitta, and Kapha") == "Ayurveda"
    assert classify_content("सूर्य सिद्धान्त ग्रह", "Sun, planets, and eclipses") == "Astronomy"
    assert classify_content("गणित शून्य", "Mathematics calculation of zero and algebra") == "Mathematics"
    assert classify_content("अथ योगानुशासनम्", "Now begins the instruction on Yoga and meditation") == "Yoga & Spirituality"
    assert classify_content("अथ शब्दानुशासनम्", "Panini sutra grammar vyakarana declension") == "Grammar"
    assert classify_content("अहं ब्रह्मास्मि", "I am Brahman, the absolute soul and reality") == "Philosophy"


def test_extract_keywords():
    """Verifies keyword extraction from Sanskrit and translated English."""
    mock_instance = MagicMock()
    mock_instance.generate.return_value = "Truth, Victory, Untruth, Virtue, Dharma"

    with patch("explain_classify.generate_with_gemini", return_value=None), \
         patch.object(LLMPipeline, "get_instance", return_value=mock_instance):
        keywords = extract_keywords("सत्यमेव जयते", "Truth alone triumphs")
        assert "Truth" in keywords
        assert "Victory" in keywords


def test_generate_explanation():
    """Verifies plain-English explanation generation."""
    mock_instance = MagicMock()
    mock_instance.generate.return_value = "This verse emphasizes that moral truth always prevails over falsehood."

    with patch("explain_classify.generate_with_gemini", return_value=None), \
         patch.object(LLMPipeline, "get_instance", return_value=mock_instance):
        explanation = generate_explanation("सत्यमेव जयते", "Truth alone triumphs")
        assert "moral truth" in explanation
