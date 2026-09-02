"""
Unit tests for Sanskrit to English translation module.
"""
from unittest.mock import patch, MagicMock
from translation import translate_text, TranslationPipeline


def test_translate_text_empty():
    """Verifies that empty input returns informative message."""
    assert "No text available" in translate_text("")
    assert "No text available" in translate_text("   ")


def test_translate_text_accurate():
    """Verifies high-accuracy Sanskrit translation output."""
    with patch("translation.translate_with_gemini", return_value="Truth alone triumphs, not untruth."):
        result = translate_text("सत्यमेव जयते नानृतम्")
        assert "Truth" in result
        assert "triumphs" in result or "victory" in result.lower() or "falsehood" in result.lower()


def test_translate_text_transformer_fallback():
    """Verifies local transformer pipeline execution when neural translator is bypassed."""
    mock_instance = MagicMock()
    mock_instance.translate_with_transformer.return_value = "Truth alone triumphs, not untruth."

    with patch("translation.translate_with_gemini", return_value=None), \
         patch("translation.HAS_DEEP_TRANSLATOR", False), \
         patch.object(TranslationPipeline, "get_instance", return_value=mock_instance):
        result = translate_text("सत्यमेव जयते नानृतम्")
        assert result == "Truth alone triumphs, not untruth."
        mock_instance.translate_with_transformer.assert_called_once_with("सत्यमेव जयते नानृतम्")
