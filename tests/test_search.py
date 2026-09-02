"""
Unit tests for multi-criteria search operations and status filtering.
"""
from search import (
    search_by_keyword,
    search_by_classification,
    search_by_text,
    search_manuscripts
)


def test_search_by_keyword(sample_manuscripts, app):
    """Verifies keyword search matches and status filtering."""
    with app.app_context():
        results = search_by_keyword("Ayurveda")
        assert len(results) == 1
        assert "Tridosha" in results[0].title

        # Search with status filter
        results_completed = search_by_keyword("Ayurveda", status="completed")
        assert len(results_completed) == 0

        results_low = search_by_keyword("Ayurveda", status="low_confidence")
        assert len(results_low) == 1


def test_search_by_classification(sample_manuscripts, app):
    """Verifies classification filtering."""
    with app.app_context():
        results = search_by_classification("Mathematics")
        assert len(results) == 1
        assert "Aryabhatiya" in results[0].title


def test_search_by_text(sample_manuscripts, app):
    """Verifies text search over translation, original text, and explanation."""
    with app.app_context():
        # Search by English translation word
        results = search_by_text("non-existence")
        assert len(results) == 1
        assert "Rigveda" in results[0].title

        # Search by Devanagari text fragment
        results_dev = search_by_text("त्रयो दोषाः")
        assert len(results_dev) == 1


def test_unified_search_manuscripts(sample_manuscripts, app):
    """Verifies advanced faceted search combinations."""
    with app.app_context():
        # Search by query + status
        results = search_manuscripts(query="Creation", status="completed")
        assert len(results) == 1

        # Search by low confidence review queue
        results_review = search_manuscripts(status="low_confidence")
        assert len(results_review) == 1
        assert results_review[0].ocr_confidence < 40.0
