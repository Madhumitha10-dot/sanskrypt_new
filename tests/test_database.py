"""
Unit tests for database module CRUD operations and model integrity.
"""
from database import (
    save_manuscript,
    get_manuscript,
    list_manuscripts,
    update_manuscript,
    delete_manuscript,
    Manuscript
)


def test_save_and_get_manuscript(app):
    """Verifies saving and retrieving a manuscript with confidence and status."""
    with app.app_context():
        manuscript = save_manuscript(
            title="Test Gita Verse",
            original_image_path="uploads/original/gita.png",
            processed_image_path="uploads/processed/gita_proc.png",
            ocr_text="कर्मण्येवाधिकारस्ते मा फलेषु कदाचन",
            ocr_confidence=35.5,
            translated_text="You have a right to perform your prescribed duties, but you are not entitled to the fruits of your actions.",
            explanation="Karma Yoga principle of selfless action without attachment to results.",
            keywords="Karma, Duty, Gita, Krishna, Action",
            classification="Philosophy",
            status="low_confidence"
        )
        assert manuscript.id is not None
        assert manuscript.status == "low_confidence"
        assert manuscript.ocr_confidence == 35.5

        fetched = get_manuscript(manuscript.id)
        assert fetched is not None
        assert fetched.title == "Test Gita Verse"
        assert fetched.classification == "Philosophy"


def test_list_manuscripts_filtered(sample_manuscripts, app):
    """Verifies listing manuscripts with status and classification filters."""
    with app.app_context():
        # List all
        all_m = list_manuscripts()
        assert len(all_m) == 3

        # Filter by status: low_confidence
        low_conf = list_manuscripts(status="low_confidence")
        assert len(low_conf) == 1
        assert low_conf[0].title == "Charaka Samhita Tridosha"

        # Filter by status: completed
        completed = list_manuscripts(status="completed")
        assert len(completed) == 2

        # Filter by classification: Ayurveda
        ayurveda = list_manuscripts(classification="Ayurveda")
        assert len(ayurveda) == 1


def test_update_and_delete_manuscript(sample_manuscripts, app):
    """Verifies updating and deleting manuscript records."""
    m_id = sample_manuscripts[0]

    with app.app_context():
        updated = update_manuscript(m_id, title="Updated Hymn Title")
        assert updated.title == "Updated Hymn Title"

        success = delete_manuscript(m_id)
        assert success is True

        assert get_manuscript(m_id) is None
