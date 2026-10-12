"""
Unit and integration tests for SansKrypt manuscript status logic and OCR quality checks.
Verifies all 12 testing requirements deterministically using mocked OCR components.
"""
import io
import pytest
from unittest.mock import patch
from PIL import Image
import numpy as np

from ocr import evaluate_ocr_quality
from database import get_manuscript, list_manuscripts, Manuscript
from config import LOW_CONFIDENCE_THRESHOLD


# ---------------------------------------------------------------------------
# Unit Tests for OCR Quality Gate Helper (evaluate_ocr_quality)
# ---------------------------------------------------------------------------

def test_clear_sanskrit_marked_completed():
    """Requirement 1: A clear Sanskrit manuscript with usable OCR output can be marked Completed."""
    sanskrit_text = "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः । मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय ॥"
    confidence = 92.5
    result = evaluate_ocr_quality(sanskrit_text, confidence)

    assert result["status"] == "completed"
    assert result["is_usable"] is True
    assert result["reasons"] == []
    assert result["metrics"]["devanagari_chars"] > 10
    assert result["metrics"]["devanagari_ratio"] > 0.8


def test_hindi_devanagari_marked_completed():
    """Requirement 2: A genuine Hindi Devanagari manuscript with usable OCR output can be marked Completed."""
    hindi_text = "कबीर दास के दोहे: बुरा जो देखन मैं चला, बुरा न मिलिया कोय । जो दिल खोजा आपना, मुझसे बुरा न कोय ॥"
    confidence = 88.0
    result = evaluate_ocr_quality(hindi_text, confidence)

    assert result["status"] == "completed"
    assert result["is_usable"] is True
    assert result["reasons"] == []
    assert result["metrics"]["devanagari_ratio"] > 0.8


def test_low_confidence_marked_needs_review():
    """Requirement 3: Low-confidence OCR output is marked Needs Review (low_confidence)."""
    sanskrit_text = "अहं ब्रह्मास्मि"
    confidence = 35.0  # Below 40.0%
    result = evaluate_ocr_quality(sanskrit_text, confidence)

    assert result["status"] == "low_confidence"
    assert result["is_usable"] is False
    assert any("below minimum threshold" in r for r in result["reasons"])


def test_empty_ocr_marked_needs_review():
    """Requirement 4: Empty or whitespace OCR output is marked Needs Review."""
    # Test empty string
    res_empty = evaluate_ocr_quality("", 0.0)
    assert res_empty["status"] == "low_confidence"
    assert res_empty["is_usable"] is False

    # Test whitespace-only string
    res_ws = evaluate_ocr_quality("   \n\t  ", 0.0)
    assert res_ws["status"] == "low_confidence"
    assert res_ws["is_usable"] is False

    # Test fallback placeholder string
    placeholder = "[No readable Sanskrit text characters could be identified by OCR]"
    res_placeholder = evaluate_ocr_quality(placeholder, 0.0)
    assert res_placeholder["status"] == "low_confidence"
    assert res_placeholder["is_usable"] is False


def test_excessive_noise_marked_needs_review():
    """Requirement 5: OCR output containing excessive noise or suspicious symbols is marked Needs Review."""
    # Text with heavy symbol noise artifacts
    noisy_text = "§#*~%^&! {[]}_+? \u0915\u0916"
    confidence = 75.0
    result = evaluate_ocr_quality(noisy_text, confidence)

    assert result["status"] == "low_confidence"
    assert result["is_usable"] is False
    assert any("noise" in r.lower() or "consistency" in r.lower() for r in result["reasons"])


def test_high_confidence_unusable_repeated_garbage():
    """Requirement 6a: High-confidence but clearly unusable repeated garbage output is marked Needs Review."""
    # Stutter loop pattern
    garbage_text = "कककककककककककक"
    confidence = 96.0  # Very high confidence score
    result = evaluate_ocr_quality(garbage_text, confidence)

    assert result["status"] == "low_confidence"
    assert result["is_usable"] is False
    assert any("repetition" in r.lower() for r in result["reasons"])


def test_high_confidence_unusable_replacement_chars():
    """Requirement 6b: High-confidence OCR with excessive replacement chars is marked Needs Review."""
    unusable_text = "सत्य?????? नानृ????"
    confidence = 85.0
    result = evaluate_ocr_quality(unusable_text, confidence)

    assert result["status"] == "low_confidence"
    assert result["is_usable"] is False
    assert any("replacement" in r.lower() for r in result["reasons"])


def test_high_confidence_unusable_predominantly_latin():
    """Requirement 6c: High-confidence OCR with predominantly non-Devanagari Latin text is marked Needs Review."""
    latin_noise_text = "The quick brown fox jumps over the lazy dog \u0915"
    confidence = 89.0
    result = evaluate_ocr_quality(latin_noise_text, confidence)

    assert result["status"] == "low_confidence"
    assert result["is_usable"] is False
    assert any("non-devanagari" in r.lower() or "consistency" in r.lower() for r in result["reasons"])


def test_high_confidence_unusable_isolated_combining_marks():
    """Requirement 6d: High-confidence OCR with only isolated combining marks without base letters is marked Needs Review."""
    marks_only = "\u093e\u093f\u0940\u0941\u0942\u0943\u0947\u0948"
    confidence = 82.0
    result = evaluate_ocr_quality(marks_only, confidence)

    assert result["status"] == "low_confidence"
    assert result["is_usable"] is False
    assert any("base letter" in r.lower() for r in result["reasons"])


def test_high_confidence_unusable_excessive_fragmentation():
    """Requirement 6e: High-confidence OCR with fragmented isolated symbols/numerals instead of words is marked Needs Review."""
    fragmented_text = "७ ४७५= १ ० ००. - 1 भ ; : + ; ५४५ 4 न न : : ५ 9 , ५4 -4 नि -.=१,५-३. 29. धिह 1 ५,१०.५ ०१०१९. ०"
    confidence = 88.6
    result = evaluate_ocr_quality(fragmented_text, confidence)

    assert result["status"] == "low_confidence"
    assert result["is_usable"] is False
    assert any("fragmented" in r.lower() for r in result["reasons"])


def test_legitimate_short_verse_fragments_accepted():
    """Requirement 7: Legitimate short Sanskrit verses or fragments are not rejected solely because of length."""
    # Short Mahavakya (14 chars)
    res1 = evaluate_ocr_quality("अहं ब्रह्मास्मि", 85.0)
    assert res1["status"] == "completed"
    assert res1["is_usable"] is True

    # Short national motto (12 chars)
    res2 = evaluate_ocr_quality("सत्यमेव जयते", 88.0)
    assert res2["status"] == "completed"
    assert res2["is_usable"] is True

    # Short Upanishadic Mahavakya (9 chars)
    res3 = evaluate_ocr_quality("तत्त्वमसि", 80.0)
    assert res3["status"] == "completed"
    assert res3["is_usable"] is True


def test_uncommon_vocabulary_not_rejected():
    """Requirement 8: Valid Sanskrit manuscript with uncommon vocabulary is not rejected by grammar/dictionary checks."""
    # Aryabhatiya astronomical table verse using rare alpha-syllabic numeration
    astronomical_text = "मखि भखि फखि धखि णखि"
    confidence = 78.2
    result = evaluate_ocr_quality(astronomical_text, confidence)

    assert result["status"] == "completed"
    assert result["is_usable"] is True
    assert result["reasons"] == []


# ---------------------------------------------------------------------------
# Integration Tests for Flask Upload Pipeline & Status Assignment
# ---------------------------------------------------------------------------

def test_pipeline_clear_sanskrit_saved_completed(client, sample_image_path):
    """End-to-End: Clear Sanskrit upload produces 'completed' status in database and view."""
    with open(sample_image_path, "rb") as f:
        img_bytes = f.read()

    data = {
        "title": "Gita Chapter 1 Verse 1",
        "image": (io.BytesIO(img_bytes), "sample.png")
    }

    with patch("app.validate_source_image", return_value=(True, "Accepted")), \
         patch("app.extract_text", return_value=("धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः", 91.0)), \
         patch("app.translate_text", return_value="Assembled on the sacred field of Kurukshetra..."), \
         patch("app.generate_explanation", return_value="The opening verse of the Bhagavad Gita..."), \
         patch("app.extract_keywords", return_value="Dharma, Kurukshetra, Gita"), \
         patch("app.classify_content", return_value="Philosophy"):

        response = client.post("/upload", data=data, follow_redirects=True, content_type="multipart/form-data")
        assert response.status_code == 200
        assert "धर्मक्षेत्रे कुरुक्षेत्रे".encode("utf-8") in response.data

        # Verify database record status
        recent = list_manuscripts(limit=1)
        assert len(recent) > 0
        assert recent[0].status == "completed"


def test_pipeline_hindi_saved_completed(client, sample_image_path):
    """End-to-End: Genuine Hindi Devanagari manuscript produces 'completed' status in database."""
    with open(sample_image_path, "rb") as f:
        img_bytes = f.read()

    data = {
        "title": "Kabir Dohe Manuscript",
        "image": (io.BytesIO(img_bytes), "hindi_manu.png")
    }

    with patch("app.validate_source_image", return_value=(True, "Accepted")), \
         patch("app.extract_text", return_value=("बुरा जो देखन मैं चला बुरा न मिलिया कोय", 86.5)), \
         patch("app.translate_text", return_value="I went searching for bad people but found none..."), \
         patch("app.generate_explanation", return_value="A famous verse by sant Kabir..."), \
         patch("app.extract_keywords", return_value="Kabir, Dohe, Bhakti"), \
         patch("app.classify_content", return_value="Literature"):

        response = client.post("/upload", data=data, follow_redirects=True, content_type="multipart/form-data")
        assert response.status_code == 200

        recent = list_manuscripts(limit=1)
        assert len(recent) > 0
        assert recent[0].status == "completed"


def test_pipeline_low_confidence_saved_needs_review(client, sample_image_path):
    """End-to-End: Low confidence (<40%) manuscript is saved as 'low_confidence' (Needs Review)."""
    with open(sample_image_path, "rb") as f:
        img_bytes = f.read()

    data = {
        "title": "Faded Fragment",
        "image": (io.BytesIO(img_bytes), "faded.png")
    }

    with patch("app.validate_source_image", return_value=(True, "Accepted")), \
         patch("app.extract_text", return_value=("अथ योगानुशासनम्", 34.0)), \
         patch("app.translate_text", return_value="Now begins the instruction of Yoga."), \
         patch("app.generate_explanation", return_value="Patanjali Yoga Sutra 1.1..."), \
         patch("app.extract_keywords", return_value="Yoga, Patanjali"), \
         patch("app.classify_content", return_value="Yoga & Spirituality"):

        response = client.post("/upload", data=data, follow_redirects=True, content_type="multipart/form-data")
        assert response.status_code == 200

        recent = list_manuscripts(limit=1)
        assert len(recent) > 0
        assert recent[0].status == "low_confidence"


def test_pipeline_empty_ocr_saved_needs_review(client, sample_image_path):
    """Requirement 4: Empty OCR output after valid source image is marked 'low_confidence'."""
    with open(sample_image_path, "rb") as f:
        img_bytes = f.read()

    data = {
        "title": "Unreadable Manuscript",
        "image": (io.BytesIO(img_bytes), "unreadable.png")
    }

    with patch("app.validate_source_image", return_value=(True, "Accepted")), \
         patch("app.extract_text", return_value=("", 0.0)), \
         patch("app.translate_text", return_value="Unable to translate unreadable text."), \
         patch("app.generate_explanation", return_value="No context available."), \
         patch("app.extract_keywords", return_value="Unreadable"), \
         patch("app.classify_content", return_value="General/Other"):

        response = client.post("/upload", data=data, follow_redirects=True, content_type="multipart/form-data")
        assert response.status_code == 200

        recent = list_manuscripts(limit=1)
        assert len(recent) > 0
        assert recent[0].status == "low_confidence"


def test_pipeline_high_conf_garbage_saved_needs_review(client, sample_image_path):
    """End-to-End: High-confidence repeated garbage is tagged 'low_confidence' in database."""
    with open(sample_image_path, "rb") as f:
        img_bytes = f.read()

    data = {
        "title": "Corrupted Scan",
        "image": (io.BytesIO(img_bytes), "corrupted.png")
    }

    with patch("app.validate_source_image", return_value=(True, "Accepted")), \
         patch("app.extract_text", return_value=("ककककककककककककककक", 94.0)), \
         patch("app.translate_text", return_value="Corrupted output"), \
         patch("app.generate_explanation", return_value="Repetitive artifact"), \
         patch("app.extract_keywords", return_value="Artifact"), \
         patch("app.classify_content", return_value="General/Other"):

        response = client.post("/upload", data=data, follow_redirects=True, content_type="multipart/form-data")
        assert response.status_code == 200

        recent = list_manuscripts(limit=1)
        assert len(recent) > 0
        assert recent[0].status == "low_confidence"


def test_blank_image_rejected_at_validation(client):
    """Requirement 9 & 10: Blank image is rejected at source-image validation with exact message."""
    img = Image.new("RGB", (200, 200), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    data = {
        "title": "Blank Test Image",
        "image": (buf, "blank.png")
    }

    with patch("ocr.validate_with_gemini_vision", return_value=None):
        response = client.post("/upload", data=data, follow_redirects=True, content_type="multipart/form-data")
        assert response.status_code == 200
        assert b"No Sanskrit text detected." in response.data

        # Verify NO new record was inserted in database
        assert len(list_manuscripts()) == 0


def test_unrelated_language_rejected_at_validation(client):
    """Requirement 9 & 10: Unrelated language image is rejected with exact message and stops downstream."""
    with patch("app.validate_source_image", return_value=(False, "No Sanskrit text detected.")):
        data = {
            "title": "English Document",
            "image": (io.BytesIO(b"fake image bytes"), "english.png")
        }
        response = client.post("/upload", data=data, follow_redirects=True, content_type="multipart/form-data")
        assert response.status_code == 200
        assert b"No Sanskrit text detected." in response.data

        # Verify NO new record was inserted in database
        assert len(list_manuscripts()) == 0


def test_repository_status_filtering_compatibility(client, sample_manuscripts):
    """Requirement 12: Existing database records, listing, and search filters remain compatible."""
    # Completed filter returns m1 (Rigveda) and m3 (Aryabhatiya)
    res_completed = client.get("/manuscripts?status=completed")
    assert res_completed.status_code == 200
    assert b"Rigveda Creation Hymn" in res_completed.data
    assert b"Charaka Samhita Tridosha" not in res_completed.data

    # Needs Review (low_confidence) filter returns m2 (Charaka Samhita)
    res_review = client.get("/manuscripts?status=low_confidence")
    assert res_review.status_code == 200
    assert b"Charaka Samhita Tridosha" in res_review.data
    assert b"Rigveda Creation Hymn" not in res_review.data


def test_damaged_manuscript_fragmented_single_chars_needs_review():
    """Requirement: Damaged/faded manuscript with unusable fragmented single-character OCR is marked Needs Review."""
    # Replicates real OCR output from faded/damaged manuscript (e.g. Screenshot_2026-10-10_201527.png)
    damaged_ocr_text = "॥ म 1 न न : : ५ 9 , ५4 -4 नि -.=१,५-३. 29. धिह 1 ५,१०.५ ०१०१९."
    confidence = 78.5  # Modest or high raw confidence, but unusable text
    result = evaluate_ocr_quality(damaged_ocr_text, confidence)

    assert result["status"] == "low_confidence"
    assert result["is_usable"] is False
    assert any("fragment" in r.lower() for r in result["reasons"])


def test_displayed_status_agrees_with_database_in_results_view(client, sample_manuscripts):
    """Requirement: The database status and displayed status agree in the manuscript results view."""
    # m1 is 'completed'
    res_completed = client.get(f"/manuscript/{sample_manuscripts[0]}")
    assert res_completed.status_code == 200
    html_completed = res_completed.data.decode("utf-8")
    assert "badge-confidence-high" in html_completed
    assert "Completed" in html_completed
    assert "alert-low-confidence" not in html_completed

    # m2 is 'low_confidence'
    res_review = client.get(f"/manuscript/{sample_manuscripts[1]}")
    assert res_review.status_code == 200
    html_review = res_review.data.decode("utf-8")
    assert "badge-confidence-low" in html_review
    assert "Needs Review" in html_review
    assert "alert-low-confidence" in html_review


def test_displayed_status_agrees_with_database_in_repository_view(client, sample_manuscripts):
    """Requirement: The database status and displayed status agree on repository cards."""
    res = client.get("/manuscripts")
    assert res.status_code == 200
    html = res.data.decode("utf-8")
    assert "Needs Review" in html
    assert "Completed" in html


def test_pipeline_damaged_faded_manuscript_saved_needs_review(client, sample_image_path):
    """End-to-End: Damaged manuscript with fragmented OCR is saved as low_confidence and displayed as Needs Review."""
    with open(sample_image_path, "rb") as f:
        img_bytes = f.read()

    data = {
        "title": "Damaged Birch Bark Manuscript",
        "image": (io.BytesIO(img_bytes), "damaged.png")
    }

    damaged_ocr = "॥ म 1 न न : : ५ 9 , ५4 -4 नि -.=१,५-३. 29. धिह 1 ५,१०.५"
    with patch("app.validate_source_image", return_value=(True, "Accepted")), \
         patch("app.extract_text", return_value=(damaged_ocr, 75.0)), \
         patch("app.translate_text", return_value="Fragmented untranslatable text"), \
         patch("app.generate_explanation", return_value="Severely weathered manuscript"), \
         patch("app.extract_keywords", return_value="Fragment"), \
         patch("app.classify_content", return_value="General/Other"):

        response = client.post("/upload", data=data, follow_redirects=True, content_type="multipart/form-data")
        assert response.status_code == 200

        # Check HTML shows Needs Review
        html = response.data.decode("utf-8")
        assert "Needs Review" in html
        assert "alert-low-confidence" in html

        # Check database record
        recent = list_manuscripts(limit=1)
        assert len(recent) > 0
        assert recent[0].status == "low_confidence"


# ---------------------------------------------------------------------------
# Additional Multi-Signal OCR Quality Regression Tests
# ---------------------------------------------------------------------------

def test_visible_multiline_manuscript_single_line_ocr_needs_review(tmp_path):
    """Multi-line manuscript with only 1 extracted line is marked Needs Review (low_confidence)."""
    # Create synthetic multi-line manuscript image (e.g. 8 visible horizontal lines)
    h, w = 400, 300
    img = np.ones((h, w, 3), dtype=np.uint8) * 220
    import cv2
    for y in range(40, 360, 40):
        cv2.line(img, (20, y), (280, y), (40, 30, 20), 4)

    img_path = str(tmp_path / "multiline_manu.png")
    cv2.imwrite(img_path, img)

    # OCR only transcribed a single title line with high confidence
    single_line_ocr = "महाराज जसवंत सिंहजी"
    confidence = 98.5

    result = evaluate_ocr_quality(single_line_ocr, confidence, image_input=img_path)

    assert result["status"] == "low_confidence"
    assert result["is_usable"] is False
    assert any("line" in r.lower() or "incomplete" in r.lower() for r in result["reasons"])
    assert result["metrics"]["visible_lines"] is not None
    assert result["metrics"]["visible_lines"] >= 4
    assert result["metrics"]["ocr_lines"] == 1


def test_faded_manuscript_avagraha_repetition_needs_review():
    """Faded manuscript with excessive avagraha placeholders is marked Needs Review."""
    # Faded OCR output with hallucinated avagraha gaps
    faded_ocr = "०१। ऽऽति ॥ ऽऽतिः ॥ ०२। ऽऽतंमन्यैऽसिननिरस तघ्रथयिसामयः ॥"
    confidence = 92.0
    result = evaluate_ocr_quality(faded_ocr, confidence)

    assert result["status"] == "low_confidence"
    assert result["is_usable"] is False
    assert any("avagraha" in r.lower() or "repetition" in r.lower() or "corrupt" in r.lower() for r in result["reasons"])


def test_spliced_punctuation_corrupted_tokens_needs_review():
    """Manuscript with punctuation spliced into Devanagari words is marked Needs Review."""
    corrupted_ocr = "लि.दिः तघ्रथयिसामयः -.=१,५-३. धेशतम्तर्सरीगेरींग्सददुरुमेवोरेव"
    confidence = 88.0
    result = evaluate_ocr_quality(corrupted_ocr, confidence)

    assert result["status"] == "low_confidence"
    assert result["is_usable"] is False
    assert any("corrupt" in r.lower() or "implausible" in r.lower() for r in result["reasons"])


def test_high_conf_incomplete_low_char_density_needs_review(tmp_path):
    """High-confidence OCR with very low character count relative to visible lines is marked Needs Review."""
    h, w = 400, 300
    img = np.ones((h, w, 3), dtype=np.uint8) * 220
    import cv2
    for y in range(40, 360, 40):
        cv2.line(img, (20, y), (280, y), (30, 20, 10), 3)

    img_path = str(tmp_path / "sparse_manu.png")
    cv2.imwrite(img_path, img)

    # Only 16 Devanagari characters extracted for ~8 visible lines
    sparse_ocr = "सिद्धम् ॥ स्वस्ति ॥"
    confidence = 96.0

    result = evaluate_ocr_quality(sparse_ocr, confidence, image_input=img_path)

    assert result["status"] == "low_confidence"
    assert result["is_usable"] is False
    assert any("incomplete" in r.lower() or "line" in r.lower() for r in result["reasons"])


def test_multi_section_collage_unwanted_latin_descriptors_needs_review():
    """Multi-section composite manuscript with Latin fragment labels is marked Needs Review."""
    text_with_labels = "शीर्षकम् Upper left fragment : ॐ ॥ ओंकारं आदि रूपेण सर्वस्कन्दस्य"
    confidence = 95.0
    result = evaluate_ocr_quality(text_with_labels, confidence)

    assert result["status"] == "low_confidence"
    assert result["is_usable"] is False
    assert any("latin" in r.lower() or "english" in r.lower() for r in result["reasons"])


def test_downstream_fluent_translation_does_not_overwrite_needs_review(client, sample_image_path):
    """Downstream fluent translation and explanation MUST NOT override a failed OCR quality check."""
    with open(sample_image_path, "rb") as f:
        img_bytes = f.read()

    data = {
        "title": "Maharaja Incomplete Record",
        "image": (io.BytesIO(img_bytes), "incomplete.png")
    }

    # Only 1 line extracted from multi-line manuscript
    incomplete_ocr = "महाराज जसवंत सिंहजी"
    # Downstream translation is fluent and natural
    fluent_translation = "His Highness Maharaja Jaswant Singh Ji, the ruler and patron."
    fluent_explanation = "This historical artifact details the administration of Maharaja Jaswant Singh."

    with patch("app.validate_source_image", return_value=(True, "Accepted")), \
         patch("app.extract_text", return_value=(incomplete_ocr, 98.5)), \
         patch("ocr.estimate_visible_text_lines", return_value=12), \
         patch("app.translate_text", return_value=fluent_translation), \
         patch("app.generate_explanation", return_value=fluent_explanation), \
         patch("app.extract_keywords", return_value="Jaswant Singh, History"), \
         patch("app.classify_content", return_value="Literature"):

        response = client.post("/upload", data=data, follow_redirects=True, content_type="multipart/form-data")
        assert response.status_code == 200

        # Database record must preserve low_confidence despite fluent translation
        recent = list_manuscripts(limit=1)
        assert len(recent) > 0
        assert recent[0].status == "low_confidence"
        assert recent[0].ocr_confidence == 98.5

        # View page must display Needs Review
        html = response.data.decode("utf-8")
        assert "Needs Review" in html
        assert "badge-confidence-low" in html
        assert "alert-low-confidence" in html


def test_palm_leaf_irregular_layout_line_detection_fallback():
    """Palm-leaf manuscript where line estimation returns 0 gracefully falls back to text quality signals."""
    # When line estimation fails or returns 0, clean text still passes
    valid_text = "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः । मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय ॥"
    result = evaluate_ocr_quality(valid_text, 88.0, image_input="non_existent_file.png")

    assert result["status"] == "completed"
    assert result["is_usable"] is True
    assert result["metrics"]["visible_lines"] == 0


def test_search_results_status_badge_consistency(client, sample_manuscripts):
    """Search results cards display consistent status badges matching the database."""
    # Search for all records
    res = client.get("/search/results?query=")
    assert res.status_code == 200
    html = res.data.decode("utf-8")

    # Should contain both Completed and Needs Review badges
    assert "badge-confidence-high" in html
    assert "badge-confidence-low" in html
    assert "Needs Review" in html
    assert "Completed" in html


def test_legitimate_short_sanskrit_phrase_with_image_marked_completed(tmp_path):
    """Verifies that legitimate, readable short Sanskrit phrases (e.g. यथा दृष्टि तथा सृष्टि।)
    with an image input are marked Completed when OCR quality is sufficient, and not falsely
    flagged as incomplete multi-line manuscripts."""
    # 1. Create a single-line manuscript image
    h, w = 300, 500
    img = np.ones((h, w, 3), dtype=np.uint8) * 230
    import cv2
    cv2.putText(img, "Yatha Drishti", (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (30, 20, 10), 2)
    img_path = str(tmp_path / "short_phrase.png")
    cv2.imwrite(img_path, img)

    # 2. Legitimate readable short Sanskrit phrase
    sanskrit_phrase = "यथा दृष्टि तथा सृष्टि।"
    confidence = 98.5

    result = evaluate_ocr_quality(sanskrit_phrase, confidence, image_input=img_path)

    assert result["status"] == "completed"
    assert result["is_usable"] is True
    assert result["reasons"] == []
    assert result["metrics"]["devanagari_ratio"] == 1.0
    assert result["metrics"]["corrupt_token_ratio"] == 0.0



