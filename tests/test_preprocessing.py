"""
Unit tests for image preprocessing module.
"""
import os
import pytest
import numpy as np
from preprocessing import preprocess_image, read_image_safely, save_image_safely


def test_read_image_safely(sample_image_path):
    """Verifies safe image loading with valid paths."""
    image = read_image_safely(sample_image_path)
    assert isinstance(image, np.ndarray)
    assert image.ndim == 3  # BGR


def test_read_image_non_existent():
    """Verifies FileNotFoundError on invalid path."""
    with pytest.raises(FileNotFoundError):
        read_image_safely("non_existent_file_path_12345.png")


def test_save_image_safely(tmp_path):
    """Verifies safe image saving."""
    img = np.ones((50, 50), dtype=np.uint8) * 128
    out_path = str(tmp_path / "saved_test.png")
    result_path = save_image_safely(img, out_path)
    assert os.path.exists(result_path)


def test_preprocess_image(sample_image_path, tmp_path):
    """Verifies multi-stage preprocessing output format and dimensions."""
    output_path = str(tmp_path / "proc_out.png")
    proc_img, saved_path = preprocess_image(sample_image_path, output_path=output_path)

    assert isinstance(proc_img, np.ndarray)
    assert proc_img.ndim == 2  # Grayscale/Binarized
    assert os.path.exists(saved_path)
    assert os.path.getsize(saved_path) > 0
