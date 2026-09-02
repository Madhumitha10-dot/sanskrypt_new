"""
SansKrypt Image Preprocessing Module
High-precision multi-stage enhancement tailored for ancient Sanskrit manuscripts:
1. Grayscale & illumination normalization
2. High-resolution bicubic stroke upscaling
3. Bilateral edge-preserving denoising
4. CLAHE adaptive contrast optimization
5. Adaptive Gaussian binarization & morphological ligature healing
"""
import os
import logging
from typing import Tuple, Optional, Dict
import cv2
import numpy as np

try:
    from config import PROCESSED_FOLDER
except ImportError:
    PROCESSED_FOLDER = os.path.join(os.path.dirname(__file__), "uploads", "processed")

logger = logging.getLogger(__name__)


def read_image_safely(image_path: str) -> np.ndarray:
    """
    Reads an image safely across all OS platforms and Unicode/Windows paths.

    Args:
        image_path: Path to the image file.

    Returns:
        numpy.ndarray: Decoded BGR image.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    with open(image_path, "rb") as f:
        file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
    
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to decode image file: {image_path}. File may be corrupt.")
    
    return image


def save_image_safely(image: np.ndarray, output_path: str) -> str:
    """
    Saves an image safely ensuring destination directories exist.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    ext = os.path.splitext(output_path)[1]
    if not ext:
        ext = ".png"
        output_path += ext

    is_success, buffer = cv2.imencode(ext, image)
    if not is_success:
        raise IOError(f"Failed to encode image for saving to: {output_path}")

    with open(output_path, "wb") as f:
        f.write(buffer)

    return os.path.abspath(output_path)


def flatten_background_illumination(gray_image: np.ndarray) -> np.ndarray:
    """
    Normalizes uneven lighting and aged parchment discoloration.
    """
    # Estimate background using morphological opening with large kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    background = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, kernel)
    # Divide original by background to remove shadows
    normalized = cv2.divide(gray_image, background, scale=255)
    return normalized


def generate_preprocessing_variants(bgr_image: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Generates multiple high-quality image variants to find the optimal OCR candidate:
    1. 'adaptive_binary': Upscaled + CLAHE + Adaptive Gaussian Threshold + Ligature Healing
    2. 'enhanced_grayscale': Upscaled + CLAHE + Contrast Normalization
    3. 'otsu_binary': Upscaled + CLAHE + Otsu Binarization
    """
    # Convert to grayscale
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    # Calculate optimal upscaling factor (ensure height >= 1000px for sharp Devanagari ligatures)
    scale_factor = 2.0
    if h < 800 or w < 800:
        scale_factor = 3.0
    elif h > 2400 or w > 2400:
        scale_factor = 1.0

    if scale_factor != 1.0:
        gray_scaled = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    else:
        gray_scaled = gray.copy()

    # Denoise with edge preservation
    denoised = cv2.bilateralFilter(gray_scaled, d=7, sigmaColor=50, sigmaSpace=50)

    # Enhance contrast with CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast_boosted = clahe.apply(denoised)

    # Variant 1: Adaptive Gaussian thresholding + Morphological healing
    blurred = cv2.GaussianBlur(contrast_boosted, (3, 3), 0)
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )
    # Connect broken strokes/shirorekha with subtle closing
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    adaptive_healed = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, morph_kernel)

    # Variant 2: Otsu thresholding
    _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return {
        "adaptive_binary": adaptive_healed,
        "enhanced_grayscale": contrast_boosted,
        "otsu_binary": otsu_thresh
    }


def preprocess_image(
    image_path: str,
    output_path: Optional[str] = None,
    apply_threshold: bool = True
) -> Tuple[np.ndarray, str]:
    """
    Main preprocessing pipeline function for manuscript images.

    Args:
        image_path: Path to the raw manuscript image.
        output_path: Destination path for enhanced image.
        apply_threshold: Whether to apply adaptive thresholding (default: True).

    Returns:
        Tuple[np.ndarray, str]: (Preprocessed image array, Saved output path)
    """
    logger.info(f"Preprocessing manuscript image: {image_path}")

    try:
        bgr_image = read_image_safely(image_path)
        variants = generate_preprocessing_variants(bgr_image)

        processed_image = variants["adaptive_binary"] if apply_threshold else variants["enhanced_grayscale"]

        if not output_path:
            base_name = os.path.basename(image_path)
            name, ext = os.path.splitext(base_name)
            output_filename = f"{name}_processed{ext if ext else '.png'}"
            output_path = os.path.join(PROCESSED_FOLDER, output_filename)

        saved_path = save_image_safely(processed_image, output_path)
        logger.info(f"Preprocessing completed. Saved enhanced manuscript to: {saved_path}")

        return processed_image, saved_path

    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}", exc_info=True)
        raise
