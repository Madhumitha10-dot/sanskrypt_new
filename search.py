"""
SansKrypt Search Module
Provides multi-criteria search capabilities across Sanskrit text, English translations,
keywords, domain classifications, and OCR confidence/status filters.
"""
import logging
from typing import List, Optional
from sqlalchemy import or_, and_, desc

from database import Manuscript

logger = logging.getLogger(__name__)


def search_by_keyword(keyword: str, status: Optional[str] = None) -> List[Manuscript]:
    """
    Searches manuscripts matching a specific keyword in the extracted keywords column.

    Args:
        keyword: Keyword term to match.
        status: Optional status filter ('completed', 'low_confidence', etc.).

    Returns:
        List[Manuscript]: Matching manuscript records.
    """
    if not keyword or not keyword.strip():
        return []

    term = f"%{keyword.strip()}%"
    query = Manuscript.query.filter(Manuscript.keywords.ilike(term))

    if status and status.lower() != "all":
        query = query.filter(Manuscript.status == status)

    return query.order_by(desc(Manuscript.upload_date)).all()


def search_by_classification(classification: str, status: Optional[str] = None) -> List[Manuscript]:
    """
    Searches manuscripts within a specific knowledge classification/domain.

    Args:
        classification: Domain category (e.g. 'Philosophy', 'Ayurveda').
        status: Optional status filter.

    Returns:
        List[Manuscript]: Matching manuscript records.
    """
    if not classification or classification.strip().lower() == "all":
        query = Manuscript.query
    else:
        query = Manuscript.query.filter(Manuscript.classification == classification.strip())

    if status and status.lower() != "all":
        query = query.filter(Manuscript.status == status)

    return query.order_by(desc(Manuscript.upload_date)).all()


def search_by_text(query_text: str, status: Optional[str] = None) -> List[Manuscript]:
    """
    Full-text search across translated English text, original Sanskrit OCR text, and explanations.

    Args:
        query_text: Text query to search.
        status: Optional status filter.

    Returns:
        List[Manuscript]: Matching manuscript records.
    """
    if not query_text or not query_text.strip():
        return []

    term = f"%{query_text.strip()}%"
    query = Manuscript.query.filter(
        or_(
            Manuscript.translated_text.ilike(term),
            Manuscript.ocr_text.ilike(term),
            Manuscript.explanation.ilike(term),
            Manuscript.title.ilike(term)
        )
    )

    if status and status.lower() != "all":
        query = query.filter(Manuscript.status == status)

    return query.order_by(desc(Manuscript.upload_date)).all()


def search_manuscripts(
    query: Optional[str] = None,
    keyword: Optional[str] = None,
    classification: Optional[str] = None,
    status: Optional[str] = None,
    min_confidence: Optional[float] = None
) -> List[Manuscript]:
    """
    Unified multi-criteria search combining free text, keywords, domain classification,
    and status/confidence filters.

    Args:
        query: Free text search across title, translation, Sanskrit OCR, and explanation.
        keyword: Keyword search filter.
        classification: Knowledge domain filter.
        status: Status filter ('completed', 'low_confidence', etc.).
        min_confidence: Minimum OCR confidence filter (0-100).

    Returns:
        List[Manuscript]: Matching manuscript records.
    """
    db_query = Manuscript.query
    conditions = []

    # Text search
    if query and query.strip():
        term = f"%{query.strip()}%"
        conditions.append(
            or_(
                Manuscript.translated_text.ilike(term),
                Manuscript.ocr_text.ilike(term),
                Manuscript.explanation.ilike(term),
                Manuscript.title.ilike(term),
                Manuscript.keywords.ilike(term)
            )
        )

    # Keyword filter
    if keyword and keyword.strip():
        kw_term = f"%{keyword.strip()}%"
        conditions.append(Manuscript.keywords.ilike(kw_term))

    # Classification filter
    if classification and classification.strip().lower() != "all":
        conditions.append(Manuscript.classification == classification.strip())

    # Status filter (crucial for finding low_confidence reviews)
    if status and status.strip().lower() != "all":
        conditions.append(Manuscript.status == status.strip())

    # Minimum confidence filter
    if min_confidence is not None:
        try:
            val = float(min_confidence)
            conditions.append(Manuscript.ocr_confidence >= val)
        except (ValueError, TypeError):
            pass

    if conditions:
        db_query = db_query.filter(and_(*conditions))

    return db_query.order_by(desc(Manuscript.upload_date)).all()
