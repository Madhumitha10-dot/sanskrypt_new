"""
SansKrypt Database Module
Manages SQLite persistence using SQLAlchemy for manuscript records, OCR data, translations, and metadata.
"""
from datetime import datetime
import logging
from typing import List, Optional, Dict, Any

import flask.globals
if not hasattr(flask.globals, "app_ctx") or flask.globals.app_ctx is None:
    class _AppCtxProxy:
        def _get_current_object(self):
            top = getattr(flask.globals, "_app_ctx_stack", None)
            if top is not None and top.top is not None:
                return top.top
            return self
    flask.globals.app_ctx = _AppCtxProxy()

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine, desc
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session

try:
    from config import SQLALCHEMY_DATABASE_URI, SQLALCHEMY_TRACK_MODIFICATIONS
except ImportError:
    SQLALCHEMY_DATABASE_URI = "sqlite:///sanskrypt.db"
    SQLALCHEMY_TRACK_MODIFICATIONS = False

logger = logging.getLogger(__name__)

# Flask-SQLAlchemy instance for web app integration
db = SQLAlchemy()

# Standalone SQLAlchemy Base for direct/CLI/test use
Base = declarative_base()


class Manuscript(db.Model):
    """
    Manuscript table representing digitized Sanskrit manuscript artifacts.
    """
    __tablename__ = "manuscripts"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    title = db.Column(db.String(255), nullable=True)
    original_image_path = db.Column(db.String(512), nullable=False)
    processed_image_path = db.Column(db.String(512), nullable=True)
    ocr_text = db.Column(db.Text, nullable=False)
    ocr_confidence = db.Column(db.Float, nullable=False, default=0.0)
    translated_text = db.Column(db.Text, nullable=False)
    explanation = db.Column(db.Text, nullable=True)
    keywords = db.Column(db.Text, nullable=True)
    classification = db.Column(db.String(100), nullable=True, default="General/Other")
    status = db.Column(db.String(50), nullable=False, default="completed")  # 'completed', 'low_confidence', 'failed'
    upload_date = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    def to_dict(self) -> Dict[str, Any]:
        """Converts model instance to dictionary."""
        return {
            "id": self.id,
            "title": self.title or f"Manuscript #{self.id}",
            "original_image_path": self.original_image_path,
            "processed_image_path": self.processed_image_path,
            "ocr_text": self.ocr_text,
            "ocr_confidence": round(self.ocr_confidence, 2) if self.ocr_confidence is not None else 0.0,
            "translated_text": self.translated_text,
            "explanation": self.explanation,
            "keywords": self.keywords,
            "classification": self.classification,
            "status": self.status,
            "upload_date": self.upload_date.strftime("%Y-%m-%d %H:%M:%S") if self.upload_date else ""
        }

    @property
    def original_filename(self) -> str:
        if not self.original_image_path:
            return ""
        norm = self.original_image_path.replace("\\", "/")
        if "/uploads/" in norm:
            return norm.split("/uploads/")[-1]
        elif norm.startswith("uploads/"):
            return norm.split("uploads/", 1)[-1]
        import os
        return os.path.basename(self.original_image_path)

    @property
    def processed_filename(self) -> str:
        if not self.processed_image_path:
            return ""
        norm = self.processed_image_path.replace("\\", "/")
        if "/uploads/" in norm:
            return norm.split("/uploads/")[-1]
        elif norm.startswith("uploads/"):
            return norm.split("uploads/", 1)[-1]
        import os
        return os.path.basename(self.processed_image_path)

    def __repr__(self) -> str:
        return f"<Manuscript {self.id}: {self.title or 'Untitled'} [{self.status}] ({self.ocr_confidence}%)>"


def init_db(app=None):
    """Initializes the database schema with the Flask app."""
    if app is not None:
        db.init_app(app)
        with app.app_context():
            db.create_all()
            logger.info("Database initialized with Flask app context.")
    else:
        engine = create_engine(SQLALCHEMY_DATABASE_URI)
        Manuscript.__table__.create(bind=engine, checkfirst=True)
        logger.info("Database initialized standalone.")


def save_manuscript(
    original_image_path: str,
    ocr_text: str,
    ocr_confidence: float,
    translated_text: str,
    explanation: str,
    keywords: str,
    classification: str,
    status: str = "completed",
    processed_image_path: Optional[str] = None,
    title: Optional[str] = None
) -> Manuscript:
    """
    Saves a processed manuscript artifact into the database.

    Returns:
        Manuscript: The newly created database record.
    """
    try:
        manuscript = Manuscript(
            title=title,
            original_image_path=original_image_path,
            processed_image_path=processed_image_path,
            ocr_text=ocr_text,
            ocr_confidence=float(ocr_confidence),
            translated_text=translated_text,
            explanation=explanation,
            keywords=keywords,
            classification=classification,
            status=status,
            upload_date=datetime.utcnow()
        )
        db.session.add(manuscript)
        db.session.commit()
        logger.info(f"Saved manuscript ID: {manuscript.id} (Status: {status}, Confidence: {ocr_confidence}%)")
        return manuscript
    except Exception as e:
        db.session.rollback()
        logger.error(f"Failed to save manuscript to DB: {str(e)}", exc_info=True)
        raise


def get_manuscript(manuscript_id: int) -> Optional[Manuscript]:
    """
    Retrieves a single manuscript by its primary key ID.
    """
    try:
        return db.session.get(Manuscript, manuscript_id)
    except Exception as e:
        logger.error(f"Error fetching manuscript {manuscript_id}: {str(e)}")
        return None


def list_manuscripts(
    status: Optional[str] = None,
    classification: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
) -> List[Manuscript]:
    """
    Lists manuscripts with optional filtering by status and classification.
    """
    try:
        query = Manuscript.query

        if status and status.lower() != "all":
            query = query.filter(Manuscript.status == status)

        if classification and classification.lower() != "all":
            query = query.filter(Manuscript.classification == classification)

        return query.order_by(desc(Manuscript.upload_date)).offset(offset).limit(limit).all()
    except Exception as e:
        logger.error(f"Error listing manuscripts: {str(e)}")
        return []


def update_manuscript(manuscript_id: int, **kwargs) -> Optional[Manuscript]:
    """
    Updates fields of an existing manuscript.
    """
    try:
        manuscript = db.session.get(Manuscript, manuscript_id)
        if not manuscript:
            return None

        for key, value in kwargs.items():
            if hasattr(manuscript, key):
                setattr(manuscript, key, value)

        db.session.commit()
        logger.info(f"Updated manuscript {manuscript_id}")
        return manuscript
    except Exception as e:
        db.session.rollback()
        logger.error(f"Failed to update manuscript {manuscript_id}: {str(e)}")
        raise


def delete_manuscript(manuscript_id: int) -> bool:
    """
    Deletes a manuscript from the database.
    """
    try:
        manuscript = db.session.get(Manuscript, manuscript_id)
        if not manuscript:
            return False

        db.session.delete(manuscript)
        db.session.commit()
        logger.info(f"Deleted manuscript {manuscript_id}")
        return True
    except Exception as e:
        db.session.rollback()
        logger.error(f"Failed to delete manuscript {manuscript_id}: {str(e)}")
        raise
