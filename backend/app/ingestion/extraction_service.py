from pathlib import Path

from sqlalchemy.orm import Session

from app.config.settings import settings
from app.ingestion.cleaner import clean_text
from app.ingestion.loaders import extract_text_from_file
from app.models.document import Document
from app.services.document_service import update_document_status


def extract_and_store_document_text(
    db: Session,
    document: Document,
) -> dict:
    try:
        update_document_status(
            db=db,
            document_id=document.id,
            status="processing",
        )

        raw_file_path = Path(document.storage_path)

        if not raw_file_path.exists():
            raise FileNotFoundError(
                f"Raw document file not found: {document.storage_path}"
            )

        file_extension = raw_file_path.suffix.lower()

        extracted_text = extract_text_from_file(
            file_path=str(raw_file_path),
            content_type=document.content_type,
            file_extension=file_extension,
        )

        cleaned_text = clean_text(extracted_text)

        output_dir = (
            Path(settings.EXTRACTED_TEXT_DIR)
            / document.user_id
            / document.id
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        extracted_text_path = output_dir / "extracted_text.txt"
        extracted_text_path.write_text(cleaned_text, encoding="utf-8")

        updated_document = update_document_status(
            db=db,
            document_id=document.id,
            status="extracted",
        )

        return {
            "document_id": document.id,
            "user_id": document.user_id,
            "status": updated_document.status if updated_document else "extracted",
            "extracted_text_path": str(extracted_text_path),
            "character_count": len(cleaned_text),
            "message": "Document text extracted successfully.",
        }

    except Exception as exc:
        update_document_status(
            db=db,
            document_id=document.id,
            status="failed",
        )

        return {
            "document_id": document.id,
            "user_id": document.user_id,
            "status": "failed",
            "extracted_text_path": "",
            "character_count": 0,
            "message": f"Document text extraction failed: {str(exc)}",
        }